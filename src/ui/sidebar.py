"""사이드바 필터 → FilterState.

모든 위젯에 즉시 사용 가능한 기본값을 준다. 로그인·초기 설정·파일 업로드 같은
관문을 두지 않으므로 접속 즉시 전체 대시보드가 조작 가능하다.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from config import settings as S
from src.ui import components as C


@dataclass(frozen=True)
class FilterState:
    period: str
    period_label: str
    period_badge: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_days: int
    age_bands: tuple[str, ...]
    gender: str
    channels: tuple[str, ...]
    venues: tuple[str, ...]
    tiers: tuple[str, ...]
    exclude_comp: bool
    exclude_month_first: bool
    inflow_weights: dict[str, float] = field(default_factory=dict)
    vts_weights: dict[str, float] = field(default_factory=dict)
    max_lag: int = S.MAX_LAG_DEFAULT
    radius_km: float = S.DEFAULT_RADIUS_KM

    def slice_sales(self, sales: pd.DataFrame) -> pd.DataFrame:
        """현재 필터를 판매 데이터에 적용한다.

        채널·티어를 전부 해제하면 '필터 없음'이 아니라 '선택 없음'으로 본다.
        아무것도 선택하지 않았는데 전체를 보여주면 KPI 수치가 화면의 선택 상태와
        어긋나 오해를 부른다 — 뷰가 안내를 띄우도록 빈 결과를 돌려준다.
        """
        if sales.empty:
            return sales
        if not self.channels or not self.tiers:
            return sales.iloc[0:0]
        out = sales.loc[sales["date"].between(self.start, self.end)]
        out = out.loc[out["channel"].isin(list(self.channels))]
        out = out.loc[out["tier"].isin(list(self.tiers))]
        if self.venues:
            # 영업장 필터는 카지노 식음에만 의미가 있다 (다른 채널은 영업장 1개).
            # 다른 채널 행을 통째로 날리지 않도록 채널을 구분해 적용한다.
            multi = out["channel"].isin(list(S.MULTI_VENUE_CHANNELS))
            out = out.loc[~multi | out["venue"].isin(list(self.venues))]
        if self.exclude_comp:
            out = out.loc[~out["is_comp"]]
        if self.exclude_month_first:
            out = out.loc[~out["is_month_first"]]
        return out

    def slice_ars(self, ars: pd.DataFrame) -> pd.DataFrame:
        if ars.empty:
            return ars
        return ars.loc[ars["date"].between(self.start, self.end)]

    def previous_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """직전 동일 길이 구간 — KPI 델타 비교용."""
        span = self.end - self.start
        prev_end = self.start - pd.Timedelta(days=1)
        return prev_end - span, prev_end


def _period_options(ars: pd.DataFrame, sales: pd.DataFrame) -> list[str]:
    """데이터가 실제로 존재하는 구간만 노출한다."""
    options: list[str] = []
    for key, meta in S.PERIODS.items():
        start = pd.Timestamp(meta["start"])
        end = pd.Timestamp(meta["end"])
        has_ars = not ars.empty and ars["date"].between(start, end).any()
        has_sales = not sales.empty and sales["date"].between(start, end).any()
        if has_ars or has_sales:
            options.append(key)
    return options or [S.DEFAULT_PERIOD]


def render(data: dict, sources: dict) -> FilterState:
    ars: pd.DataFrame = data.get("ars", pd.DataFrame())
    sales: pd.DataFrame = data.get("sales", pd.DataFrame())

    with st.sidebar:
        st.markdown(
            '<div class="kl-side-title">강원랜드 VIP 교차판매</div>'
            '<div class="kl-side-sub">카지노 → 리조트 고마진 채널 전환</div>',
            unsafe_allow_html=True,
        )

        # ── 분석 구간 ─────────────────────────────────────────────────
        st.markdown("###### 분석 구간")
        options = _period_options(ars, sales)
        labels = {k: S.PERIODS[k]["label"] for k in options}
        labels[S.PERIOD_CUSTOM] = "직접 지정"
        choices = options + [S.PERIOD_CUSTOM]
        period = st.radio(
            "구간", choices, index=0, format_func=lambda k: labels[k],
            label_visibility="collapsed", key="flt_period",
        )

        all_dates = pd.concat([
            ars["date"] if not ars.empty else pd.Series(dtype="datetime64[ns]"),
            sales["date"] if not sales.empty else pd.Series(dtype="datetime64[ns]"),
        ])
        data_min = all_dates.min() if not all_dates.empty else pd.Timestamp("2024-01-01")
        data_max = all_dates.max() if not all_dates.empty else pd.Timestamp("2024-12-31")

        if period == S.PERIOD_CUSTOM:
            picked = st.date_input(
                "기간", value=(data_min.date(), data_max.date()),
                min_value=data_min.date(), max_value=data_max.date(),
                key="flt_dates",
            )
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
            else:
                start, end = data_min, data_max
            period_label, period_badge = "직접 지정", "직접 지정"
        else:
            meta = S.PERIODS[period]
            start, end = pd.Timestamp(meta["start"]), pd.Timestamp(meta["end"])
            period_label, period_badge = meta["label"], meta["badge"]

        # ── VIP 세그먼트 ──────────────────────────────────────────────
        st.markdown("###### VIP 세그먼트")
        demo: pd.DataFrame = data.get("demo", pd.DataFrame())
        band_options = (
            [b for b in S.AGE_BANDS_ALL if b in set(demo["age_band"])]
            if not demo.empty and "age_band" in demo.columns else list(S.AGE_BANDS_ALL)
        )
        extra = (
            [b for b in demo["age_band"].unique() if b not in band_options]
            if not demo.empty and "age_band" in demo.columns else []
        )
        band_options = band_options + sorted(str(b) for b in extra)
        default_bands = [b for b in S.AGE_BANDS_VIP if b in band_options] or band_options
        age_bands = st.multiselect("연령대", band_options, default=default_bands,
                                   key="flt_age")
        gender = st.radio("성별", S.GENDER_OPTIONS, index=0, horizontal=True,
                          key="flt_gender")

        # ── 채널 / 상품 ───────────────────────────────────────────────
        st.markdown("###### 채널 · 상품")
        available_channels = (
            [c for c in S.CHANNELS if c in set(sales["channel"])]
            if not sales.empty else list(S.CHANNELS)
        )
        channels = st.multiselect(
            "채널", available_channels, default=available_channels,
            format_func=lambda c: S.CHANNEL_LABEL.get(c, c), key="flt_channels",
        )

        # 영업장 필터는 카지노 식음(영업장 5개)에만 의미가 있다
        venue_pool: list[str] = []
        if not sales.empty:
            multi = sales.loc[
                sales["channel"].isin(list(S.MULTI_VENUE_CHANNELS))
                & sales["channel"].isin(channels or available_channels)
            ]
            venue_pool = sorted(str(v) for v in multi["venue"].unique())
        if venue_pool:
            venues = st.multiselect(
                "영업장 (카지노 식음)", venue_pool, default=venue_pool,
                key="flt_venues",
                help="룸서비스·지역특산품은 상시 영업장이 1개뿐이라 필터가 적용되지 않습니다.",
            )
        else:
            venues = []
            st.caption("선택한 채널에는 영업장이 1개뿐입니다.")

        tiers = st.multiselect(
            "상품 티어", list(S.TIERS), default=list(S.TIERS),
            format_func=lambda t: S.TIER_LABEL.get(t, t), key="flt_tiers",
            help="프리미엄=명품·한우·와인 등 / 선물번들=세트·패키지 / 일반=그 외",
        )
        exclude_comp = st.toggle(
            "컴프((V)) 상품 제외", value=False, key="flt_comp",
            help="'(V)에비앙' 등 무상 제공 추정 상품을 제외합니다 (전체 수량의 약 0.6%).",
        )

        # ── 고급 설정 ─────────────────────────────────────────────────
        with st.expander("고급 설정", expanded=False):
            st.caption("유입 강도 지수(CII) 가중치")
            w_demand = st.slider("실제 입장 전환", 0.0, 1.0,
                                 S.INFLOW_WEIGHTS["demand"], 0.05, key="w_demand")
            w_pressure = st.slider("예약 수요 압력", 0.0, 1.0,
                                   S.INFLOW_WEIGHTS["pressure"], 0.05, key="w_pressure")
            w_convert = st.slider("구매 전환율", 0.0, 1.0,
                                  S.INFLOW_WEIGHTS["convert"], 0.05, key="w_convert")

            st.caption("VIP 타겟 지수(VTS) 가중치")
            v_inflow = st.slider("유입", 0.0, 1.0, S.VTS_WEIGHTS["inflow"], 0.05,
                                 key="v_inflow")
            v_base = st.slider("VIP 모수", 0.0, 1.0, S.VTS_WEIGHTS["base"], 0.05,
                               key="v_base")
            v_proven = st.slider("실증 소비", 0.0, 1.0, S.VTS_WEIGHTS["proven"], 0.05,
                                 key="v_proven")
            v_headroom = st.slider("미달 기회", 0.0, 1.0, S.VTS_WEIGHTS["headroom"],
                                   0.05, key="v_headroom")

            st.caption("이상치 · 분석 범위")
            exclude_month_first = st.toggle(
                "월초 1일 이상치 제외", value=False, key="flt_mfirst",
                help="카지노 식음 판매량이 매월 1일에 약 1.6배로 튄다 (집계 아티팩트 의심).",
            )
            max_lag = st.slider("래그 최대 시차 (일)", 1, 5, S.MAX_LAG_DEFAULT, 1,
                                key="flt_lag")
            radius_km = st.slider("가맹점 반경 (km)", 1.0, 30.0,
                                  S.DEFAULT_RADIUS_KM, 1.0, key="flt_radius")

            if st.button("필터 초기화", width="stretch"):
                for key in list(st.session_state.keys()):
                    if str(key).startswith(("flt_", "w_", "v_")):
                        del st.session_state[key]
                st.rerun()

        # ── 데이터 출처 ───────────────────────────────────────────────
        st.markdown("###### 데이터 출처")
        C.source_badges(sources)

    n_days = int((end - start).days) + 1
    return FilterState(
        period=period,
        period_label=period_label,
        period_badge=period_badge,
        start=start,
        end=end,
        n_days=max(n_days, 0),
        age_bands=tuple(age_bands),
        gender=gender,
        channels=tuple(channels),
        venues=tuple(venues),
        tiers=tuple(tiers),
        exclude_comp=bool(exclude_comp),
        exclude_month_first=bool(exclude_month_first),
        inflow_weights={"demand": w_demand, "pressure": w_pressure,
                        "convert": w_convert},
        vts_weights={"inflow": v_inflow, "base": v_base,
                     "proven": v_proven, "headroom": v_headroom},
        max_lag=int(max_lag),
        radius_km=float(radius_km),
    )
