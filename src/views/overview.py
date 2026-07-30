"""탭 1 — 메인 대시보드.

가설 검증 배너 → KPI → 유입 추이 → VIP 모수 → 캠페인 집행 캘린더.
상단에서 "카지노 유입이 리조트 소비를 견인한다"는 근거(상관계수)를 먼저 제시하고,
그 아래에서 그 관계를 실행 계획으로 번역한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import stats, vip
from src.ui import charts
from src.ui import components as C


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars_all: pd.DataFrame = ctx["data"]["ars"]
    sales_all: pd.DataFrame = ctx["data"]["sales"]
    demo: pd.DataFrame = ctx["data"]["demo"]

    ars = fs.slice_ars(ars_all)
    sales = fs.slice_sales(sales_all)

    if ars.empty and sales.empty:
        C.empty_state(
            "선택한 구간에 데이터가 없습니다.",
            "사이드바에서 분석 구간을 넓히거나 채널·티어 필터를 완화해 주세요.",
        )
        return
    if not fs.channels:
        C.empty_state(
            "채널을 하나 이상 선택해 주세요.",
            "카지노 식음 · 룸서비스 · 지역특산품 중 최소 하나가 필요합니다.",
        )
        return
    if not fs.tiers:
        C.empty_state(
            "상품 티어를 하나 이상 선택해 주세요.",
            "프리미엄 · 선물번들 · 일반 중 최소 하나가 필요합니다.",
        )
        return
    if not fs.age_bands:
        C.empty_state(
            "연령대를 하나 이상 선택해 주세요.",
            "VIP 가용 모수를 계산할 세그먼트가 비어 있습니다.",
        )
        return

    # ── 가설 검증 배너 ────────────────────────────────────────────────
    corr = stats.corr_table(ars, sales)
    head = stats.headline_corr(corr)
    if head:
        n_days = int(corr["n_days"].max())
        top = max(head.values())
        pairs = [
            (S.CHANNEL_LABEL[ch], f"{head[ch]:.3f}")
            for ch in S.CHANNELS if ch in head
        ]
        C.hero_banner(
            headline=f"카지노 유입 ↔ 리조트 소비 상관 r = {top:.2f}",
            sub="ARS 예약 유입이 클수록 리조트 내 F&B 소비가 함께 커진다 — "
                "교차판매 가설이 실데이터로 확인된다.",
            stats_pairs=pairs,
            badge="실측 조인",
            badge_detail=f"{n_days}일 날짜 조인",
        )
    else:
        C.hero_banner(
            headline="이 구간은 유입·판매 데이터가 겹치지 않습니다",
            sub="ARS 예약 데이터와 판매 데이터가 함께 존재하는 구간에서만 "
                "상관을 실측할 수 있습니다. 사이드바에서 '정밀 검증' 구간을 선택해 보세요.",
            stats_pairs=[],
            badge=fs.period_badge,
        )

    # ── 지수 산출 ─────────────────────────────────────────────────────
    infl = inflow_mod.build_inflow(ars, sales, fs.inflow_weights)
    if infl.empty:
        C.empty_state("이 구간에서는 유입 지수를 계산할 수 없습니다.")
        return

    vrb = vip.compute_vrb(infl, demo, fs.age_bands, fs.gender)
    vts = vip.compute_vts(infl, vrb, sales, fs.vts_weights)

    # 이전 동일 길이 구간 (KPI 델타 비교용)
    prev_start, prev_end = fs.previous_window()
    prev_ars = ars_all.loc[ars_all["date"].between(prev_start, prev_end)]
    prev_sales = sales_all.loc[sales_all["date"].between(prev_start, prev_end)]

    # ── KPI 5열 ──────────────────────────────────────────────────────
    st.markdown("")
    cols = st.columns(5)

    tickets_now = float(ars["tickets"].sum()) if not ars.empty else None
    tickets_prev = float(prev_ars["tickets"].sum()) if not prev_ars.empty else None
    with cols[0]:
        delta, direction = C.pct_delta(tickets_now, tickets_prev)
        C.kpi_card("입장권 구매 건수", C.fmt_int(tickets_now, "건"), delta, direction,
                   note="당첨자 실구매" if tickets_now else "이 구간 ARS 없음")

    compete_now = float(ars["compete_ratio"].mean()) if not ars.empty else None
    compete_prev = float(prev_ars["compete_ratio"].mean()) if not prev_ars.empty else None
    with cols[1]:
        delta, direction = C.point_delta(compete_now, compete_prev, 2, "배")
        C.kpi_card("예약 경쟁률", C.fmt_float(compete_now, 2, "배"), delta, direction,
                   note="접수자 / 당첨자")

    vrb_now = float(vrb["vrb"].sum()) if not vrb.empty else None
    estimated = bool(vrb["is_estimated"].any()) if not vrb.empty else False
    with cols[2]:
        ratio = float(vrb["vip_ratio"].iloc[0]) if not vrb.empty else None
        C.kpi_card("VIP 추정 모수", C.fmt_int(vrb_now, "명"),
                   f"비율 {C.fmt_pct(ratio)}", 0,
                   note="CAI 기반 추정" if estimated else "ARS 기반")

    vip_qty = float(sales.loc[sales["tier"] != S.TIER_STANDARD, "qty"].sum()) \
        if not sales.empty else 0.0
    total_qty = float(sales["qty"].sum()) if not sales.empty else 0.0
    share_now = (vip_qty / total_qty) if total_qty else None
    prev_total = float(prev_sales["qty"].sum()) if not prev_sales.empty else 0.0
    prev_vip = float(prev_sales.loc[prev_sales["tier"] != S.TIER_STANDARD, "qty"].sum()) \
        if not prev_sales.empty else 0.0
    share_prev = (prev_vip / prev_total) if prev_total else None
    with cols[3]:
        delta, direction = C.point_delta(
            (share_now or 0) * 100 if share_now is not None else None,
            (share_prev or 0) * 100 if share_prev is not None else None,
        )
        C.kpi_card("VIP 상품 비중", C.fmt_pct(share_now), delta, direction,
                   note="프리미엄 + 선물번들")

    attach_now = float(vts["attach"].median()) if "attach" in vts.columns else None
    with cols[4]:
        C.kpi_card("Attach rate", C.fmt_float(attach_now, 2), "", 0,
                   note="유입 1건당 리조트 소비량")

    # ── 유입 추이 ─────────────────────────────────────────────────────
    st.markdown("")
    has_tickets = "tickets" in infl.columns and infl["tickets"].notna().any()
    C.section_header(
        "카지노 유입 추이",
        "위: 입장권 구매 건수 · 아래: 유입 강도 지수. 스케일이 달라 이중 축 대신 "
        "x축을 공유하는 2단으로 그렸다.",
        badge=fs.period_badge,
    )
    st.plotly_chart(
        charts.inflow_stack(infl, value_col="tickets"),
        width="stretch", key="ov_inflow",
    )
    table_cols = [c for c in ("date", "dow_name", "tickets", "recv_total",
                              "buy_rate", "inflow", "grade", "source")
                  if c in infl.columns]
    C.table_view(infl[table_cols].rename(columns={
        "date": "날짜", "dow_name": "요일", "tickets": "입장권 구매",
        "recv_total": "총 접수자", "buy_rate": "구매율", "inflow": "유입 지수",
        "grade": "등급", "source": "산출",
    }))
    if not has_tickets:
        C.caveat(
            "이 구간에는 ARS 데이터가 없어 카지노 식음 소비로 산출한 객장 활동 "
            "지수(CAI)를 유입 프록시로 사용했습니다."
        )
    else:
        validity = inflow_mod.cai_validity(ars, sales)
        if validity["n_days"] >= 10 and pd.notna(validity["pearson"]):
            C.caveat(
                f"참고: 같은 구간에서 판매 기반 활동지수(CAI)와 ARS 기반 유입지수(CII)의 "
                f"상관은 r={validity['pearson']:.3f} ({validity['n_days']}일)입니다 — "
                "ARS 가 없는 기간에 CAI 를 대체 지표로 쓰는 근거입니다."
            )

    # ── VIP 모수 ──────────────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "VIP 가용 모수 구성",
        "왼쪽 도넛은 3조각으로 압축했다(색약 안전성 상한). 상세 연령×성별은 오른쪽 그룹 바.",
        badge="추정" if estimated else "",
    )
    left, right = st.columns([0.4, 0.6])
    seg = vip.segment_table(demo)
    with left:
        if seg.empty:
            C.empty_state("성별·연령 데이터가 없습니다.")
        else:
            st.plotly_chart(charts.segment_donut(seg), width="stretch", key="ov_donut")
    with right:
        if demo.empty:
            C.empty_state("성별·연령 데이터가 없습니다.")
        else:
            st.plotly_chart(charts.age_gender_bars(demo), width="stretch", key="ov_ag")
    if not demo.empty:
        C.table_view(
            demo[["gender", "age_band", "visitors"]].rename(columns={
                "gender": "성별", "age_band": "연령대", "visitors": "방문고객수"}),
            label="성별·연령 데이터 표로 보기",
        )
        C.caveat(
            "성별·연령 데이터는 기간 집계로 제공되어 일별 분해가 불가능합니다. "
            "비율만 차용하고 일별 규모는 ARS 입장권 구매 건수에서 가져왔습니다."
        )

    # ── 캠페인 집행 캘린더 ────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "VIP 타겟 지수 상위일 = 캠페인 집행 캘린더",
        "유입·모수·실증 소비·미달 기회를 합산한 우선순위. 이미 잘 파는 날보다 "
        "'유입은 많은데 아직 덜 판 날'이 높게 나온다.",
        badge=fs.period_badge,
    )
    calendar = vip.campaign_calendar(vts)
    if calendar.empty:
        C.empty_state("이 구간에서는 VTS 를 계산할 수 없습니다.")
    else:
        display = calendar.assign(
            날짜=calendar["date"].dt.strftime("%Y-%m-%d"),
            요일=calendar.get("dow_name", ""),
            유입지수=calendar["inflow"].round(1),
            VIP모수=calendar["vrb"].round(0),
            VIP소비=calendar["vip_qty"].fillna(0).round(0),
            미달기회=calendar["headroom"].round(3),
            VTS=calendar["vts"].round(1),
            등급=calendar["vts_grade"],
            산출=calendar["source"],
        )[["날짜", "요일", "유입지수", "VIP모수", "VIP소비", "미달기회",
           "VTS", "등급", "산출"]]
        st.dataframe(
            display, width="stretch", hide_index=True,
            column_config={
                "VTS": st.column_config.ProgressColumn(
                    "VTS", min_value=0, max_value=100, format="%.1f"),
                "VIP모수": st.column_config.NumberColumn("VIP 모수", format="%d"),
                "VIP소비": st.column_config.NumberColumn("VIP 소비", format="%d"),
                "미달기회": st.column_config.NumberColumn("미달 기회", format="%.3f"),
                "유입지수": st.column_config.NumberColumn("유입 지수", format="%.1f"),
            },
        )

    # ── 자동 인사이트 ─────────────────────────────────────────────────
    lines: list[str] = []
    if not calendar.empty:
        best = calendar.iloc[0]
        lines.append(
            f"집행 우선순위 1위는 {best['date']:%Y-%m-%d}"
            f"({best.get('dow_name', '')}요일) — VTS {best['vts']:.1f}, "
            f"유입 지수 {best['inflow']:.1f}."
        )
    dow = inflow_mod.dow_profile_table(ars, sales)
    if not dow.empty:
        bits = []
        for _, row in dow.iterrows():
            peak = max(S.DOW_NAMES, key=lambda d: (row[d] if pd.notna(row[d]) else -1))
            bits.append(f"{row['label']} {peak}요일({row[peak]:.2f})")
        lines.append("요일 피크: " + " · ".join(bits) +
                     " — 유입 피크와 채널 피크가 어긋나는 곳이 교차판매 여지다.")
    cross = stats.channel_cross_corr(sales)
    if not cross.empty:
        weakest = cross.loc[cross["pearson"].idxmin()]
        lines.append(
            f"가장 느슨하게 연결된 조합은 {weakest['a_label']} ↔ {weakest['b_label']}"
            f" (r={weakest['pearson']:.3f}) — 아직 같이 팔리지 않으니 기회가 가장 크다."
        )
    C.insight_box(lines)
    C.caveat(
        "고마진 지표는 단가 데이터가 제공되지 않아 상품명 티어(프리미엄/선물번들)와 "
        "판매 희소도를 결합한 프록시입니다."
    )
