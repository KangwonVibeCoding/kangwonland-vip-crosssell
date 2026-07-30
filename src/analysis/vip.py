"""VIP 가용 모수(VRB)와 VIP 타겟 지수(VTS).

VTS 상위 일자 = 캠페인 집행 캘린더. 이 대시보드의 최종 산출물이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S
from src.analysis import scoring, stats


def vip_ratio(demo: pd.DataFrame,
              age_bands: tuple[str, ...] = S.AGE_BANDS_VIP,
              gender: str = "전체") -> float:
    """전체 방문객 중 타겟 세그먼트 비중.

    데이터셋 2는 기간 집계이므로 **비율만 차용**하고 일별 규모는 ARS 에서 가져온다.
    이 결합 규칙이 VRB 의 전제다.
    """
    if demo.empty or "visitors" not in demo.columns:
        return float("nan")
    total = float(demo["visitors"].sum())
    if not total:
        return float("nan")
    sel = demo
    if gender in ("남", "여"):
        sel = sel.loc[sel["gender"] == gender]
    if age_bands:
        sel = sel.loc[sel["age_band"].isin(list(age_bands))]
    denom = float(demo.loc[demo["gender"] == gender, "visitors"].sum()) \
        if gender in ("남", "여") else total
    if not denom:
        return float("nan")
    return float(sel["visitors"].sum()) / denom


def segment_table(demo: pd.DataFrame) -> pd.DataFrame:
    """연령대 × 성별 방문객 표 + VIP 구분 (도넛·그룹바용)."""
    if demo.empty:
        return pd.DataFrame(columns=["gender", "age_band", "visitors", "segment"])

    def seg(band: str) -> str:
        if band in S.AGE_BANDS_VIP:
            return "VIP(40~60대)"
        if band in S.AGE_BANDS_SEMI:
            return "준VIP(30대)"
        return "기타"

    return demo.assign(segment=demo["age_band"].map(seg))


def compute_vrb(inflow: pd.DataFrame, demo: pd.DataFrame,
                age_bands: tuple[str, ...] = S.AGE_BANDS_VIP,
                gender: str = "전체") -> pd.DataFrame:
    """일별 VIP 가용 모수.

    ARS 가 있는 날(source='CII')은 `tickets`(실제 입장 건수)를 규모로 쓴다.
    ARS 가 없는 날(source='CAI')은 tickets 를 모르므로, CAI 를 CII 구간의 평균
    tickets 로 환산해 **추정치**로 표시한다 — UI 에 `추정` 배지를 붙인다.
    """
    if inflow.empty:
        return pd.DataFrame(columns=["date", "vrb", "scale", "is_estimated"])

    ratio = vip_ratio(demo, age_bands, gender)
    if not np.isfinite(ratio):
        ratio = 0.0

    df = inflow.copy()
    has_tickets = "tickets" in df.columns
    tickets = df["tickets"] if has_tickets else pd.Series(np.nan, index=df.index)
    tickets = pd.to_numeric(tickets, errors="coerce")

    # CAI 구간의 스케일 환산 계수: CII 구간에서 tickets 와 inflow 의 평균 비율
    known = tickets.notna() & (tickets > 0)
    if known.any():
        factor = float(tickets.loc[known].mean() / max(df.loc[known, "inflow"].mean(), 1e-9))
    else:
        factor = 1.0

    scale = tickets.where(known, df["inflow"] * factor)
    return df[["date"]].assign(
        scale=scale.to_numpy(),
        vip_ratio=ratio,
        vrb=(scale * ratio).to_numpy(),
        is_estimated=(~known).to_numpy(),
    )


def compute_vts(inflow: pd.DataFrame, vrb: pd.DataFrame, sales: pd.DataFrame,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """VIP 타겟 지수 (0~100) — 마케팅 집행 우선순위.

    네 신호를 합친다:
      inflow   사람이 많은 날
      base     VIP 모수가 큰 날
      proven   고마진 소비가 실증된 날
      headroom 유입 대비 교차판매가 아직 덜 된 날 = 기회

    headroom 이 핵심이다. 유입만 보면 "이미 잘 파는 날"을 또 공략하게 되는데,
    마케팅 여지는 오히려 attach rate 가 낮은 날에 있다.
    """
    weights = weights or S.VTS_WEIGHTS
    if inflow.empty:
        return pd.DataFrame(columns=["date", "vts", "grade"])

    keep = [c for c in ("date", "dow", "dow_name", "month", "inflow", "grade",
                        "source", "tickets", "is_capped") if c in inflow.columns]
    df = inflow[keep].copy()
    df = df.merge(vrb[["date", "vrb", "is_estimated"]], on="date", how="left")

    # 채널별 일별 판매량
    room = stats.daily_qty(sales, S.CH_ROOM)
    casino = stats.daily_qty(sales, S.CH_CASINO)
    local = stats.daily_qty(sales, S.CH_LOCAL)
    vip_qty = stats.daily_qty(sales, vip_only=True)

    idx = pd.Index(df["date"])
    def pick(s: pd.Series) -> pd.Series:
        return s.reindex(idx).astype("float64") if not s.empty else pd.Series(
            np.nan, index=idx, dtype="float64"
        )

    room_v, casino_v, local_v, vip_v = pick(room), pick(casino), pick(local), pick(vip_qty)
    onsite = (room_v.fillna(0) + casino_v.fillna(0))

    # attach rate = 유입 대비 리조트 내 소비량. 분모는 실제 입장 건수(tickets)가
    # 있으면 그것을, 없으면 inflow 점수를 쓴다 (구간 B 는 상대 비교만 유효).
    if "tickets" in df.columns:
        denom = pd.to_numeric(df["tickets"], errors="coerce")
        denom = denom.where(denom > 0, pd.to_numeric(df["inflow"], errors="coerce"))
    else:
        denom = pd.to_numeric(df["inflow"], errors="coerce")
    denom = pd.Series(denom.to_numpy(), index=idx).replace(0, np.nan)
    attach = onsite / denom

    parts = {
        "inflow": scoring.nrm(pd.Series(df["inflow"].to_numpy(), index=idx)),
        "base": scoring.nrm(pd.Series(df["vrb"].to_numpy(), index=idx)),
        "proven": scoring.nrm(vip_v),
        "headroom": 1.0 - scoring.nrm(attach),
    }
    vts = scoring.wsum(parts, weights)

    out = df.assign(
        room_qty=room_v.to_numpy(),
        casino_qty=casino_v.to_numpy(),
        local_qty=local_v.to_numpy(),
        vip_qty=vip_v.to_numpy(),
        attach=attach.to_numpy(),
        headroom=parts["headroom"].to_numpy(),
        vts=vts.to_numpy(),
        vts_grade=scoring.grade(vts).to_numpy(),
    )
    return out.sort_values("date").reset_index(drop=True)


def campaign_calendar(vts: pd.DataFrame, top_n: int = S.TOP_N_VTS_DAYS) -> pd.DataFrame:
    """VTS 상위 N일 = 캠페인 집행 캘린더. 대시보드의 최종 답."""
    if vts.empty:
        return vts
    cols = ["date", "dow_name", "inflow", "vrb", "vip_qty", "headroom",
            "vts", "vts_grade", "source", "is_estimated"]
    have = [c for c in cols if c in vts.columns]
    out = vts.sort_values("vts", ascending=False).head(top_n)[have].copy()
    if "dow_name" not in out.columns:
        out = out.assign(
            dow_name=pd.to_datetime(out["date"]).dt.dayofweek.map(
                dict(enumerate(S.DOW_NAMES))
            )
        )
    return out.reset_index(drop=True)
