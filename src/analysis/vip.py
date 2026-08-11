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
    """전체 고객 중 타겟 세그먼트 비중.

    데이터셋 2는 일자별로 오지만 보유 구간이 판매·ARS 와 겹치지 않고 `visitors`
    (원본 `CUST_WHOL_CNT`)가 누적 고객 수라 규모로 쓸 수 없다. 그래서 **비율만
    차용**하고 일별 규모는 ARS 에서 가져온다. 이 결합 규칙이 VRB 의 전제다.
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


def base_is_redundant(inflow: pd.Series, vrb: pd.Series,
                      threshold: float = S.VTS_BASE_REDUNDANT_R) -> bool:
    """VRB 가 유입의 상수배인가 — 그렇다면 base 축은 inflow 축의 복사본이다.

    성별·연령을 **일자별로 붙일 수 없어** `vip_ratio` 가 상수이므로
    VRB = tickets × 상수 가 된다(실측 r=0.9988). ARS 가 없는 구간은 더 노골적이라
    scale 자체가 inflow × 계수다.

    비율이 상수인지 직접 보지 않고 상관으로 판정하는 이유는, **판매 기간과 겹치는
    일자별 성별·연령이 들어오면 이 함수가 저절로 False 를 돌려주며 base 축이
    복구되어야** 하기 때문이다. 분산이 없는(=순위 정보가 없는) 계열도 중복으로
    본다 — 상수 축은 가중치만 먹고 순위를 바꾸지 않는다.
    """
    r = stats.pearson(inflow, vrb)
    return (not np.isfinite(r)) or abs(r) >= threshold


def drop_base_weight(weights: dict[str, float]) -> dict[str, float]:
    """base 축 가중치를 headroom 으로 이관한다.

    남은 축에 균등 분배하지 않는다. base 를 빼고 그냥 재정규화하면 inflow 비중이
    0.35 → 0.47 로 **오히려 커져서** VTS 가 유입 지수에 더 가까워진다(실측 0.963).
    이 지수의 존재 이유는 유입 상위일이 아닌 날을 찾는 것이므로 headroom 으로 옮긴다.
    """
    out = {k: v for k, v in weights.items() if k != "base"}
    out["headroom"] = out.get("headroom", 0.0) + float(weights.get("base", 0.0))
    return out


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

    ⚠ base 축은 VRB 가 유입의 상수배인 동안 **자동으로 빠진다**(`base_is_redundant`).
    성별·연령을 일자별로 붙일 수 없는 현재 데이터에서는 항상 그렇고, 그 상태로
    두 축을 다 쓰면 유입에 0.60 을 준 것과 같아져 지수가 유입의 재탕이 된다.
    어느 경로였는지는 결과의 `base_signal` 컬럼에 남는다 — CII 의 `pressure_signal`
    과 같은 규약이다. **없는 신호를 가중치로 채워 넣지 않는다.**
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
    redundant = base_is_redundant(df["inflow"], df["vrb"])
    if redundant:
        parts.pop("base")
        weights = drop_base_weight(weights)
    # 사이드바가 4축 가중치를 그대로 넘겨도 안전하다 — wsum 이 parts 에 없는 키를
    # 버리고 남은 가중치로 재정규화한다 (CII 의 pressure 축과 같은 처리).
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
        base_signal=("유입과 중복 — 제외" if redundant else "VRB"),
    )
    return out.sort_values("date").reset_index(drop=True)


def vts_vs_inflow(vts: pd.DataFrame, top_n: int = S.TOP_N_VTS_DAYS) -> dict:
    """VTS 가 유입 지수와 얼마나 다른가 — 지수의 존재 이유를 수치로 검증한다.

    반환: {corr, overlap, top_n, new_days(=유입 상위에는 없는 VTS 상위일)}

    이 값을 화면에 띄우는 것이 요점이다. 겹침이 10/10 이면 VTS 를 볼 이유가 없고,
    그 사실은 숨길 게 아니라 사용자가 알아야 할 정보다.
    """
    empty = {"corr": float("nan"), "overlap": 0, "top_n": 0, "new_days": []}
    if vts.empty or "vts" not in vts.columns or "inflow" not in vts.columns:
        return empty
    k = min(top_n, len(vts))
    if k == 0:
        return empty
    top_vts = vts.nlargest(k, "vts")
    top_inflow = set(vts.nlargest(k, "inflow")["date"])
    new_days = [d for d in top_vts["date"] if d not in top_inflow]
    return {
        "corr": stats.pearson(vts["inflow"], vts["vts"]),
        "overlap": k - len(new_days),
        "top_n": k,
        "new_days": new_days,
    }


def campaign_calendar(vts: pd.DataFrame, top_n: int = S.TOP_N_VTS_DAYS) -> pd.DataFrame:
    """VTS 상위 N일 = 캠페인 집행 캘린더. 대시보드의 최종 답.

    `inflow_rank`(유입 순위)와 `inflow_only_miss`(유입 상위 N일에는 없는 날)를
    같이 돌려준다. VTS 는 유입과 상관이 높을 수밖에 없는 지수라서(사람이 많은 날이
    실제로 중요하다) **차이가 나는 날이 어디인지**를 표에 드러내야 이 지수를 볼
    이유가 화면에서 증명된다.
    """
    if vts.empty:
        return vts
    cols = ["date", "dow_name", "inflow", "vrb", "vip_qty", "headroom",
            "vts", "vts_grade", "source", "is_estimated", "base_signal"]
    have = [c for c in cols if c in vts.columns]

    ranked = vts.assign(
        inflow_rank=vts["inflow"].rank(ascending=False, method="min").astype("int64"),
    )
    top_inflow = set(vts.nlargest(min(top_n, len(vts)), "inflow")["date"])
    out = ranked.sort_values("vts", ascending=False).head(top_n)[
        [*have, "inflow_rank"]
    ].copy()
    out = out.assign(inflow_only_miss=~out["date"].isin(top_inflow))
    if "dow_name" not in out.columns:
        out = out.assign(
            dow_name=pd.to_datetime(out["date"]).dt.dayofweek.map(
                dict(enumerate(S.DOW_NAMES))
            )
        )
    return out.reset_index(drop=True)
