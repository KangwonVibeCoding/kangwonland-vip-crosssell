"""스코어링 프리미티브 — 정규화, 가중합, 등급화.

모든 지수(CII·CAI·VTS·CSM)가 이 세 함수 위에 쌓인다. 여기가 흔들리면 전부
흔들리므로 경계 조건(상수 시리즈, 빈 시리즈, 0으로 나누기)을 명시적으로 다룬다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S


def nrm(series: pd.Series, *, robust: bool = True) -> pd.Series:
    """0~1 정규화.

    robust=True 면 5~95 분위로 클리핑한 뒤 min-max 한다. 실데이터에 **매월 1일
    판매량 스파이크**(카지노 식음 1.45배, 2년 평균)가 있어서 단순 min-max 를 쓰면 그 하루가
    스케일을 독점하고 나머지가 바닥에 깔린다. 클리핑이 이를 막는다.

    상수 시리즈는 0.5 를 반환한다 (0으로 나누기 방지). 값이 하나도 없으면 빈
    시리즈를 그대로 돌려준다.
    """
    s = pd.to_numeric(pd.Series(series), errors="coerce").astype("float64")
    if s.empty:
        return s
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype="float64")

    if robust and len(valid) >= 5:
        lo, hi = valid.quantile(list(S.ROBUST_CLIP))
    else:
        lo, hi = valid.min(), valid.max()

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype="float64")

    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5)


def wsum(parts: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """가중합 → 0~100 스케일.

    가중치 합으로 자동 정규화하므로 사이드바 슬라이더가 합계 1을 벗어나도 안전하다.
    parts 에 없는 키는 무시하고, 남은 가중치만으로 다시 정규화한다.
    """
    used = {k: v for k, v in parts.items() if k in weights and v is not None}
    if not used:
        return pd.Series(dtype="float64")

    index = next(iter(used.values())).index
    total_w = sum(abs(weights[k]) for k in used)
    if not total_w:
        return pd.Series(np.full(len(index), 50.0), index=index, dtype="float64")

    acc = pd.Series(np.zeros(len(index)), index=index, dtype="float64")
    for key, series in used.items():
        aligned = pd.to_numeric(series, errors="coerce").reindex(index).fillna(0.5)
        acc = acc + aligned.astype("float64") * weights[key]
    return (acc / total_w * 100.0).clip(0, 100)


def grade(series: pd.Series) -> pd.Series:
    """점수 → 혼잡도 등급 라벨 (피크 / 성수 / 보통 / 한산)."""
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    for threshold, label in S.GRADE_BINS:
        out = out.mask(out.isna() & (s >= threshold), label)
    return out.fillna(S.GRADE_BINS[-1][1])


def percentile_rank(series: pd.Series) -> pd.Series:
    """0~1 백분위 순위. 희소도(rarity) 계산에 쓴다."""
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.full(len(s), 0.5), index=s.index, dtype="float64")
    return s.rank(pct=True).fillna(0.5).astype("float64")


def margin_proxy(sales: pd.DataFrame) -> pd.DataFrame:
    """상품(item_id) 단위 고마진 프록시.

    단가 컬럼이 제공되지 않으므로 세 신호를 결합한다:
      1) 티어  — 상품명 기반 프리미엄 / 선물번들 / 일반
      2) 희소도 — 판매수량이 적을수록 고단가일 확률이 높다는 가정
      3) 할인율 — 할인 판매 비중이 높으면 감점 (실측 수량 기준 2.3% 로 희소해서
                  보조 신호로만 쓴다)

    반환 컬럼: item_id, channel, venue, item, tier, is_comp, total_qty,
              rarity, discount_share, margin_proxy
    """
    if sales.empty:
        return pd.DataFrame(columns=[
            "item_id", "channel", "venue", "item", "tier", "is_comp",
            "total_qty", "rarity", "discount_share", "margin_proxy",
        ])

    agg = (
        sales.assign(discount_qty=sales["qty"].where(sales["is_discount"], 0))
        .groupby("item_id", as_index=False)
        .agg(
            channel=("channel", "first"),
            venue=("venue", "first"),
            item=("item", "first"),
            tier=("tier", "first"),
            is_comp=("is_comp", "any"),
            total_qty=("qty", "sum"),
            discount_qty=("discount_qty", "sum"),
        )
    )
    agg = agg.assign(
        discount_share=(
            agg["discount_qty"] / agg["total_qty"].replace(0, np.nan)
        ).fillna(0.0)
    ).drop(columns=["discount_qty"])

    # 희소도는 채널 안에서 비교해야 의미가 있다 (카지노 식음과 룸서비스의
    # 절대 판매량 스케일이 20배 이상 다르다 — 실측 평균 9,565 vs 187)
    rarity = 1.0 - agg.groupby("channel")["total_qty"].transform(percentile_rank)

    base = agg["tier"].map(S.TIER_MARGIN_BASE).astype("float64").fillna(
        S.TIER_MARGIN_BASE[S.TIER_STANDARD]
    )
    score = base + S.RARITY_BONUS * rarity - S.DISCOUNT_PENALTY * agg["discount_share"]
    # 컴프/바우처는 무상 제공 → 마진 기여 0 으로 본다
    score = score.mask(agg["is_comp"].astype(bool), 0.0)
    return agg.assign(rarity=rarity, margin_proxy=score.clip(0.0, 1.0))
