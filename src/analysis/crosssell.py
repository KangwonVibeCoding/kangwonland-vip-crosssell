"""교차판매 매칭 스코어(CSM)와 번들 추천.

핵심 아이디어는 **유입 탄력성(lift)** 이다. 고유입일과 저유입일의 판매 비중을
비교해, 카지노 유입이 늘 때 더 팔리는 상품을 찾아낸다. 비중(share)으로 비교하기
때문에 "그냥 많이 팔리는 상품"이 아니라 "유입에 반응하는 상품"이 걸린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import scoring

EPS = 1e-9


def compute_lift(sales: pd.DataFrame, inflow: pd.DataFrame,
                 quantile: float = S.HI_LO_QUANTILE) -> pd.DataFrame:
    """상품별 유입 탄력성.

    lift = share(item | 고유입일) / share(item | 저유입일)

    lift > 1 이면 유입 증가 시 판매 비중이 커지는 상품 → 교차판매 타겟.

    ⚠ 저유입일 판매가 0인 상품이 실제로 존재한다(구간 A 기준 72종). 그대로 나누면
    lift 가 수천만까지 발산해 CSM 상위를 독점한다. **가법 스무딩**을 적용해
    "판매 1개분의 비중"(alpha)을 분자·분모에 더한다 — 표본이 없는 것을 무한한
    탄력성으로 읽지 않기 위해서다.

    고/저 구간을 나눌 표본이 부족하면 lift=NaN 으로 두고 CSM 은 나머지 신호로만
    계산한다 (조용히 1.0 으로 채우면 없는 근거를 있는 것처럼 보이게 된다).
    """
    cols = ["item_id", "lift", "hi_share", "lo_share", "hi_qty", "lo_qty"]
    if sales.empty:
        return pd.DataFrame(columns=cols)

    hi, lo = inflow_mod.hi_lo_days(inflow, quantile)
    if not hi or not lo:
        return pd.DataFrame(columns=cols)

    def totals(days: set) -> tuple[pd.Series, float]:
        sub = sales.loc[sales["date"].isin(days)]
        if sub.empty:
            return pd.Series(dtype="float64"), 0.0
        by_item = sub.groupby("item_id")["qty"].sum()
        return by_item, float(by_item.sum())

    hi_qty, hi_total = totals(hi)
    lo_qty, lo_total = totals(lo)
    if not hi_total or not lo_total:
        return pd.DataFrame(columns=cols)

    items = hi_qty.index.union(lo_qty.index)
    hi_q = hi_qty.reindex(items).fillna(0.0)
    lo_q = lo_qty.reindex(items).fillna(0.0)
    hi_share = hi_q / hi_total
    lo_share = lo_q / lo_total

    # 판매 1개분에 해당하는 비중 — 관측이 더 촘촘한(총량이 큰) 쪽을 기준으로 잡는다
    alpha = 1.0 / max(hi_total, lo_total, 1.0)

    frame = pd.DataFrame({
        "item_id": items,
        "hi_share": hi_share.to_numpy(),
        "lo_share": lo_share.to_numpy(),
        "hi_qty": hi_q.to_numpy(),
        "lo_qty": lo_q.to_numpy(),
    })
    lift = (frame["hi_share"] + alpha) / (frame["lo_share"] + alpha)

    # 표본 요건 미달 상품은 lift 를 계산하지 않는다. 판매일수 2일짜리 상품도
    # lift 20~60 이 나오는데 그건 탄력성이 아니라 표본 잡음이다.
    per_item = sales.groupby("item_id").agg(
        days=("date", "nunique"), qty=("qty", "sum")
    ).reindex(items)
    enough = (
        (per_item["days"].fillna(0) >= S.LIFT_MIN_DAYS)
        & (per_item["qty"].fillna(0) >= S.LIFT_MIN_QTY)
    ).to_numpy()

    return frame.assign(lift=lift.where(enough, np.nan))[cols]


def compute_csm(sales: pd.DataFrame, inflow: pd.DataFrame,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """상품별 교차판매 매칭 스코어 (0~100).

    elasticity  유입에 반응하는가 (lift, log 압축)
    scale       물량이 받쳐주는가
    margin      고마진인가 (프록시)
    """
    weights = weights or S.CSM_WEIGHTS
    base = scoring.margin_proxy(sales)
    if base.empty:
        return base.assign(lift=np.nan, csm=np.nan)

    lift = compute_lift(sales, inflow)
    out = base.merge(lift, on="item_id", how="left")
    # lift 가 비었으면 merge 결과가 object dtype 이 된다 → log1p 가 터진다
    out = out.assign(lift=pd.to_numeric(out["lift"], errors="coerce"))

    # log1p 로 압축한다 — 스무딩 후에도 lift 는 롱테일이다
    elasticity = scoring.nrm(np.log1p(out["lift"].clip(lower=0)))
    parts = {
        "elasticity": elasticity,
        "scale": scoring.nrm(out["total_qty"]),
        "margin": out["margin_proxy"].astype("float64"),
    }
    csm = scoring.wsum(parts, weights)
    # 무상 제공 상품은 교차판매 대상이 아니다. 마진 0 만으로는 물량·탄력성 신호가
    # 남아 상위를 점령하므로(실측 CSM 75) 스코어 자체를 0 으로 눕힌다.
    csm = csm.mask(out["is_comp"].astype(bool), 0.0)

    return out.assign(
        channel_label=out["channel"].map(S.CHANNEL_LABEL),
        tier_label=out["tier"].map(S.TIER_LABEL),
        csm=csm.to_numpy(),
    ).sort_values("csm", ascending=False).reset_index(drop=True)


def top_items(csm: pd.DataFrame, channel: str | None = None, n: int = S.TOP_N_ITEMS,
              *, tiers: tuple[str, ...] | None = None,
              exclude_comp: bool = False, by: str = "csm") -> pd.DataFrame:
    """채널·티어로 필터링한 상위 상품."""
    if csm.empty:
        return csm
    out = csm
    if channel:
        out = out.loc[out["channel"] == channel]
    if tiers:
        out = out.loc[out["tier"].isin(list(tiers))]
    if exclude_comp:
        out = out.loc[~out["is_comp"]]
    if by not in out.columns:
        by = "total_qty"
    return out.sort_values(by, ascending=False).head(n).reset_index(drop=True)


def cooccurrence(sales: pd.DataFrame, item_a: str, item_b: str) -> dict[str, float | int]:
    """두 상품이 같은 날 함께 상승한 날 수 — 번들 추천의 근거 수치."""
    a = sales.loc[sales["item_id"] == item_a].groupby("date")["qty"].sum()
    b = sales.loc[sales["item_id"] == item_b].groupby("date")["qty"].sum()
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return {"days": int(len(common)), "both_up": 0}
    a, b = a.loc[common], b.loc[common]
    both_up = int(((a > a.median()) & (b > b.median())).sum())
    return {"days": int(len(common)), "both_up": both_up}


def recommend_bundles(csm: pd.DataFrame, sales: pd.DataFrame,
                      top_n: int = 3) -> list[dict]:
    """룸서비스 × 지역특산품 번들 추천.

    두 채널을 굳이 묶는 이유: 실측 채널 상관에서 룸서비스↔특산품이 가장 느슨하다
    = 아직 같이 안 팔리고 있다 = 교차판매 여지가 가장 크다.

    ⚠ 상시 판매 상품만 후보로 쓴다. 판매일수 1~2일짜리 상품은 CSM 이 높게 나와도
    "재고가 그때만 있었던 것"이라 번들로 걸 수 없다.
    """
    if csm.empty or sales.empty:
        return []

    days = sales.groupby("item_id")["date"].nunique()
    regular = set(days.loc[days >= S.BUNDLE_MIN_DAYS].index)
    if not regular:
        return []
    pool = csm.loc[csm["item_id"].isin(regular)]

    room = top_items(pool, S.CH_ROOM, n=top_n * 2, exclude_comp=True)
    local = top_items(pool, S.CH_LOCAL, n=top_n * 2, exclude_comp=True)
    if room.empty or local.empty:
        return []

    out: list[dict] = []
    for i in range(min(top_n, len(room), len(local))):
        r, l = room.iloc[i], local.iloc[i]
        co = cooccurrence(sales, str(r["item_id"]), str(l["item_id"]))
        out.append({
            "rank": i + 1,
            "room_item": str(r["item"]),
            "room_csm": float(r["csm"]),
            "room_tier": S.TIER_LABEL.get(str(r["tier"]), str(r["tier"])),
            "local_item": str(l["item"]),
            "local_csm": float(l["csm"]),
            "local_tier": S.TIER_LABEL.get(str(l["tier"]), str(l["tier"])),
            "combined_csm": float(r["csm"] + l["csm"]) / 2.0,
            "co_days": co["days"],
            "both_up_days": co["both_up"],
        })
    return out


def heatmap_dow_month(sales: pd.DataFrame, channel: str | None = None,
                      *, tiers: tuple[str, ...] | None = None,
                      exclude_comp: bool = False) -> pd.DataFrame:
    """요일 × 월 수요 히트맵용 피벗.

    ⚠ 원본에 주문 시각(hour) 컬럼이 없어 원래 계획했던 '심야 피크타임' 분석은
    불가능하다. 요일 × 월 로 대체하면 주간 패턴과 계절성을 한 장에서 볼 수 있다
    (실측: 룸서비스 1월·일요일, 특산품 9월·일요일 셀이 최고).

    반환: index=월(1~12), columns=요일명(월~일), 값=평균 일판매량
    """
    df = sales if channel is None else sales.loc[sales["channel"] == channel]
    if tiers:
        df = df.loc[df["tier"].isin(list(tiers))]
    if exclude_comp:
        df = df.loc[~df["is_comp"]]
    if df.empty:
        return pd.DataFrame()

    daily = df.groupby(["date", "month", "dow"], as_index=False)["qty"].sum()
    pivot = daily.pivot_table(index="month", columns="dow", values="qty", aggfunc="mean")
    pivot = pivot.reindex(index=range(1, 13), columns=range(7))
    pivot.columns = [S.DOW_NAMES[c] for c in pivot.columns]
    pivot.index = [f"{m}월" for m in pivot.index]
    # ⚠ 축 이름을 명시한다. 한국어에서 '월' 은 month 이자 Monday 라, reset_index() 로
    # 표를 만들 때 인덱스 컬럼을 '월' 로 부르면 요일 '월' 과 이름이 충돌한다.
    pivot.index.name = "기준 월"
    pivot.columns.name = "요일"
    return pivot


def treemap_frame(sales: pd.DataFrame, channel: str = S.CH_CASINO,
                  *, venues: tuple[str, ...] | None = None,
                  exclude_comp: bool = False, top_n: int = 60) -> pd.DataFrame:
    """영업장 → 상품 계층 트리맵용 프레임 (카지노 식음 5개 영업장)."""
    df = sales.loc[sales["channel"] == channel]
    if venues:
        df = df.loc[df["venue"].isin(list(venues))]
    if exclude_comp:
        df = df.loc[~df["is_comp"]]
    if df.empty:
        return pd.DataFrame(columns=["venue", "item", "qty", "tier_label"])

    agg = (
        df.groupby(["venue", "item", "tier"], as_index=False)["qty"].sum()
        .sort_values("qty", ascending=False)
        .head(top_n)
    )
    return agg.assign(tier_label=agg["tier"].map(S.TIER_LABEL))
