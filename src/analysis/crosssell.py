"""교차판매 매칭 스코어(CSM)와 번들 추천.

핵심 아이디어는 **유입 탄력성(lift)** 이다. 고유입일과 저유입일의 판매 비중을
비교해, 카지노 유입이 늘 때 더 팔리는 상품을 찾아낸다. 비중(share)으로 비교하기
때문에 "그냥 많이 팔리는 상품"이 아니라 "유입에 반응하는 상품"이 걸린다.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import scoring

EPS = 1e-9


def sample_gate(sales: pd.DataFrame) -> dict:
    """lift 표본 요건을 **창 길이와 채널 스케일에 맞춰** 계산한다.

    반환: {window_days, min_days, min_qty(채널→수량), smooth_qty}

    두 축을 함께 건다.
      · 판매일수 — 절대 5일과 창의 20% 중 큰 값. 절대값만 두면 731일 창에서
        게이트가 사실상 사라진다(실측 1,688종 중 1,534종 통과, lift 최대 24,027).
      · 수량 — 절대 10개와 **그 채널 상품 판매량 중앙값** 중 큰 값. 채널 공통
        절대값을 쓰면 스케일 차(카지노 식음 : 룸서비스 ≈ 10배)에 막혀 한쪽 채널이
        통째로 탈락한다. 중앙값을 쓰면 "채널 안에서 중간 이상 팔린 상품"이라는
        같은 의미가 어느 창·어느 채널에서도 성립한다.

    `smooth_qty` 는 가법 스무딩 상수의 분자로, 전체 상품 판매량의 중앙값이다
    (채널을 섞은 값 — lift 의 share 자체가 전 채널 합계 기준이기 때문).

    실측: 31일 창 → 7일 · 카지노 129 / 특산품 40 / 룸서비스 16개
          731일 창 → 147일 · 카지노 983 / 특산품 356 / 룸서비스 87개
    """
    if sales.empty:
        return {"window_days": 0, "min_days": S.LIFT_MIN_DAYS,
                "min_qty": {}, "smooth_qty": float(S.LIFT_MIN_QTY)}

    window_days = int(sales["date"].nunique())
    min_days = int(max(S.LIFT_MIN_DAYS,
                       math.ceil(window_days * S.LIFT_MIN_DAY_RATIO)))
    per_item = sales.groupby(["channel", "item_id"])["qty"].sum()
    by_channel = per_item.groupby(level="channel").quantile(S.LIFT_MIN_QTY_QUANTILE)
    min_qty = {
        str(ch): float(max(S.LIFT_MIN_QTY, v)) for ch, v in by_channel.items()
    }
    smooth_qty = float(max(S.LIFT_MIN_QTY,
                           per_item.quantile(S.LIFT_MIN_QTY_QUANTILE)))
    return {"window_days": window_days, "min_days": min_days,
            "min_qty": min_qty, "smooth_qty": smooth_qty}


def compute_lift(sales: pd.DataFrame, inflow: pd.DataFrame,
                 quantile: float = S.HI_LO_QUANTILE) -> pd.DataFrame:
    """상품별 유입 탄력성.

    lift = share(item | 고유입일) / share(item | 저유입일)

    lift > 1 이면 유입 증가 시 판매 비중이 커지는 상품 → 교차판매 타겟.

    ⚠ 저유입일 판매가 0인 상품이 실제로 존재한다(구간 A 기준 72종). 그대로 나누면
    lift 가 수천만까지 발산해 CSM 상위를 독점한다. **가법 스무딩**을 적용해
    alpha 를 분자·분모에 더한다 — 표본이 없는 것을 무한한 탄력성으로 읽지 않기
    위해서다.

    alpha 는 "판매 1개분의 비중"이 아니라 **중앙값 규모 상품 1개분의 비중**이다.
    1개분으로 잡으면 총량이 커질수록 alpha 가 0에 수렴해(731일 창 실측 3.2e-07)
    스무딩이 사실상 사라진다. 중앙값 기준으로 잡으면 창 길이와 무관하게 다음 성질이
    성립한다: **중앙값 규모 상품이 저유입일에 하나도 안 팔려도 lift ≤ 2.**
    관측이 두꺼운 상품일수록 alpha 의 영향은 자동으로 작아진다.

    ⚠ 고·저 유입일은 **달 안에서** 나눈다(`hi_lo_days_by_month`). 전역 분위로
    자르면 유입 추세가 고유입일을 특정 시기에 몰아넣어 lift 가 수명주기를 재게
    된다 — 실측 731일 창에서 최댓값이 47.05(2023 상반기 단종 상품)에서 2.02 로
    내려갔다. 자세한 근거는 그 함수의 docstring 에 있다.

    고/저 구간을 나눌 표본이 부족하면 lift=NaN 으로 두고 CSM 은 나머지 신호로만
    계산한다 (조용히 1.0 으로 채우면 없는 근거를 있는 것처럼 보이게 된다).
    반환 프레임의 `attrs` 에 게이트 값과 탈락 종수를 남긴다 — 화면에서 "몇 종을
    무엇 때문에 뺐는지" 밝히기 위해서다.
    """
    cols = ["item_id", "lift", "hi_share", "lo_share", "hi_qty", "lo_qty"]
    if sales.empty:
        return pd.DataFrame(columns=cols)

    # 달 안에서 고·저를 나눈다 — 전역 분위로 자르면 유입 추세가 있는 긴 창에서
    # 고유입일이 특정 시기로 몰려 lift 가 상품 수명주기를 재게 된다
    hi, lo = inflow_mod.hi_lo_days_by_month(inflow, quantile)
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

    # 표본 요건은 창 길이·채널 스케일을 따라간다 — 절대값만 두면 긴 창에서
    # 게이트가 사라지고, 채널 공통값을 쓰면 작은 채널이 통째로 탈락한다
    gate = sample_gate(sales)

    # 중앙값 크기 상품 1개분에 해당하는 비중 — 관측이 촘촘한(총량이 큰) 쪽을 기준으로
    alpha = gate["smooth_qty"] / max(hi_total, lo_total, 1.0)

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
        days=("date", "nunique"), qty=("qty", "sum"), channel=("channel", "first")
    ).reindex(items)
    qty_floor = per_item["channel"].map(gate["min_qty"]).astype("float64").fillna(
        float(S.LIFT_MIN_QTY)
    )
    enough = (
        (per_item["days"].fillna(0) >= gate["min_days"])
        & (per_item["qty"].fillna(0) >= qty_floor)
    ).to_numpy()

    out = frame.assign(lift=lift.where(enough, np.nan))[cols]
    out.attrs["gate"] = {
        **gate, "alpha": alpha,
        "kept": int(enough.sum()), "dropped": int((~enough).sum()),
    }
    return out


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
    measured = out["lift"].notna()

    # log1p 로 압축한다 — 스무딩 후에도 lift 는 롱테일이다
    elasticity = scoring.nrm(np.log1p(out["lift"].clip(lower=0)))

    # ⚠ 표본 미달 상품의 탄력성을 **0.5 로 두면 안 된다.** `nrm` 의 기본 채움값이
    # 0.5 인데, 실측된 상품의 탄력성 중앙값은 0.24(731일 창)라서 0.5 는 중립이
    # 아니라 상위 18% 대우다 — 측정된 상품의 82%보다 높다. 그러면 "표본이 부족하다"는
    # 사실이 점수 보너스가 되고, 실제로 유입 반응이 나빴던 상품(lift 0.71)이
    # 아무것도 모르는 상품보다 33점 낮게 나왔다.
    #
    # 대안으로 축을 빼고 남은 가중치로 재정규화하는 방법(CII pressure·VTS base 와
    # 같은 규약)을 재 봤는데, 여기서는 반대로 뒤집힌다: 물량·마진이 둘 다 최상인
    # 미측정 상품이 100점으로 1위가 된다(실측 +22.5점). 축이 둘만 남으면 만점이
    # 쉬워지기 때문이다. 그래서 **실측 분포의 중앙값**으로 채운다 — 미측정 상품은
    # 1위가 될 수도, 공짜 보너스를 받을 수도 없는 '평범한 상품' 대우가 된다.
    fill = float(S.CSM_ELASTICITY_FILL_DEFAULT)
    if measured.any():
        fill = float(elasticity.loc[measured].quantile(S.CSM_ELASTICITY_FILL_QUANTILE))
    elasticity = elasticity.mask(~measured, fill)
    parts = {
        "elasticity": elasticity,
        "scale": scoring.nrm(out["total_qty"]),
        "margin": out["margin_proxy"].astype("float64"),
    }
    csm = scoring.wsum(parts, weights)
    # 무상 제공 상품은 교차판매 대상이 아니다. 마진 0 만으로는 물량·탄력성 신호가
    # 남아 상위를 점령하므로(실측 CSM 75) 스코어 자체를 0 으로 눕힌다.
    csm = csm.mask(out["is_comp"].astype(bool), 0.0)

    scored = out.assign(
        channel_label=out["channel"].map(S.CHANNEL_LABEL),
        tier_label=out["tier"].map(S.TIER_LABEL),
        lift_measured=measured.to_numpy(),
        elasticity=elasticity.to_numpy(),
        csm=csm.to_numpy(),
    ).sort_values("csm", ascending=False).reset_index(drop=True)
    # 게이트 정보를 화면까지 들고 간다. 몇 종을 왜 뺐는지 밝히지 않으면 "전 상품을
    # 다 본 순위"로 읽힌다 (attrs 는 merge·sort 를 거치며 사라질 수 있어 다시 단다).
    scored.attrs["gate"] = {**lift.attrs.get("gate", {}),
                            "elasticity_fill": fill,
                            "measured": int(measured.sum())}
    return scored


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
    "재고가 그때만 있었던 것"이라 번들로 걸 수 없다. 요건은 lift 게이트와 같은
    이유로 창 길이에 비례한다 — 8일이라는 절대값은 2년 창에서 아무것도 거르지 못한다.
    """
    if csm.empty or sales.empty:
        return []

    window_days = int(sales["date"].nunique())
    min_days = max(S.BUNDLE_MIN_DAYS,
                   math.ceil(window_days * S.BUNDLE_MIN_DAY_RATIO))
    days = sales.groupby("item_id")["date"].nunique()
    regular = set(days.loc[days >= min_days].index)
    if not regular:
        return []
    pool = csm.loc[csm["item_id"].isin(regular)]

    # VIP 티어만 후보로 둔다. 물량 축(0.30)이 있어서 티어를 열어 두면 긴 구간에서
    # '공기밥 × 특산품' 같은 조합이 상위로 올라온다 — 카드에 '프리미엄 × 프리미엄'
    # 이라고 써 놓고 실제로는 일반 상품을 미는 셈이 된다.
    vip_tiers = (S.TIER_PREMIUM, S.TIER_BUNDLE)
    room = top_items(pool, S.CH_ROOM, n=top_n * 2, tiers=vip_tiers, exclude_comp=True)
    local = top_items(pool, S.CH_LOCAL, n=top_n * 2, tiers=vip_tiers, exclude_comp=True)
    if room.empty or local.empty:
        # 티어 필터로 후보가 사라지면 번들을 만들지 못하는 것보다 낫다 — 전 티어로
        # 한 번 더 시도하되, 카드에는 실제 티어가 그대로 표기된다.
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
