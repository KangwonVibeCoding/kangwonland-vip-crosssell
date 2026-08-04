"""유입 → 소비 시차(래그) 분석.

실데이터에서 발견한 것을 기능화한 모듈이다. ARS 유입(D)과 채널 판매량(D+k)의
상관을 k=0..3 으로 훑으면 채널마다 반응 시점이 다르다:

    카지노 식음   D+0 +0.796  (당일 즉시 소비 — 객장 안에 있으니 당연)
    룸서비스      D+0 +0.733  (당일 숙박 소비)
    지역특산품    D+1 +0.467  (익일! = 체크아웃 선물 구매)

특산품이 D+1 이라는 사실이 마케팅 처방을 바꾼다: 유입 피크일 당일엔 F&B,
**익일 오전엔 특산품 쿠폰**. 요일 인덱스(특산품 일요일 1.68)와도 일관된다.

⚠ 위 상관은 **요일을 통제하지 않은 값**이다. 요일 더미를 빼면 카지노 식음
0.796 → 0.370, 룸서비스 0.733 → 0.558 로 순서가 뒤집힌다(`stats.confound_table`).
그래서 `prescriptions()` 는 D+0 집행 순서를 래그 상관이 아니라 편상관으로 매긴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S
from src.analysis import stats


def _shift_corr(inflow_series: pd.Series, qty: pd.Series, lag: int) -> tuple[float, float, int]:
    """유입(D)과 판매(D+lag)의 상관. (pearson, spearman, n)"""
    if inflow_series.empty or qty.empty:
        return float("nan"), float("nan"), 0
    shifted = qty.copy()
    shifted.index = pd.to_datetime(pd.Series(qty.index)) - pd.Timedelta(days=lag)
    x, y = stats.align(inflow_series, shifted)
    if len(x) < 3:
        return float("nan"), float("nan"), int(len(x))
    return stats.pearson(x, y), stats.spearman(x, y), int(len(x))


def lag_matrix(inflow: pd.DataFrame, sales: pd.DataFrame,
               max_lag: int = S.MAX_LAG_DEFAULT,
               *, use_tickets: bool = True) -> pd.DataFrame:
    """채널 × 시차 상관 행렬 (long 형태).

    반환 컬럼: channel, channel_label, lag, pearson, spearman, n_days

    유입 신호는 가능하면 `tickets`(실제 입장 건수)를 쓴다 — 실측 기준값이 이
    지표로 계산된 것이라 회귀 테스트와 맞아떨어진다. 없으면 inflow 점수로 대체한다.
    """
    if inflow.empty or sales.empty:
        return pd.DataFrame(columns=["channel", "channel_label", "lag",
                                     "pearson", "spearman", "n_days"])

    if use_tickets and "tickets" in inflow.columns:
        signal = inflow.dropna(subset=["tickets"]).set_index("date")["tickets"]
    else:
        signal = inflow.set_index("date")["inflow"]
    signal = pd.to_numeric(signal, errors="coerce").dropna()

    rows: list[dict] = []
    for channel in S.CHANNELS:
        qty = stats.daily_qty(sales, channel)
        if qty.empty:
            continue
        for lag in range(max_lag + 1):
            p, sp, n = _shift_corr(signal, qty, lag)
            rows.append({
                "channel": channel,
                "channel_label": S.CHANNEL_LABEL[channel],
                "lag": lag,
                "pearson": p,
                "spearman": sp,
                "n_days": n,
            })
    return pd.DataFrame(rows)


def best_lags(matrix: pd.DataFrame) -> pd.DataFrame:
    """채널별 최적 시차 — pearson 이 최대인 lag.

    실측 기대값: casino_fnb→0, roomservice→0, local_goods→1
    """
    if matrix.empty:
        return pd.DataFrame(columns=["channel", "channel_label", "best_lag",
                                     "pearson", "n_days"])
    valid = matrix.dropna(subset=["pearson"])
    if valid.empty:
        return pd.DataFrame(columns=["channel", "channel_label", "best_lag",
                                     "pearson", "n_days"])
    idx = valid.groupby("channel")["pearson"].idxmax()
    out = valid.loc[idx].rename(columns={"lag": "best_lag"})
    return out[["channel", "channel_label", "best_lag", "pearson", "n_days"]] \
        .sort_values("channel").reset_index(drop=True)


def best_lag_by_item(inflow: pd.DataFrame, sales: pd.DataFrame,
                     max_lag: int = S.MAX_LAG_DEFAULT,
                     *, min_days: int = 10, min_qty: int = 30) -> pd.DataFrame:
    """상품별 최적 시차 → "D+0 공략 상품" / "D+1 공략 상품" 그룹.

    상품 단위 시계열은 짧고 희소하므로 필터를 둔다: 판매 일수 min_days 이상,
    총 수량 min_qty 이상. 걸러진 상품은 조용히 버리지 않고 UI 에서 몇 개가
    제외됐는지 표기할 수 있게 반환 프레임의 attrs 에 기록한다.
    """
    empty = pd.DataFrame(columns=["item_id", "item", "channel", "channel_label",
                                  "tier", "best_lag", "pearson", "total_qty", "n_days"])
    if inflow.empty or sales.empty:
        return empty

    if "tickets" in inflow.columns:
        signal = inflow.dropna(subset=["tickets"]).set_index("date")["tickets"]
    else:
        signal = inflow.set_index("date")["inflow"]
    signal = pd.to_numeric(signal, errors="coerce").dropna()
    if len(signal) < min_days:
        return empty

    # 유입 신호가 있는 날짜로 한정해 상품별 일별 수량을 만든다
    window = sales.loc[sales["date"].isin(set(signal.index))]
    if window.empty:
        return empty

    grouped = window.groupby(["item_id", "date"])["qty"].sum()
    totals = window.groupby("item_id")["qty"].agg(["sum", "count"])
    eligible = totals.loc[(totals["sum"] >= min_qty) & (totals["count"] >= min_days)].index
    dropped = int(len(totals) - len(eligible))
    if len(eligible) == 0:
        out = empty
        out.attrs["dropped"] = dropped
        return out

    meta = (
        window.loc[window["item_id"].isin(eligible)]
        .groupby("item_id", as_index=False)
        .agg(item=("item", "first"), channel=("channel", "first"),
             tier=("tier", "first"), total_qty=("qty", "sum"))
    )

    rows: list[dict] = []
    for item_id in eligible:
        series = grouped.loc[item_id]
        best_p, best_k, best_n = float("nan"), 0, 0
        for lag in range(max_lag + 1):
            p, _, n = _shift_corr(signal, series, lag)
            if np.isfinite(p) and (not np.isfinite(best_p) or p > best_p):
                best_p, best_k, best_n = p, lag, n
        rows.append({"item_id": item_id, "best_lag": best_k,
                     "pearson": best_p, "n_days": best_n})

    out = meta.merge(pd.DataFrame(rows), on="item_id", how="inner")
    out = out.assign(channel_label=out["channel"].map(S.CHANNEL_LABEL))
    out = out.sort_values("pearson", ascending=False).reset_index(drop=True)
    out.attrs["dropped"] = dropped
    return out


def lag_groups(item_lags: pd.DataFrame, lag: int, n: int = 10,
               *, min_pearson: float = 0.1) -> pd.DataFrame:
    """특정 시차에 반응하는 상품 상위 N개."""
    if item_lags.empty:
        return item_lags
    sel = item_lags.loc[
        (item_lags["best_lag"] == lag) & (item_lags["pearson"] >= min_pearson)
    ]
    return sel.head(n).reset_index(drop=True)


def prescriptions(best: pd.DataFrame, dow_profile: pd.DataFrame,
                  month_profile: pd.DataFrame,
                  confound: pd.DataFrame | None = None) -> list[dict]:
    """실행 처방 카드 3장의 내용을 데이터에서 생성한다.

    수치를 문장에 박아 넣는 대신 계산 결과를 그대로 옮긴다 — 필터가 바뀌면
    처방도 함께 바뀌어야 하기 때문이다.

    `confound`(= `stats.confound_table()`)를 주면 **D+0 카드의 우선순위를 원 래그
    상관이 아니라 요일 통제 편상관으로 매긴다.** 원 상관 순서는 카지노 식음이
    1위지만 요일을 빼면 룸서비스가 1위이고, 요일은 마케팅으로 바꿀 수 없는
    변수이므로 집행 순서는 통제 후 순서를 따라야 한다.
    """
    cards: list[dict] = []
    lag_of = {r.channel: int(r.best_lag) for r in best.itertuples()} if not best.empty else {}
    r_of = {r.channel: float(r.pearson) for r in best.itertuples()} if not best.empty else {}

    conf = confound if confound is not None else pd.DataFrame()
    partial_of = {r.channel: float(r.partial) for r in conf.itertuples()} \
        if not conf.empty else {}
    p_of = {r.channel: float(r.p) for r in conf.itertuples()} if not conf.empty else {}
    verdict_of = {r.channel: str(r.verdict) for r in conf.itertuples()} \
        if not conf.empty else {}

    def _p_text(channel: str) -> str:
        p = p_of.get(channel)
        if p is None or not np.isfinite(p):
            return ""
        return " p<0.001" if p < 0.001 else f" p={p:.3f}"

    # 편상관이 있으면 그 내림차순, 없으면 래그 상관 내림차순
    def _order(channels: list[str]) -> list[str]:
        if partial_of:
            return sorted(channels,
                          key=lambda c: (partial_of.get(c, float("-inf")),
                                         r_of.get(c, float("-inf"))),
                          reverse=True)
        return sorted(channels, key=lambda c: r_of.get(c, float("-inf")), reverse=True)

    same_day = _order([c for c, k in lag_of.items() if k == 0])
    next_day = _order([c for c, k in lag_of.items() if k >= 1])

    if same_day:
        top = same_day[0]
        if top in partial_of:
            # 조사를 붙이지 않는 문형으로 쓴다 — 채널 이름이 데이터에서 오므로
            # '카지노 식음는' 같은 문장이 발표 화면에 뜰 수 있다.
            detail = (f"당일 집행 1순위는 {S.CHANNEL_LABEL[top]} — 요일을 통제해도 "
                      f"남는 관계가 가장 견고하다({verdict_of.get(top, '')}). "
                      "나머지 당일 채널은 그다음 순서로 붙인다.")
            evidence = " · ".join(
                f"{S.CHANNEL_LABEL[c]} 편상관 {partial_of[c]:+.3f}{_p_text(c)}"
                f" (원 r {r_of[c]:+.3f})"
                for c in same_day if c in partial_of
            )
        else:
            detail = "카지노 프리미엄 F&B와 룸서비스 VIP 어메니티를 당일 집행한다."
            evidence = " · ".join(f"{S.CHANNEL_LABEL[c]} r={r_of[c]:+.3f}" for c in same_day)
        cards.append({
            "title": "D+0 · 유입 피크일 당일",
            "targets": [S.CHANNEL_LABEL[c] for c in same_day],
            "detail": detail,
            "evidence": evidence,
        })
    if next_day:
        evidence = " · ".join(
            f"{S.CHANNEL_LABEL[c]} D+{lag_of[c]} r={r_of[c]:+.3f}" for c in next_day
        )
        detail = "체크아웃 동선에 지역특산품 선물 패키지 쿠폰을 푸시한다."
        # 래그 상관만으로는 D+1 을 주장할 수 없다 — D+1 과 D+0 의 차이는 n≈30 에서
        # 통계적으로 구별되지 않고, 요일을 통제하면 특산품 편상관은 유의하지 않다.
        # 근거를 요일 프로파일로 정직하게 옮겨 적는다.
        weak = [c for c in next_day
                if verdict_of.get(c) == S.CONFOUND_VERDICT_NULL]
        if weak:
            evidence += " · " + " · ".join(
                f"요일 통제 시 {S.CHANNEL_LABEL[c]} {partial_of[c]:+.3f}{_p_text(c)}"
                for c in weak
            )
            detail += " 근거는 래그 상관이 아니라 요일 프로파일(토 저조 → 일 급등)이다."
        cards.append({
            "title": "D+1 · 익일 오전",
            "targets": [S.CHANNEL_LABEL[c] for c in next_day],
            "detail": detail,
            "evidence": evidence,
        })

    # 계절 오버레이 — 채널별 최고 월을 데이터에서 찾는다
    # 요일 처방 — 유입 피크 요일과 각 채널의 피크 요일이 어긋나는 지점이 기회다
    if not dow_profile.empty:
        dow_cols = [d for d in S.DOW_NAMES if d in dow_profile.columns]
        peaks: dict[str, tuple[str, float]] = {}
        for _, row in dow_profile.iterrows():
            vals = {d: row[d] for d in dow_cols if pd.notna(row[d])}
            if vals:
                top = max(vals, key=lambda d: vals[d])
                peaks[str(row["label"])] = (top, float(vals[top]))
        if peaks:
            cards.append({
                "title": "요일 배분",
                "targets": ["전 채널"],
                "detail": "유입 피크 요일과 채널 소비 피크 요일이 어긋나는 곳에 여지가 있다.",
                "evidence": " · ".join(
                    f"{label} {d}({v:.2f})" for label, (d, v) in peaks.items()
                ),
            })

    # 계절 오버레이 — 채널별 최고 월을 데이터에서 찾는다
    if not month_profile.empty:
        month_cols = [m for m in range(1, 13) if m in month_profile.columns]
        season_bits: list[str] = []
        for _, row in month_profile.iterrows():
            vals = {m: row[m] for m in month_cols if pd.notna(row[m])}
            if vals:
                peak = max(vals, key=lambda m: vals[m])
                season_bits.append(f"{row['label']} {peak}월({vals[peak]:.2f})")
        if season_bits:
            cards.append({
                "title": "계절 오버레이",
                "targets": ["전 채널"],
                "detail": "채널별 성수기가 다르다. 월별로 집중 채널을 바꿔 배분한다.",
                "evidence": " · ".join(season_bits),
            })
    return cards
