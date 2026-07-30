"""통계 프리미티브 — 상관계수, 요일·월 인덱스, 일별 집계.

scipy 를 쓰지 않고 직접 구현한다. Streamlit Cloud 빌드 의존성을 줄이는 것이
목적이며, 필요한 것은 pearson / spearman 두 개뿐이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S


def pearson(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """표본 피어슨 상관계수. 표본 부족·분산 0 이면 nan."""
    a = np.asarray(pd.Series(x).to_numpy(), dtype="float64")
    b = np.asarray(pd.Series(y).to_numpy(), dtype="float64")
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else float("nan")


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """순위 상관계수. 동순위는 평균 순위로 처리한다(pandas rank 기본)."""
    sx, sy = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    both = pd.concat([sx, sy], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    if len(both) < 3:
        return float("nan")
    return pearson(both.iloc[:, 0].rank(), both.iloc[:, 1].rank())


def daily_qty(sales: pd.DataFrame, channel: str | None = None,
              *, vip_only: bool = False) -> pd.Series:
    """일별 판매수량 합계. index=date(datetime), name='qty'."""
    df = sales if channel is None else sales.loc[sales["channel"] == channel]
    if vip_only:
        df = df.loc[df["tier"].isin([S.TIER_PREMIUM, S.TIER_BUNDLE])]
    if df.empty:
        return pd.Series(dtype="float64", name="qty")
    return df.groupby("date")["qty"].sum().sort_index().rename("qty")


def daily_tx(sales: pd.DataFrame, channel: str | None = None) -> pd.Series:
    """일별 거래(행) 수 — 주문 다양성 = 체류 인원 프록시."""
    df = sales if channel is None else sales.loc[sales["channel"] == channel]
    if df.empty:
        return pd.Series(dtype="float64", name="tx")
    return df.groupby("date").size().sort_index().rename("tx")


def dow_index(series: pd.Series) -> pd.Series:
    """요일 인덱스 = 요일별 평균 / 전체 평균. index=0(월)…6(일).

    실측 예: ARS 구매건수 토 1.45 / 일 1.31, 특산품 일 1.68 / 목 0.66
    """
    if series.empty:
        return pd.Series([np.nan] * 7, index=range(7), name="dow_index")
    idx = pd.to_datetime(pd.Series(series.index))
    grouped = series.to_numpy(dtype="float64")
    frame = pd.DataFrame({"dow": idx.dt.dayofweek.to_numpy(), "v": grouped})
    base = frame["v"].mean()
    if not base:
        return pd.Series([np.nan] * 7, index=range(7), name="dow_index")
    out = frame.groupby("dow")["v"].mean() / base
    return out.reindex(range(7)).rename("dow_index")


def month_index(series: pd.Series) -> pd.Series:
    """월 인덱스 = 월별 평균 / 전체 평균. index=1…12.

    실측 예: 특산품 9월 1.48(추석 선물), 룸서비스 1월 1.23(설·스키 성수기)
    """
    if series.empty:
        return pd.Series([np.nan] * 12, index=range(1, 13), name="month_index")
    idx = pd.to_datetime(pd.Series(series.index))
    frame = pd.DataFrame({"m": idx.dt.month.to_numpy(),
                          "v": series.to_numpy(dtype="float64")})
    base = frame["v"].mean()
    if not base:
        return pd.Series([np.nan] * 12, index=range(1, 13), name="month_index")
    out = frame.groupby("m")["v"].mean() / base
    return out.reindex(range(1, 13)).rename("month_index")


def align(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """두 일별 시계열을 공통 날짜로 정렬한다 (내부 조인)."""
    common = left.index.intersection(right.index)
    return left.loc[common].sort_index(), right.loc[common].sort_index()


def corr_table(ars: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """가설 검증 표 — ARS 유입 지표 × 채널 판매량 상관.

    ARS 와 판매 데이터가 겹치는 날짜만 사용한다. 겹치는 날이 3일 미만이면 빈
    DataFrame 을 반환하고, 호출자가 안내 메시지로 처리한다.

    실측 기대값 (2024-12, 31일):
      recv_total ↔ casino_fnb  Pearson 0.801 / Spearman 0.860
      tickets    ↔ roomservice Pearson 0.733
    """
    if ars.empty or sales.empty:
        return pd.DataFrame()

    inflow_metrics = {
        "recv_total": "내국인 총 접수자",
        "tickets": "당첨자 입장권 구매 건수",
        "buy_rate": "당첨자 입장권 구매율",
    }
    rows: list[dict] = []
    ars_idx = ars.set_index("date")
    for metric, label in inflow_metrics.items():
        if metric not in ars_idx.columns:
            continue
        for channel in S.CHANNELS:
            for vip_only in (False, True):
                qty = daily_qty(sales, channel, vip_only=vip_only)
                x, y = align(ars_idx[metric], qty)
                if len(x) < 3:
                    continue
                rows.append({
                    "metric": metric,
                    "metric_label": label,
                    "channel": channel,
                    "channel_label": S.CHANNEL_LABEL[channel],
                    "vip_only": vip_only,
                    "n_days": int(len(x)),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                })
    return pd.DataFrame(rows)


def headline_corr(corr: pd.DataFrame) -> dict[str, float]:
    """가설 검증 배너용 — 채널별 대표 상관계수 (recv_total 기준, 전체 상품)."""
    if corr.empty:
        return {}
    base = corr.loc[(corr["metric"] == "recv_total") & (~corr["vip_only"])]
    return {r.channel: float(r.pearson) for r in base.itertuples()}


def channel_cross_corr(sales: pd.DataFrame) -> pd.DataFrame:
    """채널 간 일별 판매량 상관 — 교차판매 여지(느슨한 연결 = 기회) 판단용."""
    series = {ch: daily_qty(sales, ch) for ch in S.CHANNELS}
    rows: list[dict] = []
    channels = list(S.CHANNELS)
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            a, b = align(series[channels[i]], series[channels[j]])
            if len(a) < 3:
                continue
            rows.append({
                "a": channels[i], "b": channels[j],
                "a_label": S.CHANNEL_LABEL[channels[i]],
                "b_label": S.CHANNEL_LABEL[channels[j]],
                "pearson": pearson(a, b), "spearman": spearman(a, b),
                "n_days": int(len(a)),
            })
    return pd.DataFrame(rows)
