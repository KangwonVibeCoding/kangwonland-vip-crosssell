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
        # 컬럼은 있는데 값이 전부 결측인 경우가 있다 — 2026년 ARS 전처리본에는
        # recv_total 이 없어서 concat 후 그 구간이 통째로 NaN 이다. 컬럼 존재만
        # 확인하면 상관계수 NaN 행이 표에 그대로 실린다.
        if pd.to_numeric(ars_idx[metric], errors="coerce").notna().sum() < 3:
            continue
        for channel in S.CHANNELS:
            for vip_only in (False, True):
                qty = daily_qty(sales, channel, vip_only=vip_only)
                x, y = align(ars_idx[metric], qty)
                # align 은 날짜만 맞춘다. 값이 결측인 날은 상관 계산에서 빠지므로
                # 유효 쌍 수로 판단해야 n_days 가 정직해진다.
                pairs = pd.concat([x, y], axis=1).apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                if len(pairs) < 3:
                    continue
                x, y = pairs.iloc[:, 0], pairs.iloc[:, 1]
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
    """가설 검증 배너용 — 채널별 대표 상관계수 (전체 상품).

    기준축은 recv_total(내국인 총 접수자)이다. 다만 2026년 ARS 전처리본에는
    그 컬럼이 없어 corr_table 에서 통째로 빠지므로, 없으면 tickets 로 내려간다
    — 배너가 빈 값으로 뜨는 것보다 근거 지표를 바꿔 표시하는 편이 낫다.
    호출자는 `headline_metric()` 으로 어느 축이 쓰였는지 확인할 수 있다.
    """
    if corr.empty:
        return {}
    metric = headline_metric(corr)
    if metric is None:
        return {}
    base = corr.loc[(corr["metric"] == metric) & (~corr["vip_only"])]
    return {r.channel: float(r.pearson) for r in base.itertuples()}


def headline_metric(corr: pd.DataFrame) -> str | None:
    """headline_corr 가 실제로 사용한 유입 지표명. 배지 표기에 쓴다."""
    if corr.empty:
        return None
    for metric in ("recv_total", "tickets"):
        if (corr["metric"] == metric).any():
            return metric
    return None


# ── 요일 교란 통제 ────────────────────────────────────────────────────
# r=0.80 은 그 자체로 방어되지 않는다. 토요일엔 유입도 매출도 많으므로 **요일이
# 양변을 동시에 밀어올리는 공통 원인**일 수 있고, 그렇다면 "유입 → 소비" 라는
# 인과 서사가 흔들린다. 아래 함수들이 그 가능성을 수치로 잘라낸다.
#
# scipy 를 쓰지 않는다 — 필요한 것은 numpy 의 최소자승과 재표집뿐이고,
# Cloud 빌드 의존성을 늘리지 않는 것이 이 프로젝트의 방침이다.
def residual_on_dow(values: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """요일로 설명되는 변동을 제거한 잔차.

    더미는 6개만 만든다(월요일이 기준 범주). 7개를 넣으면 절편과 완전공선이 된다.
    """
    y = np.asarray(values, dtype="float64")
    d = np.asarray(dow)
    X = np.column_stack([np.ones(len(y)), *[(d == k).astype(float) for k in range(1, 7)]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _paired(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """두 일별 시계열을 공통 날짜로 맞추고 결측 쌍을 버린다. (x, y, dayofweek)"""
    a, b = align(x, y)
    pairs = pd.concat([a, b], axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    if pairs.empty:
        return np.array([]), np.array([]), np.array([])
    dates = pd.DatetimeIndex(pairs.index)
    return (pairs.iloc[:, 0].to_numpy("float64"),
            pairs.iloc[:, 1].to_numpy("float64"),
            dates.dayofweek.to_numpy())


def partial_corr_dow(x: pd.Series, y: pd.Series) -> float:
    """요일을 통제한 편상관 — 잔차끼리의 피어슨 상관.

    실측 (구간 A, recv_total 기준): 카지노 식음 0.370 / 룸서비스 0.558 /
    지역특산품 0.176. **원 상관과 순서가 뒤집힌다**(0.801 > 0.740 → 0.558 > 0.370).
    """
    xv, yv, dow = _paired(x, y)
    if len(xv) < S.CONFOUND_MIN_DAYS:
        return float("nan")
    return pearson(residual_on_dow(xv, dow), residual_on_dow(yv, dow))


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(B, n) 두 행렬의 행별 피어슨 상관 → (B,).

    재표집 통계는 반복이 10,000회다. 행마다 `pearson()` 을 부르면 pandas 변환
    비용이 곱해져 3채널에 5초가 넘는다(실측) — 화면에서 매 리런마다 치를 수 없는
    값이다. 잔차는 유한한 float 로 들어오는 것이 보장되므로 행렬로 한 번에 센다.
    """
    ca = a - a.mean(axis=1, keepdims=True)
    cb = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((ca * ca).sum(axis=1) * (cb * cb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (ca * cb).sum(axis=1) / denom, np.nan)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, *,
                 resamples: int = S.CONFOUND_RESAMPLES,
                 seed: int = S.CONFOUND_SEED) -> tuple[float, float]:
    """상관계수의 부트스트랩 95% CI. n=31 표본이 갖는 불확실성 폭을 드러낸다."""
    a, b = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if a.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(resamples, a.size))
    vals = _rowwise_pearson(a[idx], b[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_p(x: np.ndarray, y: np.ndarray, *,
                  resamples: int = S.CONFOUND_RESAMPLES,
                  seed: int = S.CONFOUND_SEED) -> float:
    """H0: 두 계열은 무관. y 를 섞어 |r| 이 관측치 이상 나오는 비율.

    분포 가정을 하지 않는다 — n=31 에 정규성을 가정한 t 검정보다 정직하다.
    분자·분모에 1을 더하는 것은 관측 자체를 하나의 재배치로 세는 관례다
    (p=0 이라는 불가능한 값을 보고하지 않게 된다).
    """
    a, b = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if a.size < 3:
        return float("nan")
    observed = abs(pearson(a, b))
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.broadcast_to(b, (resamples, b.size)), axis=1)
    vals = np.abs(_rowwise_pearson(np.broadcast_to(a, shuffled.shape), shuffled))
    hits = int(np.count_nonzero(vals[np.isfinite(vals)] >= observed))
    return (hits + 1) / (resamples + 1)


def confound_verdict(p: float) -> str:
    """p 값 → 판정 문구. 화면·처방·README 가 같은 단어를 쓰게 한 곳에서 만든다."""
    if not np.isfinite(p):
        return "판정 불가"
    for threshold, label in S.CONFOUND_VERDICTS:
        if p < threshold:
            return label
    return S.CONFOUND_VERDICT_NULL


def confound_metric(ars: pd.DataFrame, min_days: int = S.CONFOUND_MIN_DAYS) -> str | None:
    """편상관의 기준 유입축. recv_total 우선, 없으면 tickets (배너와 같은 규약)."""
    if ars.empty:
        return None
    for metric in ("recv_total", "tickets"):
        if metric not in ars.columns:
            continue
        if pd.to_numeric(ars[metric], errors="coerce").notna().sum() >= min_days:
            return metric
    return None


def confound_table(ars: pd.DataFrame, sales: pd.DataFrame, *,
                   resamples: int = S.CONFOUND_RESAMPLES,
                   seed: int = S.CONFOUND_SEED,
                   min_days: int = S.CONFOUND_MIN_DAYS) -> pd.DataFrame:
    """채널별 원 상관 vs 요일 통제 편상관 — 부트스트랩 CI 와 순열검정 p 포함.

    반환은 **편상관 내림차순**이다. 원 상관 순서와 다르다는 사실 자체가 이 표의
    메시지이므로, 정렬을 raw 로 되돌리지 말 것.

    실측 (구간 A, n=31):
      카지노 식음 0.801 → 0.370 (p 0.044)  룸서비스 0.740 → 0.558 (p 0.002)
      지역특산품 0.380 → 0.176 (p 0.357)
    """
    cols = ["channel", "channel_label", "n_days", "raw", "partial", "delta",
            "ci_lo", "ci_hi", "p", "verdict", "metric", "metric_label"]
    if ars.empty or sales.empty or "date" not in ars.columns:
        return pd.DataFrame(columns=cols)

    ars_idx = ars.set_index("date")
    metric = confound_metric(ars_idx, min_days)
    if metric is None:
        return pd.DataFrame(columns=cols)
    metric_label = {"recv_total": "내국인 총 접수자",
                    "tickets": "입장권 구매 건수"}.get(metric, metric)
    signal = pd.to_numeric(ars_idx[metric], errors="coerce").dropna()

    rows: list[dict] = []
    for channel in S.CHANNELS:
        xv, yv, dow = _paired(signal, daily_qty(sales, channel))
        # 요일 더미를 세우려면 표본이 어느 정도 있어야 한다. 짧은 구간에서
        # 억지로 계산하면 잔차가 거의 0이 되어 편상관이 요동친다.
        if len(xv) < min_days or len(np.unique(dow)) < 5:
            continue
        raw = pearson(xv, yv)
        rx, ry = residual_on_dow(xv, dow), residual_on_dow(yv, dow)
        partial = pearson(rx, ry)
        lo, hi = bootstrap_ci(rx, ry, resamples=resamples, seed=seed)
        p = permutation_p(rx, ry, resamples=resamples, seed=seed)
        rows.append({
            "channel": channel,
            "channel_label": S.CHANNEL_LABEL[channel],
            "n_days": int(len(xv)),
            "raw": raw,
            "partial": partial,
            "delta": partial - raw,
            "ci_lo": lo,
            "ci_hi": hi,
            "p": p,
            "verdict": confound_verdict(p),
            "metric": metric,
            "metric_label": metric_label,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows)
            .sort_values("partial", ascending=False, na_position="last")
            .reset_index(drop=True))[cols]


def confound_flips_order(table: pd.DataFrame) -> bool:
    """원 상관 1위와 편상관 1위가 다른가 — 배너에서 서사를 바꿀지 판단한다."""
    if len(table) < 2 or table[["raw", "partial"]].isna().to_numpy().any():
        return False
    return (table.loc[table["raw"].idxmax(), "channel"]
            != table.loc[table["partial"].idxmax(), "channel"])


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
