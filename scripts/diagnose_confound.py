"""요일 교란 통제 + 표본 불확실성 진단 — README 가설 검증 절의 수치를 재현한다.

왜 필요한가:
  "r=0.80" 은 그 자체로는 방어되지 않는다. 토요일엔 유입도 많고 매출도 많으므로
  **요일이 양변을 동시에 밀어올리는 공통 원인**일 수 있고, 그렇다면 "유입 → 소비"
  라는 인과 서사가 흔들린다. 여기서 그 가능성을 수치로 확인한다.

무엇을 계산하는가:
  1) 편상관 — 요일 더미(6개)로 양변을 회귀한 **잔차끼리의 상관**
  2) 부트스트랩 95% CI — n=31 표본이 갖는 불확실성 폭
  3) 순열검정 p — 관측된 관계가 우연일 확률
  4) 특산품 D+0 vs D+1 **차이**의 CI — 발표 핵심 주장이 통계적으로 버티는지

무엇을 발견했는가 (README 에 반영됨):
  - 원 상관의 절반 이상이 주말 효과였다 (카지노 0.801 → 0.370)
  - **요일을 통제하면 순서가 뒤집힌다** — 룸서비스 0.558 > 카지노 식음 0.370
  - `(D+1 r) − (D+0 r)` 의 CI 가 0을 포함 → 래그 상관만으로는 D+1 을 주장할 수 없다.
    D+1 의 근거는 요일 프로파일(특산품 토 0.92 → 일 1.63)이 지탱한다

scipy 를 쓰지 않는다 — Cloud 빌드를 가볍게 유지하려는 프로젝트 방침이고,
필요한 것은 numpy 의 최소자승과 재표집뿐이다.

사용법:
    python scripts/diagnose_confound.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

from config import settings as S                         # noqa: E402
from src.analysis import stats                           # noqa: E402
from src.data import loaders                             # noqa: E402

SEED = 42
B = 10_000
DOW_NAMES = ("월", "화", "수", "목", "금", "토", "일")
CHANNELS = ((S.CH_CASINO, "카지노 식음"), (S.CH_ROOM, "룸서비스"), (S.CH_LOCAL, "지역특산품"))

rng = np.random.default_rng(SEED)


# ── 통계 프리미티브 (scipy 없이) ───────────────────────────────────────
def pearson(x: np.ndarray, y: np.ndarray) -> float:
    a, b = x - x.mean(), y - y.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom else float("nan")


def residual_on_dow(y: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """요일로 설명되는 변동을 제거한 잔차.

    더미는 6개만 만든다(월요일이 기준 범주). 7개를 넣으면 절편과 공선성이 생긴다.
    """
    dummies = [(dow == d).astype(float) for d in range(1, 7)]
    X = np.column_stack([np.ones(len(y)), *dummies])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def bootstrap_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    vals = np.array([pearson(x[i], y[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_p(x: np.ndarray, y: np.ndarray) -> float:
    """H0: 두 계열은 무관. y 를 섞어 |r| 이 관측치 이상 나오는 비율."""
    observed = abs(pearson(x, y))
    hits = sum(abs(pearson(x, rng.permutation(y))) >= observed for _ in range(B))
    return (hits + 1) / (B + 1)


def diff_ci(x: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> tuple[float, float, float]:
    """같은 x 에 대한 두 상관의 차이(r1 − r0) CI. 0을 포함하면 구별 불가."""
    idx = rng.integers(0, len(x), size=(B, len(x)))
    vals = np.array([pearson(x[i], y1[i]) - pearson(x[i], y0[i]) for i in idx])
    vals = vals[np.isfinite(vals)]
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)),
            float((vals <= 0).mean()))


def main() -> int:
    data, sources = loaders.load_all_impl()
    if sources["ars"] not in S.REAL_SOURCES or sources["sales"] not in S.REAL_SOURCES:
        print(f"실데이터가 없습니다 (ars={sources['ars']}, sales={sources['sales']}).")
        print("data/incoming 에 CSV 를 넣고 scripts/ingest.py 를 먼저 실행하세요.")
        return 1

    period = S.PERIODS[S.PERIOD_A]
    start, end = pd.Timestamp(period["start"]), pd.Timestamp(period["end"])
    ars = (data["ars"].loc[data["ars"]["date"].between(start, end)]
           .set_index("date").sort_index())
    sales = data["sales"].loc[data["sales"]["date"].between(start, end)]

    print("=" * 78)
    print(f"구간 A ({start:%Y-%m-%d} ~ {end:%Y-%m-%d})  ·  부트스트랩/순열 {B:,}회  ·  seed={SEED}")
    print("=" * 78)

    print("\n[1] 요일 통제 편상관 — 유입(recv_total) vs 채널 판매량\n")
    print(f"{'채널':<12}{'원 r':>8}{'통제 후':>10}{'감소':>8}{'95% CI':>22}{'p':>9}")
    print("-" * 78)
    for channel, label in CHANNELS:
        x_s, y_s = stats.align(ars["recv_total"], stats.daily_qty(sales, channel))
        x, y = x_s.to_numpy(float), y_s.to_numpy(float)
        dow = x_s.index.dayofweek.to_numpy()

        raw = pearson(x, y)
        rx, ry = residual_on_dow(x, dow), residual_on_dow(y, dow)
        partial = pearson(rx, ry)
        lo, hi = bootstrap_ci(rx, ry)
        print(f"{label:<12}{raw:>8.3f}{partial:>10.3f}{partial - raw:>8.3f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>22}{permutation_p(rx, ry):>9.4f}")

    print("\n[2] 원 상관의 부트스트랩 95% CI (요일 통제 전)\n")
    for channel, label in CHANNELS:
        x_s, y_s = stats.align(ars["recv_total"], stats.daily_qty(sales, channel))
        x, y = x_s.to_numpy(float), y_s.to_numpy(float)
        lo, hi = bootstrap_ci(x, y)
        print(f"  {label:<12} r={pearson(x, y):.3f}   "
              f"95% CI [{lo:.3f}, {hi:.3f}]   폭 {hi - lo:.3f}   n={len(x)}")

    print("\n[3] 특산품 D+0 vs D+1 — 핵심 주장이 통계적으로 버티는가\n")
    local = stats.daily_qty(sales, S.CH_LOCAL)
    shifted: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in range(4):
        s = local.copy()
        s.index = s.index - pd.Timedelta(days=k)          # D+k 판매를 D 로 당긴다
        x_s, y_s = stats.align(ars["tickets"], s)
        shifted[k] = (x_s.to_numpy(float), y_s.to_numpy(float))
        print(f"  D+{k}   r = {pearson(*shifted[k]):+.3f}   n = {len(x_s)}")

    n = min(len(shifted[0][0]), len(shifted[1][0]))
    lo, hi, p_le0 = diff_ci(shifted[1][0][:n], shifted[0][1][:n], shifted[1][1][:n])
    print(f"\n  (D+1 r) − (D+0 r) 의 95% CI: [{lo:+.3f}, {hi:+.3f}]")
    print(f"  차이 ≤ 0 일 확률: {p_le0:.3f}")
    print(f"  → {'구별 가능' if lo > 0 else '통계적으로 구별 불가 — 요일 프로파일이 근거를 지탱한다'}")

    print("\n[4] 보강 증거 — 요일 프로파일\n")
    print(f"{'계열':<12}" + "".join(f"{d:>7}" for d in DOW_NAMES) + f"{'피크':>6}")
    print("-" * 78)
    series = [("ARS 유입", stats.dow_index(ars["tickets"]))]
    series += [(label, stats.dow_index(stats.daily_qty(sales, ch))) for ch, label in CHANNELS]
    for label, idx in series:
        peak = DOW_NAMES[int(idx.idxmax())]
        print(f"{label:<12}" + "".join(f"{idx.get(i, float('nan')):>7.2f}" for i in range(7))
              + f"{peak:>6}")

    print("\n  특산품만 토요일에 평균 이하(0.92)로 죽어 있다가 일요일에 1.63 으로 튄다.")
    print("  다른 채널은 토·일 모두 높다. 이 대비가 D+1 (체크아웃 선물) 의 근거다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
