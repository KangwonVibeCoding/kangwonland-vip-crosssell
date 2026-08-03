"""data/raw → data/sample : 배포용 축약본 생성.

왜 필요한가:
  `.gitignore` 가 `data/raw/` 를 제외하므로 그대로 push 하면 Streamlit Cloud 배포판에
  원본이 올라가지 않는다. 그러면 심사자가 보는 화면이 내장 데모 데이터로 뜨고,
  가설 검증 배너의 r=0.80 이 실측값으로 표시되지 않는다.

무엇을 담는가:
  **연간 전체를 정규화된 parquet 으로** 담는다. 기간을 자르지 않는 이유는 월 인덱스
  (룸서비스 1월 1.23 · 특산품 9월 1.48)가 계절성 분석의 핵심 근거이기 때문이다.
  CSV 12.4MB 를 parquet 으로 저장하면 문자열 사전 인코딩 덕에 크게 줄어든다.

무엇을 담지 않는가:
  실데이터로 확보되지 않은 데이터셋(현재 성별·연령, 가맹점 — API 키 미설정)은
  내보내지 않는다. 내장 폴백 데이터를 'SAMPLE' 로 커밋하면 모킹을 실데이터로
  위장하는 셈이 된다.

사용법:
    python scripts/make_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                     # noqa: E402

from config import settings as S                        # noqa: E402
from src.data import loaders                            # noqa: E402

# (키, 로더, 출력 파일명)
TARGETS = (
    ("ars", loaders._load_ars_impl, "ars"),
    ("sales", loaders._load_sales_impl, "sales"),
    ("golf", loaders._load_golf_impl, "golf"),
    ("demo", loaders._load_demographics_impl, "demographics"),
    ("merchants", loaders._load_merchants_impl, "merchants"),
)


# 구간 A(실측 조인)의 근거 기간. 축약본에 이 기간이 없으면 배포판에서 상관
# r=0.80 배너와 특산품 D+1 래그가 통째로 사라진다.
PERIOD_A_START = pd.Timestamp(S.PERIODS[S.PERIOD_A]["start"])
PERIOD_A_END = pd.Timestamp(S.PERIODS[S.PERIOD_A]["end"])


def guard_ars(df: pd.DataFrame) -> str | None:
    """ARS 축약본이 구간 A 를 담고 있는지 검사. 문제가 있으면 사유를 반환한다.

    ⚠ 이 가드가 존재하는 이유: `data/sample/ars.parquet` 은 2024-12 ARS 의
    **유일한 사본**이다(원본 CSV 는 gitignore 되는 data/raw 에만 있고,
    2026년 전처리본에는 2024-12 이 없다). raw 에서 2024-12 파일이 빠진 채로
    이 스크립트를 돌리면 축약본을 2026년분으로 덮어써 복원 원천이 사라지고,
    scripts/restore_ars_2024.py 도 더는 쓸 수 없게 된다.
    """
    covered = df.loc[df["date"].between(PERIOD_A_START, PERIOD_A_END)]
    if len(covered) >= 28:
        return None
    return (
        f"구간 A({PERIOD_A_START:%Y-%m} ) 가 {len(covered)}일뿐입니다 — "
        "덮어쓰면 기존 축약본의 2024-12 사본이 사라집니다. "
        "먼저 python scripts/restore_ars_2024.py 를 실행하세요"
    )


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def main() -> int:
    S.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("배포용 축약본 생성 — data/raw → data/sample")
    print("=" * 78)

    written: list[tuple[str, int, int]] = []
    skipped: list[tuple[str, str]] = []

    for key, loader, name in TARGETS:
        try:
            df, source = loader()
        except Exception as exc:                          # noqa: BLE001
            skipped.append((name, f"로드 실패 — {exc}"))
            continue

        if source not in S.REAL_SOURCES:
            label = S.SOURCE_LABEL.get(source, (source, ""))[0]
            skipped.append((name, f"실데이터가 아님 ({label}) — 커밋하지 않습니다"))
            continue
        if df.empty:
            skipped.append((name, "비어 있음"))
            continue
        if key == "ars":
            problem = guard_ars(df)
            if problem:
                skipped.append((name, problem))
                continue

        dest = S.SAMPLE_DIR / f"{name}.parquet"
        df.to_parquet(dest, index=False, compression="snappy")
        size = dest.stat().st_size
        written.append((name, len(df), size))
        span = ""
        if "date" in df.columns and not df["date"].isna().all():
            span = f"  {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}"
        print(f"  ✔ {name:<14} {len(df):>7,}행  {human(size):>10}{span}")

    for name, reason in skipped:
        print(f"  – {name:<14} 건너뜀: {reason}")

    total = sum(size for _, _, size in written)
    print("-" * 78)
    print(f"합계 {len(written)}개 파일 · {human(total)}")

    if written:
        print()
        print("이 파일들은 .gitignore 의 예외 규칙(!data/sample/**)에 따라 커밋됩니다.")
        print("Cloud 배포판이 실데이터로 뜨려면 반드시 함께 push 해야 합니다.")
    if any(name in ("demographics", "merchants") for name, _ in skipped):
        print()
        print("※ 성별·연령 / 가맹점은 Open API 데이터입니다. .streamlit/secrets.toml 에")
        print("  DATA_GO_KR_KEY 를 넣고 다시 실행하면 축약본에 포함됩니다.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
