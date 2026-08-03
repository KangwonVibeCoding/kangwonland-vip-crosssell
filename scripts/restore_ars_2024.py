"""data/sample/ars.parquet → data/raw/ars_20241201_20241231.csv 복원.

왜 필요한가:
  2026년 ARS 전처리본(`data/raw/ars_merged.csv`)에는 2024-12 이 없고, 컬럼도
  3개(winners/tickets/buy_rate)로 축소돼 있다. 그런데 판매 데이터는 2023~2024
  이므로, 2024-12 ARS 가 없으면 **ARS 와 판매가 겹치는 날이 0일**이 되어
  구간 A(실측 조인)·상관 r=0.80·특산품 D+1 래그가 전부 성립하지 않는다.

  다행히 배포용 축약본 `data/sample/ars.parquet` 에 2024-12 31일치가
  `recv_total`(내국인 총 접수자)까지 온전히 남아 있다. 이것을 원본 CSV 형태로
  되돌려 놓으면 로더의 `ars_*.csv` glob 이 자동으로 함께 concat 한다.

  ⚠ 이 데이터가 내장 폴백(`fallback._ARS_ROWS`)의 모사값이 아니라 실측인지
  검증한 뒤에만 기록한다 — 모사값을 실데이터로 되살리면 회귀 기준값 전체가
  허구 위에 서게 된다.

사용법:
    python scripts/restore_ars_2024.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                     # noqa: E402

from config import settings as S                        # noqa: E402
from src.data import fallback                           # noqa: E402

TARGET = "ars_20241201_20241231.csv"
START, END = pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")

# 표준 컬럼 → 원본 한글 컬럼. 원본과 같은 모양으로 되돌려야 COLUMN_MAP 의
# 기존 경로를 그대로 타고, ingest.py 의 헤더 시그니처 판정과도 일관된다.
TO_KOREAN = {
    "recv_total": "내국인 총 접수자",
    "recv_ars": "ARS 접수자",
    "recv_mobile": "모바일 접수자",
    "winners": "총 당첨자",
    "win_ars": "ARS 당첨자",
    "win_mobile": "모바일 당첨자",
    "tickets": "당첨자 입장권 구매 건 수",
    "buy_rate": "당첨자 입장권 구매율",
}


def main() -> int:
    src = S.SAMPLE_DIR / "ars.parquet"
    if not src.exists():
        print(f"✗ {src} 가 없습니다. 복원할 원본이 없습니다.")
        return 1

    ars = pd.read_parquet(src)
    dec = ars.loc[ars["date"].between(START, END)].sort_values("date")
    if len(dec) != 31:
        print(f"✗ 2024-12 이 31일이 아닙니다 ({len(dec)}일). 중단합니다.")
        return 1

    missing = [c for c in TO_KOREAN if c not in dec.columns]
    if missing:
        print(f"✗ 컬럼 누락 {missing} — 복원해도 구간 A 분석이 불완전합니다.")
        return 1

    # ── 실측 검증: 내장 폴백의 모사값과 같으면 복원하지 않는다 ──────────
    fb = fallback.ars_df()
    if len(fb) == len(dec):
        identical = (
            dec["recv_total"].to_numpy() == fb["내국인 총 접수자"].to_numpy()
        ).all()
        if identical:
            print("✗ 이 데이터는 내장 폴백(fallback._ARS_ROWS)과 동일합니다.")
            print("  모사 데이터를 실데이터로 되살릴 수 없습니다. 중단합니다.")
            return 1

    out = pd.DataFrame({"기준일자": dec["date"].dt.strftime("%Y-%m-%d")})
    for std, kr in TO_KOREAN.items():
        col = dec[std]
        # buy_rate 는 0~1 로 정규화돼 있다. 원본은 % 표기(30~53)이므로 되돌린다.
        out[kr] = (col * 100).round(2) if std == "buy_rate" else col.astype("int64")

    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = S.RAW_DIR / TARGET
    out.to_csv(dest, index=False, encoding="utf-8")

    print(f"✔ 복원: data/raw/{TARGET}  ({len(out)}행)")
    print(f"  기간  {out['기준일자'].iloc[0]} ~ {out['기준일자'].iloc[-1]}")
    print(f"  총접수자  {out['내국인 총 접수자'].min():,} ~ "
          f"{out['내국인 총 접수자'].max():,}")
    print()
    print("이제 로더가 ars_*.csv 를 glob 으로 합쳐 구간 A(2024-12)가 복구됩니다.")
    print("⚠ parquet 캐시를 지우고 다시 로드하세요:")
    print("   Remove-Item data\\processed\\*.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
