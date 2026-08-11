"""data/incoming → data/raw 정규화.

원본 CSV 는 한글·공백·괄호가 섞인 파일명 + CP949 인코딩이고, 파일명 규칙조차
일관되지 않다('(주)강원랜드_' 접두가 있는 것과 없는 것이 섞여 있다).
git / Linux / Streamlit Cloud 에서 문제가 되므로 한 번에 정리한다.

  - 판정은 **파일 내용의 헤더 시그니처**로 한다 (파일명에 의존하지 않는다)
  - CP949 → UTF-8 재저장
  - ASCII 소문자 스네이크케이스 파일명 + 실제 데이터 기간을 파일명에 부착
  - 검증 리포트를 콘솔과 data/raw/_ingest_report.txt 에 출력

사용법:
    python scripts/ingest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as S                       # noqa: E402
from src.data import schema                            # noqa: E402

# ── 헤더 시그니처 → 데이터셋 종류 ──────────────────────────────────────
# 파일명이 신뢰할 수 없으므로 내용으로 판정한다. 판정 순서가 중요하다:
# 더 구체적인(고유 컬럼을 가진) 규칙을 먼저 둔다.
SIG_ARS = "당첨자입장권구매율"
SIG_ARS_EN = "purchaserate"              # 전처리본(ars_merged.csv)의 영문 헤더
SIG_ARS_SHORT = "입장권구매율"            # 전처리 통합본(2023~2026)의 축약 헤더
SIG_GOLF = "이용인원"                     # 골프장만 보유 (판매는 '판매수량')
SIG_VENUE_ID = "영업장아이디id"          # 카지노 식음만 보유
SIG_ITEM_ID = "상품아이디id"
SIG_DISCOUNT = "할인구분여부"

KIND_ARS = "ars"
KIND_GOLF = "golf_visitors"
KIND_CASINO = "casino_fnb"
KIND_ROOM = "roomservice"
KIND_LOCAL = "local_goods"

ROOM_VENUE_HINT = "In Room Dining"
LOCAL_VENUE_HINT = "특산품"


def classify(df: pd.DataFrame) -> str | None:
    """헤더(+필요시 영업장명 값)로 데이터셋 종류를 판정한다."""
    keys = {schema.normalize_key(c) for c in df.columns}

    if SIG_ARS in keys or SIG_ARS_EN in keys or SIG_ARS_SHORT in keys:
        return KIND_ARS
    if SIG_GOLF in keys:
        return KIND_GOLF
    if SIG_VENUE_ID in keys:
        return KIND_CASINO

    # 룸서비스와 특산품은 헤더가 거의 같다 (영업일자/영업장명/상품명/…).
    # 구분 신호 두 개를 함께 쓴다:
    #   - 룸서비스에는 할인구분여부가 있고 상품아이디(ID)가 없다
    #   - 특산품에는 상품아이디(ID)가 있고 할인구분여부가 없다
    # 그래도 애매하면 영업장명 실제 값으로 확정한다.
    has_discount = SIG_DISCOUNT in keys
    has_item_id = SIG_ITEM_ID in keys
    if has_discount and not has_item_id:
        return KIND_ROOM
    if has_item_id and not has_discount:
        return KIND_LOCAL

    venue_col = next(
        (c for c in df.columns if schema.normalize_key(c) == "영업장명"), None
    )
    if venue_col is not None:
        venues = df[venue_col].astype("string").dropna().unique().tolist()
        joined = " ".join(str(v) for v in venues[:20])
        if ROOM_VENUE_HINT.lower() in joined.lower():
            return KIND_ROOM
        if LOCAL_VENUE_HINT in joined:
            return KIND_LOCAL
    return None


def date_span(df: pd.DataFrame) -> tuple[str, str] | None:
    """데이터의 실제 기간을 YYYYMMDD 로 반환 (파일명 부착용)."""
    col = next(
        (c for c in df.columns
         if schema.normalize_key(c) in ("기준일자", "영업일자")),
        None,
    )
    if col is None:
        return None
    d = pd.to_datetime(df[col], errors="coerce").dropna()
    if d.empty:
        return None
    return d.min().strftime("%Y%m%d"), d.max().strftime("%Y%m%d")


def target_name(kind: str, span: tuple[str, str] | None) -> str:
    """출력 파일명. ARS 는 여러 파일이 공존하므로 기간을 반드시 붙인다.

    ⚠ 다른 종류는 `{kind}_{시작연도}.csv` 라서 같은 해 파일이 두 개 들어오면
    뒤엣것이 앞엣것을 덮어쓴다. 여러 해가 하나로 합쳐진 `*_merged.csv` 는
    이 함수를 거치지 않고 이미 data/raw 에 있는 파일들이다.
    """
    if kind == KIND_ARS:
        if span is None:
            return "ars_unknown.csv"
        return f"ars_{span[0]}_{span[1]}.csv"
    if span and span[0][:4] != span[1][:4]:
        # 여러 해에 걸친 파일은 연도 하나로 이름 지으면 덮어쓰기 사고가 난다
        return f"{kind}_{span[0]}_{span[1]}.csv"
    year = span[0][:4] if span else "unknown"
    return f"{kind}_{year}.csv"


def main() -> int:
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(S.INCOMING_DIR.glob("*.csv"))
    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    out("=" * 78)
    out("강원랜드 데이터 ingest — data/incoming → data/raw")
    out("=" * 78)

    if not files:
        out(f"⚠ {S.INCOMING_DIR} 에 CSV 가 없습니다.")
        out("  공공데이터포털에서 내려받은 CSV 를 이 폴더에 넣고 다시 실행하세요.")
        S.INGEST_REPORT.write_text("\n".join(lines), encoding="utf-8")
        return 1

    written: list[str] = []
    problems: list[str] = []

    for src in files:
        out("")
        out(f"▶ {src.name}  ({src.stat().st_size:,} bytes)")
        enc = schema.detect_encoding(src)
        out(f"   인코딩: {enc or '판별 실패'}")
        try:
            df = schema.read_csv_kr(src, dtype="string")
        except Exception as exc:                          # noqa: BLE001
            problems.append(f"{src.name}: 읽기 실패 — {exc}")
            out(f"   ✗ 읽기 실패: {exc}")
            continue

        kind = classify(df)
        if kind is None:
            problems.append(f"{src.name}: 데이터셋 종류 판정 실패")
            out(f"   ✗ 종류 판정 실패. 헤더: {list(df.columns)}")
            continue

        span = date_span(df)
        name = target_name(kind, span)
        dest = S.RAW_DIR / name

        out(f"   종류: {kind}")
        out(f"   행 수: {len(df):,}  컬럼 수: {len(df.columns)}")
        if span:
            out(f"   기간: {span[0]} ~ {span[1]}")

        # 컬럼 매핑 점검 — 미매칭이 있으면 경고만 하고 계속 진행한다
        _, unmatched = schema.rename_columns(df)
        if unmatched:
            out(f"   ⚠ 미매칭 컬럼 {len(unmatched)}개: {unmatched}")
            problems.append(f"{src.name}: 미매칭 컬럼 {unmatched}")

        # 결측 점검
        blanks = {
            str(c): int((df[c].isna() | df[c].astype("string").str.strip().eq("")).sum())
            for c in df.columns
        }
        nonzero = {k: v for k, v in blanks.items() if v}
        if nonzero:
            out(f"   결측/공백: {nonzero}")

        df.to_csv(dest, index=False, encoding="utf-8")
        written.append(name)
        out(f"   ✔ 저장: data/raw/{name}")

    out("")
    out("-" * 78)
    out(f"완료: {len(written)}개 파일 → data/raw/")
    for n in sorted(written):
        out(f"  • {n}")
    if problems:
        out("")
        out(f"확인 필요 {len(problems)}건:")
        for p in problems:
            out(f"  ! {p}")

    # ARS 파일이 여러 개면 구간 A 가 자동 확장된다는 점을 알려준다
    ars_files = sorted(S.RAW_DIR.glob("ars_*.csv"))
    if ars_files:
        out("")
        out(f"ARS 파일 {len(ars_files)}개 — 로더가 glob 으로 전부 concat 합니다.")
        out("  (다른 월 ARS 파일을 추가하면 정밀 조인 구간이 자동으로 넓어집니다)")

    S.INGEST_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n리포트: {S.INGEST_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
