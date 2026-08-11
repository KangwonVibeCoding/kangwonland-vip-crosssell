"""Open API → data/raw/*.csv (빌드 타임 1회 실행).

**런타임에 공공데이터 API 를 호출하지 않기 위한 스크립트다.** 배포 환경에서
실시간 호출이 깨지는 사고(인증키 미등록, CORS, Mixed Content, 그리고 가장 흔한
'제공기관 서버가 죽음')를 피하려면, 데이터를 미리 파일로 받아 저장소에 넣고
앱은 그 파일만 읽어야 한다.

받은 CSV 는 `data/raw/` 에 떨어지고, 로더가 L2 단계에서 자동으로 집어간다
(`loaders._load_demographics_impl` 은 API 보다 raw/sample CSV 를 먼저 본다).
배포판까지 반영하려면 `scripts/make_sample.py` 를 이어서 실행한다.

    python scripts/fetch_api_data.py                # 전체
    python scripts/fetch_api_data.py demographics   # 하나만

인증키는 `.streamlit/secrets.toml` 의 `DATA_GO_KR_KEY` 또는 동명 환경변수에서
읽는다. **Decoding 키여야 한다** — requests 가 파라미터를 URL 인코딩하므로
Encoding 키를 넣으면 이중 인코딩으로 SERVICE_KEY_IS_NOT_REGISTERED 가 난다.

이미 받아둔 CSV 가 있는데 이번 호출이 실패하면 **덮어쓰지 않는다.** 성공한
수집물을 실패 응답으로 날리는 것이 이 스크립트의 유일한 파괴적 사고다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as S                       # noqa: E402
from src.data import api_client                        # noqa: E402
from src.data import schema                            # noqa: E402

# 키 → (엔드포인트, 출력 파일 접두, 정규화 함수, 필수 컬럼)
# 출력 파일명은 로더의 glob(`demographics*.csv` / `merchants*.csv`)과 맞춘다.
TARGETS: dict[str, dict] = {
    "demographics": {
        "endpoint": S.EP_DEMOGRAPHICS,
        "prefix": "demographics",
        "label": "고객성별연령분석현황",
        "normalize": schema.normalize_demographics,
        "required": S.DEMO_REQUIRED,
        "params": S.DEMO_PARAMS,          # dateFrom/dateTo 필수
    },
    "merchants": {
        "endpoint": S.EP_MERCHANTS,
        "prefix": "merchants",
        "label": "하이원포인트 가맹점 상세정보",
        "normalize": schema.normalize_merchants,
        "required": S.MERCHANT_REQUIRED,
    },
}

MAX_PAGES = 50          # 1000행 × 50 = 5만행. 두 데이터셋 모두 이보다 훨씬 작다

# 포털 오류 코드별 다음 행동. 코드가 무엇을 뜻하는지 모르면 우리 쪽에서 고칠 수
# 없는 문제에 시간을 쓰게 된다.
ERROR_HINTS: dict[str, tuple[str, ...]] = {
    "04": (
        "→ HTTP_ERROR(04) 는 두 가지를 구분하지 않습니다. **필수 파라미터 누락**을",
        "  먼저 의심하세요 — 제공기관 서버 다운으로 보이지만 아닌 경우가 많습니다.",
        "  성별연령 엔드포인트가 그랬습니다: dateFrom/dateTo 를 빼면 04, 넣으면 200.",
        "  (2026-08-04 에 이를 서버 장애로 오진하고 나흘을 보냈습니다.)",
        "  1) 포털 상세기능 문서의 **요청변수 중 필수 항목**을 전부 보내는지 확인",
        "     → 맞다면 config/settings.py 의 해당 EP 파라미터 상수에 추가하세요.",
        "  2) 그래도 04 라면 그때가 진짜 제공기관 서버 문제입니다 — 시간을 두고",
        "     다시 실행하세요. 게이트웨이 인증은 통과한 상태이므로 키 문제는 아닙니다",
        "     (잘못된 키는 403 SERVICE_KEY_IS_NOT_REGISTERED 로 옵니다).",
    ),
    "20": (
        "→ SERVICE_ACCESS_DENIED(20): 포털에서 해당 API 활용신청이 승인되지",
        "  않았습니다. 마이페이지 → 데이터활용 → 승인 상태를 확인하세요.",
    ),
    "30": (
        "→ SERVICE_KEY_IS_NOT_REGISTERED(30): 키 문제입니다. **Decoding 키**를",
        "  넣었는지 확인하세요 — Encoding 키를 넣으면 이중 인코딩으로 이 오류가 납니다.",
    ),
}


def existing(prefix: str) -> Path | None:
    """이미 받아둔 수집물. 실패 시 덮어쓰기를 막는 근거가 된다."""
    found = sorted(S.RAW_DIR.glob(f"{prefix}*.csv"))
    return found[-1] if found else None


def fetch_one(name: str, spec: dict, out) -> bool:
    out("")
    out(f"▶ {name} — {spec['label']}")

    endpoint = spec["endpoint"]
    if not endpoint:
        out("   ✗ 엔드포인트가 설정되지 않았습니다 (config/settings.py).")
        out("     포털 → 마이페이지 → 데이터활용 → Open API → 개발계정 → 상세기능 의")
        out("     '요청주소' 전체를 그대로 넣으세요.")
        return False

    out(f"   {endpoint}")
    prior = existing(spec["prefix"])
    if prior:
        out(f"   기존 수집물: {prior.name} (호출 실패 시 유지됩니다)")

    rows = api_client.fetch_paged(endpoint, max_pages=MAX_PAGES,
                                  extra=spec.get("params"))
    if not rows:
        errors = api_client.get_errors()[-3:]
        for msg in errors:
            out(f"   ! {msg}")
        out("   ✗ 응답을 받지 못했습니다.")
        # 오류 코드마다 다음 행동이 완전히 다르다. 특히 04 는 서버 장애와 필수
        # 파라미터 누락을 구분하지 않아서, 서버 탓으로 넘겨짚으면 고칠 수 있는
        # 것을 못 고친 채 며칠을 보낸다 — 실제로 그랬다(2026-08-04~08).
        for code, hint in ERROR_HINTS.items():
            if any(f"코드 {code}" in msg for msg in errors):
                for line in hint:
                    out(f"     {line}")
                break
        return False

    raw = pd.DataFrame(rows)
    out(f"   ✔ {len(raw):,}행 수신 · 필드 {len(raw.columns)}개")
    out(f"     {list(raw.columns)}")

    # 정규화가 실패해도 원본은 남긴다 — 컬럼 매핑을 고치는 근거가 되기 때문이다.
    try:
        norm = spec["normalize"](raw)
        missing = schema.validate(norm, spec["required"], name)
    except Exception as exc:                              # noqa: BLE001
        norm, missing = None, [f"{name}: 정규화 실패 — {exc}"]

    if missing:
        for m in missing:
            out(f"   ⚠ {m}")
        out("     → 원본 그대로 저장합니다. config/settings.py 의 COLUMN_MAP 을")
        out("       실제 필드명에 맞춰 보완한 뒤 다시 실행하세요.")
    elif norm is not None:
        out(f"   ✔ 정규화 통과 — 표준 컬럼 {list(norm.columns)}")

    dest = S.RAW_DIR / f"{spec['prefix']}_api.csv"
    raw.to_csv(dest, index=False, encoding="utf-8")
    out(f"   ✔ 저장: data/raw/{dest.name}  ({dest.stat().st_size:,} bytes)")
    return True


def main() -> int:
    S.RAW_DIR.mkdir(parents=True, exist_ok=True)
    wanted = sys.argv[1:] or list(TARGETS)
    unknown = [w for w in wanted if w not in TARGETS]
    if unknown:
        print(f"알 수 없는 대상: {unknown}. 사용 가능: {list(TARGETS)}")
        return 1

    lines: list[str] = []

    def out(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    out("=" * 78)
    out("공공데이터 Open API → data/raw/*.csv  (런타임 호출을 없애기 위한 사전 수집)")
    out("=" * 78)

    if not api_client.api_key():
        out("✗ 인증키가 없습니다.")
        out('  .streamlit/secrets.toml 에  DATA_GO_KR_KEY = "디코딩키"  를 넣으세요.')
        return 1

    ok = [name for name in wanted if fetch_one(name, TARGETS[name], out)]

    out("")
    out("-" * 78)
    out(f"성공 {len(ok)} / 요청 {len(wanted)}")
    if ok:
        out("")
        out("다음 단계:")
        out("  1) Remove-Item data\\processed\\*.parquet   # 옛 캐시 제거")
        out("  2) python scripts/make_sample.py           # 배포용 축약본 갱신")
        out("  3) pytest tests/ -q")
    failed = [n for n in wanted if n not in ok]
    if failed:
        out("")
        out(f"실패: {failed}")
        out("  기존 수집물이 있으면 그대로 유지됐습니다 (덮어쓰지 않음).")

    (S.RAW_DIR / "_api_fetch_report.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0 if len(ok) == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
