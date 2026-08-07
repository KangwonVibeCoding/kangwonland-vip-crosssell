"""data.go.kr Open API 클라이언트 — 실패해도 절대 앱을 죽이지 않는다.

흡수하는 실패 유형:
  키 부재 / 인증 오류 / 타임아웃 / 네트워크 단절 / HTTP 4xx·5xx /
  JSON 파싱 실패 / 응답 스키마 변경

성공 응답은 디스크에 캐시해두고, 다음 호출이 실패하면 캐시를 대신 쓴다.
그마저 없으면 None 을 반환하고 호출자가 임베드 폴백으로 넘어간다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import requests

from config import settings as S

# 오류를 세션에 누적한다. streamlit 이 없는 환경(pytest)에서도 동작해야 하므로
# import 를 함수 안에서 시도하고 실패하면 모듈 레벨 리스트에 담는다.
_LOCAL_ERRORS: list[str] = []


def _record_error(msg: str) -> None:
    try:
        import streamlit as st

        st.session_state.setdefault("api_errors", []).append(msg)
    except Exception:  # noqa: BLE001  streamlit 컨텍스트 밖
        _LOCAL_ERRORS.append(msg)


def get_errors() -> list[str]:
    try:
        import streamlit as st

        return list(st.session_state.get("api_errors", []))
    except Exception:  # noqa: BLE001
        return list(_LOCAL_ERRORS)


def api_key() -> str | None:
    """secrets → 환경변수 순으로 조회. 없으면 None (예외를 던지지 않는다).

    로컬은 .streamlit/secrets.toml, Cloud 는 앱 Settings → Secrets 를 쓴다.
    둘 다 같은 코드 경로로 동작한다.
    """
    try:
        import streamlit as st

        key = st.secrets.get("DATA_GO_KR_KEY")
        if key and "PUT_YOUR" not in str(key):
            return str(key)
    except Exception:  # noqa: BLE001  secrets 파일 없음 / streamlit 밖
        pass
    import os

    key = os.environ.get("DATA_GO_KR_KEY")
    return key if key and "PUT_YOUR" not in key else None


def _cache_path(endpoint: str, params: dict) -> Path:
    stamp = hashlib.md5(
        json.dumps([endpoint, sorted(params.items())], ensure_ascii=False).encode()
    ).hexdigest()[:12]
    safe = endpoint.replace("/", "_").replace(":", "_")
    return S.API_CACHE_DIR / f"{safe}_{stamp}.json"


def _write_cache(endpoint: str, params: dict, rows: list[dict]) -> None:
    try:
        S.API_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(endpoint, params).write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001  캐시 실패는 무시한다 (읽기 전용 FS 등)
        pass


def _read_cache(endpoint: str, params: dict) -> list[dict] | None:
    try:
        p = _cache_path(endpoint, params)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) and data else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _extract_rows(payload: Any) -> list[dict]:
    """응답 본문에서 레코드 리스트를 찾아낸다.

    공공데이터포털은 포털/엔드포인트에 따라 형태가 다르다:
      odcloud   : {"data": [...]}
      apis(구)  : {"response": {"body": {"items": [...]}}}
                  또는 items 가 {"item": [...]} 로 한 겹 더 감싸인 경우
    스키마가 바뀌어도 여러 경로를 시도해 살아남는다.
    """
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        raise ValueError(f"예상치 못한 응답 타입: {type(payload).__name__}")

    # 포털 오류 봉투를 먼저 걸러낸다. 그냥 두면 "레코드 목록을 못 찾음" 으로
    # 뭉뚱그려져 원인(키 미등록/미승인/제공기관 서버 오류)이 사라진다.
    #   04 HTTP_ERROR                        원본 서버가 응답하지 않음
    #   20 SERVICE_ACCESS_DENIED_ERROR       활용신청 미승인
    #   30 SERVICE_KEY_IS_NOT_REGISTERED     키 미등록 (Encoding 키를 쓴 경우 포함)
    header = payload.get("OpenAPI_ServiceResponse", {}).get("cmmMsgHeader")
    if isinstance(header, dict):
        raise ValueError(
            f"{header.get('errMsg', '알 수 없는 오류')} "
            f"(코드 {header.get('returnReasonCode', '?')}) "
            f"— {header.get('returnAuthMsg', '')}".strip()
        )

    if isinstance(payload.get("data"), list):
        return payload["data"]

    body = payload.get("response", {}).get("body", {})
    items = body.get("items")
    if isinstance(items, list):
        return items
    if isinstance(items, dict) and isinstance(items.get("item"), list):
        return items["item"]

    for value in payload.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value
    raise ValueError("응답에서 레코드 목록을 찾지 못했습니다")


def page_params(endpoint: str, page: int, size: int) -> dict:
    """포털 계열에 맞는 페이지 파라미터. 섞어 보내면 HTTP_ERROR 가 난다(실측).

      apis.data.go.kr → pageNo / numOfRows + type=json
      api.odcloud.kr  → page   / perPage
    """
    if "odcloud.kr" in endpoint:
        return {"page": page, "perPage": size}
    return {"pageNo": page, "numOfRows": size, "type": "json"}


def fetch_json(endpoint: str, params: dict | None = None) -> list[dict] | None:
    """Open API 호출. 실패 시 디스크 캐시 → None 순으로 폴백한다.

    `endpoint` 는 전체 URL 이다 (settings.EP_*). 값이 비어 있으면 호출하지 않는다 —
    아직 확인되지 않은 엔드포인트를 부르면 무의미한 오류만 쌓인다.
    """
    if not endpoint:
        return None
    params = dict(params or {})
    key = api_key()
    if not key:
        cached = _read_cache(endpoint, params)
        if cached is None:
            _record_error(f"{endpoint}: 인증키가 설정되지 않았습니다 (모킹으로 대체)")
        return cached

    try:
        resp = requests.get(
            endpoint,
            params={**params, "serviceKey": key},
            timeout=S.API_TIMEOUT,
        )
        resp.raise_for_status()
        rows = _extract_rows(resp.json())
        if not rows:
            raise ValueError("응답이 비어 있습니다")
        _write_cache(endpoint, params, rows)
        return rows
    except Exception as exc:  # noqa: BLE001  모든 실패를 흡수한다
        _record_error(f"{endpoint}: {type(exc).__name__} — {exc}")
        return _read_cache(endpoint, params)


def fetch_paged(endpoint: str, max_pages: int = 5,
                page_size: int | None = None,
                extra: dict | None = None) -> list[dict] | None:
    """페이지네이션 순회. 첫 페이지가 실패하면 즉시 폴백으로 넘긴다.

    마지막 페이지 판정은 **직전 페이지의 행 수**로 한다. 예전 구현은 첫 페이지
    길이만 보고 있어서, 2페이지가 짧아도 남은 페이지를 계속 두드렸다.

    `extra` 는 엔드포인트가 요구하는 추가 파라미터다 (성별연령의 dateFrom/dateTo
    처럼 **필수인데 페이지와 무관한** 것들). 빠뜨리면 포털이 HTTP_ERROR(04) 를
    돌려주는데, 그 코드는 제공기관 서버 다운과 구분되지 않는다 — 실제로 그렇게
    오진한 적이 있다(S.DEMO_PARAMS 주석 참고).
    """
    size = page_size or S.API_PAGE_SIZE
    base = dict(extra or {})
    first = fetch_json(endpoint, {**base, **page_params(endpoint, 1, size)})
    if not first:
        return None
    rows = list(first)
    prev = first
    for page in range(2, max_pages + 1):
        if len(prev) < size:
            break
        nxt = fetch_json(endpoint, {**base, **page_params(endpoint, page, size)})
        if not nxt:
            break
        rows.extend(nxt)
        prev = nxt
    return rows
