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


def fetch_json(endpoint: str, params: dict | None = None) -> list[dict] | None:
    """Open API 호출. 실패 시 디스크 캐시 → None 순으로 폴백한다."""
    params = dict(params or {})
    key = api_key()
    if not key:
        cached = _read_cache(endpoint, params)
        if cached is None:
            _record_error(f"{endpoint}: 인증키가 설정되지 않았습니다 (모킹으로 대체)")
        return cached

    try:
        resp = requests.get(
            f"{S.API_BASE}/{endpoint}",
            params={**params, "serviceKey": key, "returnType": "JSON", "type": "json"},
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


def fetch_paged(endpoint: str, max_pages: int = 5) -> list[dict] | None:
    """페이지네이션 순회. 첫 페이지가 실패하면 즉시 폴백으로 넘긴다."""
    first = fetch_json(endpoint, {"page": 1, "perPage": S.API_PAGE_SIZE,
                                 "pageNo": 1, "numOfRows": S.API_PAGE_SIZE})
    if not first:
        return None
    rows = list(first)
    for page in range(2, max_pages + 1):
        if len(first) < S.API_PAGE_SIZE:
            break
        nxt = fetch_json(endpoint, {"page": page, "perPage": S.API_PAGE_SIZE,
                                    "pageNo": page, "numOfRows": S.API_PAGE_SIZE})
        if not nxt:
            break
        rows.extend(nxt)
        if len(nxt) < S.API_PAGE_SIZE:
            break
    return rows
