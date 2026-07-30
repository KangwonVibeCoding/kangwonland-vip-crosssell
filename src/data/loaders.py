"""데이터 로더 — 4계층 폴백. 예외를 밖으로 던지지 않는다.

폴백 체인 (각 단계는 개별 try/except):
  L1 data/processed/*.parquet   → PROCESSED
  L2 data/raw/*.csv (ingest)    → RAW      + parquet 캐시 기록
  L3 data/sample/*.csv          → SAMPLE
  L4 Open API                   → API
  L5 data/mock/*.csv            → MOCK
  L6 fallback.py 임베드 상수     → EMBEDDED  (파일·네트워크 접근 0 → 절대 실패 없음)

모든 로더는 `(DataFrame, source)` 를 반환한다.

streamlit 캐시가 필요한 표면은 `load_*` 이고, 순수 로직은 `_load_*_impl` 이다.
pytest 는 `_impl` 을 직접 호출한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings as S
from src.data import api_client, fallback, schema

# ── 내부 유틸 ─────────────────────────────────────────────────────────


def _read_parquet(name: str, directory: Path | None = None) -> pd.DataFrame | None:
    try:
        p = (directory or S.PROCESSED_DIR) / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            return df if not df.empty else None
    except Exception:  # noqa: BLE001  pyarrow 부재·손상 파일 등
        pass
    return None


def _read_sample(name: str) -> pd.DataFrame | None:
    """배포용 축약본 — 정규화가 끝난 parquet 이므로 그대로 쓴다.

    `scripts/make_sample.py` 가 만든다. CSV 12.4MB 를 그대로 커밋하는 대신 parquet
    으로 압축해 저장소를 가볍게 유지하면서 연간 전체를 담는다(월 인덱스 분석에 필요).
    """
    return _read_parquet(name, S.SAMPLE_DIR)


def _write_parquet(name: str, df: pd.DataFrame) -> None:
    try:
        S.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(S.PROCESSED_DIR / f"{name}.parquet", index=False)
    except Exception:  # noqa: BLE001  캐시 실패는 치명적이지 않다
        pass


def _csv_candidates(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    try:
        for pat in patterns:
            found.extend(sorted(directory.glob(pat)))
    except Exception:  # noqa: BLE001
        return []
    # 중복 제거 (패턴이 겹칠 수 있다)
    seen: set[Path] = set()
    return [p for p in found if not (p in seen or seen.add(p))]


# ── 1. ARS 예약현황 ───────────────────────────────────────────────────
# 특례: 여러 파일을 glob 으로 concat 한다. ARS 파일을 추가로 넣으면 정밀 조인
# 구간(구간 A)이 코드 수정 없이 자동 확장된다.
_ARS_PATTERNS = ("ars_*.csv", "ars.csv")


def _ars_from_csv(directory: Path) -> pd.DataFrame | None:
    """디렉토리의 ARS CSV 를 전부 읽어 날짜 기준으로 합친다."""
    paths = _csv_candidates(directory, _ARS_PATTERNS)
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            frames.append(schema.normalize_ars(schema.read_csv_kr(p, dtype="string")))
        except Exception:  # noqa: BLE001  한 파일 실패는 건너뛴다
            continue
    if not frames:
        return None
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _load_ars_impl() -> tuple[pd.DataFrame, str]:
    cached = _read_parquet("ars")
    if cached is not None:
        return cached, S.SRC_PROCESSED

    df = _ars_from_csv(S.RAW_DIR)
    if df is not None:
        _write_parquet("ars", df)
        return df, S.SRC_RAW

    sample = _read_sample("ars")
    if sample is not None:
        return sample, S.SRC_SAMPLE

    df = _ars_from_csv(S.SAMPLE_DIR)
    if df is not None:
        return df, S.SRC_SAMPLE

    df = _ars_from_csv(S.MOCK_DIR)
    if df is not None:
        return df, S.SRC_MOCK

    return schema.normalize_ars(fallback.ars_df()), S.SRC_EMBEDDED


# ── 2. 판매 3종 통합 (long) ────────────────────────────────────────────
_SALES_PATTERNS = {
    S.CH_CASINO: ("casino_fnb*.csv",),
    S.CH_ROOM: ("roomservice*.csv",),
    S.CH_LOCAL: ("local_goods*.csv",),
}


def _sales_from_csv(directory: Path) -> pd.DataFrame | None:
    """디렉토리에서 채널별 판매 CSV 를 읽어 단일 long 테이블로 합친다."""
    frames: list[pd.DataFrame] = []
    for channel, patterns in _SALES_PATTERNS.items():
        for p in _csv_candidates(directory, patterns):
            try:
                raw = schema.read_csv_kr(p, dtype="string")
                frames.append(schema.normalize_sales(raw, channel))
                break  # 채널당 첫 성공 파일만
            except Exception:  # noqa: BLE001
                continue
    return pd.concat(frames, ignore_index=True) if frames else None


def _load_sales_impl() -> tuple[pd.DataFrame, str]:
    cached = _read_parquet("sales")
    if cached is not None:
        return cached, S.SRC_PROCESSED

    df = _sales_from_csv(S.RAW_DIR)
    if df is not None:
        _write_parquet("sales", df)
        return df, S.SRC_RAW

    sample = _read_sample("sales")
    if sample is not None:
        return sample, S.SRC_SAMPLE

    df = _sales_from_csv(S.SAMPLE_DIR)
    if df is not None:
        return df, S.SRC_SAMPLE

    df = _sales_from_csv(S.MOCK_DIR)
    if df is not None:
        return df, S.SRC_MOCK

    frames = [
        schema.normalize_sales(raw, ch)
        for ch, raw in fallback.sales_by_channel().items()
    ]
    return pd.concat(frames, ignore_index=True), S.SRC_EMBEDDED


# ── 3. 고객성별연령분석현황 (Open API) ─────────────────────────────────
def _load_demographics_impl() -> tuple[pd.DataFrame, str]:
    cached = _read_parquet("demographics")
    if cached is not None:
        return cached, S.SRC_PROCESSED

    for directory, source in ((S.RAW_DIR, S.SRC_RAW), (S.SAMPLE_DIR, S.SRC_SAMPLE)):
        for p in _csv_candidates(directory, ("demographics*.csv",)):
            try:
                df = schema.normalize_demographics(schema.read_csv_kr(p, dtype="string"))
                if not df.empty:
                    return df, source
            except Exception:  # noqa: BLE001
                continue

    sample = _read_sample("demographics")
    if sample is not None:
        return sample, S.SRC_SAMPLE

    rows = api_client.fetch_paged(S.EP_DEMOGRAPHICS)
    if rows:
        try:
            df = schema.normalize_demographics(pd.DataFrame(rows))
            if not df.empty and not schema.validate(df, S.DEMO_REQUIRED, "demographics"):
                _write_parquet("demographics", df)
                return df, S.SRC_API
        except Exception:  # noqa: BLE001  응답 스키마가 달라도 폴백으로 계속
            pass

    for p in _csv_candidates(S.MOCK_DIR, ("demographics*.csv",)):
        try:
            return schema.normalize_demographics(
                schema.read_csv_kr(p, dtype="string")
            ), S.SRC_MOCK
        except Exception:  # noqa: BLE001
            continue

    return schema.normalize_demographics(fallback.demographics_df()), S.SRC_EMBEDDED


# ── 4. 하이원포인트 가맹점 (Open API) ──────────────────────────────────
def _load_merchants_impl() -> tuple[pd.DataFrame, str]:
    cached = _read_parquet("merchants")
    if cached is not None:
        return cached, S.SRC_PROCESSED

    for directory, source in ((S.RAW_DIR, S.SRC_RAW), (S.SAMPLE_DIR, S.SRC_SAMPLE)):
        for p in _csv_candidates(directory, ("merchants*.csv",)):
            try:
                df = schema.normalize_merchants(schema.read_csv_kr(p, dtype="string"))
                if not df.empty:
                    return df, source
            except Exception:  # noqa: BLE001
                continue

    sample = _read_sample("merchants")
    if sample is not None:
        return sample, S.SRC_SAMPLE

    rows = api_client.fetch_paged(S.EP_MERCHANTS)
    if rows:
        try:
            df = schema.normalize_merchants(pd.DataFrame(rows))
            if not df.empty and not schema.validate(df, S.MERCHANT_REQUIRED, "merchants"):
                _write_parquet("merchants", df)
                return df, S.SRC_API
        except Exception:  # noqa: BLE001
            pass

    for p in _csv_candidates(S.MOCK_DIR, ("merchants*.csv",)):
        try:
            return schema.normalize_merchants(
                schema.read_csv_kr(p, dtype="string")
            ), S.SRC_MOCK
        except Exception:  # noqa: BLE001
            continue

    return schema.normalize_merchants(fallback.merchants_df()), S.SRC_EMBEDDED


# ── streamlit 캐시 래퍼 ────────────────────────────────────────────────
# _impl 은 순수 함수이므로 pytest 에서 streamlit 없이 호출할 수 있다.
# 캐시는 첫 호출 시점에 지연 적용한다 — import 시점에 데코레이터를 붙이면
# streamlit 런타임 밖(pytest·스크립트)에서 "No runtime found" 경고가 쏟아진다.
_cache_registry: dict[str, object] = {}


def _cached(fn):
    def wrapper(*args, **kwargs):
        key = fn.__name__
        cached = _cache_registry.get(key)
        if cached is None:
            try:
                import streamlit as st
                from streamlit.runtime import exists as runtime_exists

                cached = (
                    st.cache_data(ttl=S.CACHE_TTL, show_spinner=False)(fn)
                    if runtime_exists()
                    else fn
                )
            except Exception:  # noqa: BLE001  streamlit 부재 / API 변경
                cached = fn
            _cache_registry[key] = cached
        return cached(*args, **kwargs)

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


load_ars = _cached(_load_ars_impl)
load_sales = _cached(_load_sales_impl)
load_demographics = _cached(_load_demographics_impl)
load_merchants = _cached(_load_merchants_impl)


def load_all_impl() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """4개 논리 데이터셋을 한 번에. 어떤 실패도 앱을 죽이지 않는다."""
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for key, fn in (
        ("ars", _load_ars_impl),
        ("sales", _load_sales_impl),
        ("demo", _load_demographics_impl),
        ("merchants", _load_merchants_impl),
    ):
        try:
            frames[key], sources[key] = fn()
        except Exception:  # noqa: BLE001  이론상 도달 불가 — 그래도 방어한다
            emergency = {
                "ars": lambda: schema.normalize_ars(fallback.ars_df()),
                "sales": lambda: pd.concat(
                    [schema.normalize_sales(r, c)
                     for c, r in fallback.sales_by_channel().items()],
                    ignore_index=True,
                ),
                "demo": lambda: schema.normalize_demographics(fallback.demographics_df()),
                "merchants": lambda: schema.normalize_merchants(fallback.merchants_df()),
            }[key]
            frames[key], sources[key] = emergency(), S.SRC_EMBEDDED
    return frames, sources
