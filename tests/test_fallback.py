"""L6 임베드 폴백 — 데이터가 하나도 없을 때의 최후 계층.

이 계층은 파일도 네트워크도 안 쓰므로 **항상 실행된다**(skip 없음). 그런데도
여태 테스트가 하나도 없었고, 그 사이 판매 데이터가 2년으로 확장됐을 때 상수 갱신을
빠뜨려 한동안 1년치 구값(특산품 9월 1.48 / 일요일 1.68)을 모사하고 있었다.
139개가 전부 통과하는 동안 아무도 못 잡았다.

여기서 고정하는 것은 두 가지다.

1. **폴백이 실패하지 않는다** — 예외 없음, 빈 프레임 없음, 호출마다 동일.
2. **폴백이 의미 있는 모양으로 보인다** — 특산품 D+1, 일요일 급등 같은 발표 근거가
   폴백 화면에서도 재현돼야 데모가 성립한다. 난수를 안 쓰는 이유가 이것이다.

기대값은 `test_stats.py` 의 실측 기준값을 **그대로 import** 한다. 회귀 기준값이
바뀌면 이 파일이 먼저 깨져서 fallback.py 갱신을 강제한다 (CLAUDE.md <document_sync>).
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings as S
from src.analysis import lag, stats
from src.data import fallback, loaders, schema

from tests.test_stats import (
    EXPECTED_BEST_LAG,
    EXPECTED_DOW_LOCAL,
    EXPECTED_MONTH,
    TOL_IDX,
)


@pytest.fixture
def embedded(tmp_path, monkeypatch):
    """모든 데이터 디렉토리를 없애 L6 까지 내려보낸다 — 실제 파일은 건드리지 않는다."""
    for name in ("PROCESSED_DIR", "RAW_DIR", "SAMPLE_DIR", "MOCK_DIR"):
        monkeypatch.setattr(S, name, tmp_path / f"no_{name.lower()}")
    ars, ars_src = loaders._load_ars_impl()
    sales, sales_src = loaders._load_sales_impl()
    return ars, sales, {"ars": ars_src, "sales": sales_src}


@pytest.fixture
def embedded_sales() -> pd.DataFrame:
    """로더를 거치지 않고 폴백 판매 3종을 직접 표준화한 long 테이블."""
    frames = [
        schema.normalize_sales(raw, ch)
        for ch, raw in fallback.sales_by_channel().items()
    ]
    return pd.concat(frames, ignore_index=True)


# ── 1. 절대 실패하지 않는다 ────────────────────────────────────────────
def test_every_frame_is_non_empty():
    """폴백 함수는 예외를 던지지 않고 빈 프레임도 내지 않는다."""
    frames = {
        "ars": fallback.ars_df(),
        "golf": fallback.golf_df(),
        "demographics": fallback.demographics_df(),
        "merchants": fallback.merchants_df(),
        **fallback.sales_by_channel(),
    }
    for name, df in frames.items():
        assert not df.empty, f"{name} 폴백이 비어 있습니다"


def test_output_is_deterministic():
    """호출마다 같은 결과 — 상품 ID 가 md5 기반인 이유다.

    내장 hash() 를 쓰면 PYTHONHASHSEED 무작위화로 실행마다 ID 가 달라진다.
    """
    pd.testing.assert_frame_equal(fallback.ars_df(), fallback.ars_df())
    for ch, df in fallback.sales_by_channel().items():
        pd.testing.assert_frame_equal(df, fallback.sales_df(ch))


def test_loaders_fall_through_to_embedded(embedded):
    """데이터 디렉토리가 전부 없으면 출처가 EMBEDDED 여야 한다."""
    ars, sales, sources = embedded
    assert sources == {"ars": S.SRC_EMBEDDED, "sales": S.SRC_EMBEDDED}
    assert len(ars) == 31, "구간 A 와 같은 31일을 모사한다"
    assert set(sales["channel"].unique()) == set(S.CHANNELS)


# ── 2. 상수가 실측을 따라간다 (이번 누락의 재발 방지) ──────────────────
def test_constants_track_measured_baselines():
    """폴백 상수 == test_stats.py 의 실측 회귀 기준값.

    폴백은 '그럴듯한 숫자'가 아니라 **실측 모사**다. 회귀 기준값이 바뀌었는데
    여기가 안 바뀌면 폴백 화면이 조용히 낡은 사실을 말하게 된다.
    """
    dow_local = fallback._DOW_INDEX[S.CH_LOCAL]
    pos = {name: i for i, name in enumerate(S.DOW_NAMES)}
    for name, expected in EXPECTED_DOW_LOCAL.items():
        assert dow_local[pos[name]] == pytest.approx(expected, abs=TOL_IDX), (
            f"특산품 {name}요일 지수가 실측({expected})과 다릅니다"
        )

    for channel, (month, expected) in EXPECTED_MONTH.items():
        months = fallback._MONTH_INDEX[channel]
        assert months[month - 1] == pytest.approx(expected, abs=TOL_IDX), (
            f"{channel} {month}월 지수가 실측({expected})과 다릅니다"
        )
        assert months.index(max(months)) == month - 1, (
            f"{channel} 폴백 피크 월이 실측 피크({month}월)와 다릅니다"
        )


# ── 3. 발표 근거가 폴백에서도 재현된다 ─────────────────────────────────
def test_lag_finding_survives_in_fallback(embedded_sales):
    """채널별 최적 시차 — 특산품만 D+1. 이 프로젝트의 핵심 발견이다."""
    ars = schema.normalize_ars(fallback.ars_df())
    best = lag.best_lags(lag.lag_matrix(ars, embedded_sales, max_lag=3))
    got = dict(zip(best["channel"], best["best_lag"]))
    assert got == EXPECTED_BEST_LAG


def test_local_goods_spikes_on_sunday(embedded_sales):
    """특산품만 토요일에 죽었다가 일요일에 튄다 — D+1 의 진짜 근거.

    test_stats.py 가 실데이터에 대해 거는 것과 같은 불변식이다. 폴백에서 이게
    깨지면 데이터 없는 환경의 탭3 이 반대 서사를 말하게 된다.

    ⚠ 이것은 **구간 A(2024-12) 불변식**이다 — 이 모듈이 생성하는 구간이고,
    `_DOW_INDEX` 도 구간 A 실측이다. 2년 창에서는 특산품 토요일이 1.20(평균 이상)
    이라 `SAT < 1.0` 이 성립하지 않는다. 재현되는 것은 '일요일 > 토요일' 방향뿐이다
    (2023·2024 각 1.36 vs 1.20). 구간 A 가 넓어져 이 테스트가 깨지면 폴백 상수를
    새 구간 A 실측으로 갱신하고, 낙차를 인용하는 문서도 같이 고칠 것.
    """
    SAT, SUN = 5, 6
    local = stats.dow_index(stats.daily_qty(embedded_sales, S.CH_LOCAL))
    assert local[SAT] < 1.0, "특산품 토요일은 평균 이하여야 한다"
    assert local[SUN] / local[SAT] > 1.5, "토→일 급등 폭이 근거의 핵심"

    for channel in (S.CH_CASINO, S.CH_ROOM):
        d = stats.dow_index(stats.daily_qty(embedded_sales, channel))
        assert d[SAT] > 1.0, f"{channel} 토요일은 평균 이상이어야 한다"
        assert d[SUN] / d[SAT] < 1.2, f"{channel} 은 특산품 같은 급등이 없어야 한다"


def test_demographics_vip_ratio_matches_design():
    """폴백 성별·연령의 VIP 비중 = 실측 vip_ratio 0.666.

    VRB = 입장권 구매 건수 × vip_ratio 이므로 이 비율이 폴백 VIP 모수 카드의
    규모를 그대로 결정한다. 어긋나면 실측과 다른 모수를 말하게 된다.
    """
    demo = schema.normalize_demographics(fallback.demographics_df())
    total = demo["visitors"].sum()
    vip = demo.loc[demo["age_band"].isin(S.AGE_BANDS_VIP), "visitors"].sum()
    assert vip / total == pytest.approx(0.666, abs=0.01)
