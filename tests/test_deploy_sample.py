"""배포 환경 시뮬레이션 — data/raw 없이 data/sample 만으로 동작하는지.

Streamlit Cloud 는 `.gitignore` 때문에 `data/raw/` 를 받지 못한다. 커밋 대상인
`data/sample/` 만으로 **가설 검증 배너의 r=0.80 이 실측값으로 떠야** 데모가 성립한다.
여기서 깨지면 배포판이 조용히 내장 데모 데이터로 돌아간다.

RAW / PROCESSED 경로를 존재하지 않는 곳으로 바꿔치기해 실제 파일은 건드리지 않는다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import lag, stats
from src.data import loaders


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    """raw / processed 가 없는 상태 = Cloud 배포 환경."""
    if not any(S.SAMPLE_DIR.glob("*.parquet")):
        pytest.skip("data/sample 이 비어 있습니다 — python scripts/make_sample.py 실행 필요")
    monkeypatch.setattr(S, "RAW_DIR", tmp_path / "no_raw")
    monkeypatch.setattr(S, "PROCESSED_DIR", tmp_path / "no_processed")
    monkeypatch.setattr(S, "MOCK_DIR", tmp_path / "no_mock")
    return loaders.load_all_impl()


def test_sales_and_ars_come_from_sample(deployed):
    data, sources = deployed
    for key in ("ars", "sales", "golf"):
        assert sources[key] == S.SRC_SAMPLE, f"{key} 출처가 {sources[key]}"
        assert sources[key] in S.REAL_SOURCES


def test_sample_is_not_truncated(deployed):
    """축약본이 2년 전체를 담고 있는지 — 월 인덱스 분석에 필요하다."""
    data, _ = deployed
    sales = data["sales"]
    assert len(sales) == 284_866
    assert sales["date"].dt.month.nunique() == 12
    assert set(sales["date"].dt.year) == {2023, 2024}
    assert len(data["ars"]) == 212
    assert len(data["golf"]) == 2_703


def test_sample_ars_carries_period_a(deployed):
    """축약본 ARS 에 2024-12 이 남아 있는지 — 배포판 구간 A 의 전제.

    2026년 ARS 전처리본에는 2024-12 이 없다. 이 31일이 빠지면 판매 데이터와
    겹치는 날이 0일이 되어 아래 두 테스트(r=0.80, D+1)가 근거를 잃는다.
    """
    data, _ = deployed
    ars = data["ars"]
    period_a = ars.loc[ars["date"].between(
        pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")
    )]
    assert len(period_a) == 31
    # 총 접수자까지 살아 있어야 배너 기준축이 recv_total 로 유지된다
    assert period_a["recv_total"].notna().all()


def test_headline_correlation_survives_deployment(deployed):
    """배포판에서도 r=0.80 이 실측으로 재현되는지 — 데모의 핵심."""
    data, _ = deployed
    start, end = pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")
    ars = data["ars"].loc[data["ars"]["date"].between(start, end)]
    sales = data["sales"].loc[data["sales"]["date"].between(start, end)]
    head = stats.headline_corr(stats.corr_table(ars, sales))
    assert head[S.CH_CASINO] == pytest.approx(0.801, abs=0.01)
    assert head[S.CH_ROOM] == pytest.approx(0.740, abs=0.01)


def test_lag_finding_survives_deployment(deployed):
    """특산품 D+1 발견이 배포판에서도 재현되는지."""
    data, _ = deployed
    start, end = pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")
    ars = data["ars"].loc[data["ars"]["date"].between(start, end)]
    sales = data["sales"].loc[data["sales"]["date"].between(start, end)]
    infl = inflow_mod.compute_cii(ars).rename(columns={"cii": "inflow"})
    best = lag.best_lags(lag.lag_matrix(infl, sales, max_lag=3))
    got = {r.channel: int(r.best_lag) for r in best.itertuples()}
    assert got == {S.CH_CASINO: 0, S.CH_ROOM: 0, S.CH_LOCAL: 1}


def test_seasonality_survives_deployment(deployed):
    """월 인덱스(특산품 9월 1.43 / 룸서비스 1월 1.21) — 기간을 자르지 않은 이유.

    2023년이 더해지며 2년 평균으로 희석됐다(9월 1.48→1.43, 1월 1.23→1.21).
    피크 월은 그대로이고 두 해 모두에서 재현된다.
    """
    data, _ = deployed
    sales = data["sales"]
    local = stats.month_index(stats.daily_qty(sales, S.CH_LOCAL))
    room = stats.month_index(stats.daily_qty(sales, S.CH_ROOM))
    assert local.get(9) == pytest.approx(1.43, abs=0.02)
    assert room.get(1) == pytest.approx(1.21, abs=0.02)


def test_api_datasets_fall_back_without_key(deployed):
    """키가 없으면 성별·연령/가맹점은 폴백으로 뜨되 앱은 살아 있어야 한다."""
    data, sources = deployed
    for key in ("demo", "merchants"):
        assert not data[key].empty
        assert sources[key] in S.SOURCE_LABEL


def test_sample_carries_no_unused_identifiers(deployed):
    """커밋되는 축약본에 앱이 안 쓰는 식별 정보가 실리지 않았는지.

    가맹점 API 는 사업자등록번호·전화번호를 함께 주는데 지도·스코어·표 어디에서도
    쓰지 않는다. data/sample 은 공개 저장소에 커밋되므로 쓰지 않는 컬럼을 실어
    보내면 노출면만 늘어난다 (scripts/make_sample.py 의 DROP_COLUMNS).
    """
    data, sources = deployed
    merchants = data["merchants"]
    assert sources["merchants"] == S.SRC_SAMPLE
    for column in ("biz_no", "tel"):
        assert column not in merchants.columns, f"{column} 이 축약본에 남아 있습니다"
    # 화면이 실제로 쓰는 컬럼은 그대로 있어야 한다
    for column in ("merchant", "lat", "lon", "category", "has_coord"):
        assert column in merchants.columns
    assert int(merchants["has_coord"].sum()) == 1_540


def test_no_real_data_at_all(tmp_path, monkeypatch):
    """data/ 전체가 없어도 내장 폴백으로 앱이 살아남는지 (최종 안전장치)."""
    monkeypatch.setattr(S, "RAW_DIR", tmp_path / "x")
    monkeypatch.setattr(S, "PROCESSED_DIR", tmp_path / "y")
    monkeypatch.setattr(S, "SAMPLE_DIR", tmp_path / "z")
    monkeypatch.setattr(S, "MOCK_DIR", tmp_path / "w")
    data, sources = loaders.load_all_impl()
    for key in ("ars", "sales", "golf", "demo", "merchants"):
        assert not data[key].empty, f"{key} 가 비었습니다 — 폴백 실패"
        assert sources[key] == S.SRC_EMBEDDED
    # 내장 데이터로도 스코어링이 돌아야 한다
    infl = inflow_mod.build_inflow(data["ars"], data["sales"])
    assert not infl.empty
    assert infl["inflow"].between(0, 100).all()
