"""뷰 로직 스모크 테스트 — 브라우저 없이 각 탭이 쓰는 계산을 전부 실행한다.

streamlit 위젯 렌더는 확인할 수 없지만, 뷰가 호출하는 분석·차트 함수는 모두
여기서 돌려본다. 엣지케이스(빈 데이터, 단일 행, 판매 없는 구간)도 함께 검증한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings as S
from src.analysis import crosssell, inflow as inflow_mod, lag, stats, vip
from src.data import loaders
from src.ui import charts, maps, theme


@pytest.fixture(scope="module")
def bundle():
    theme.register_plotly_template()
    data, sources = loaders.load_all_impl()
    return data, sources


@pytest.fixture(scope="module")
def period_a(bundle):
    data, _ = bundle
    start, end = pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")
    ars = data["ars"].loc[data["ars"]["date"].between(start, end)]
    sales = data["sales"].loc[data["sales"]["date"].between(start, end)]
    if ars.empty or sales.empty:
        pytest.skip("구간 A 데이터 없음")
    return ars, sales


def test_all_sources_resolved(bundle):
    """4개 데이터셋이 모두 어떤 계층으로든 해결됐는지 (None 금지)."""
    data, sources = bundle
    for key in ("ars", "sales", "demo", "merchants"):
        assert key in data and not data[key].empty
        assert sources[key] in S.SOURCE_LABEL


def test_overview_pipeline(period_a, bundle):
    """탭1 이 쓰는 전 계산 경로."""
    data, _ = bundle
    ars, sales = period_a
    corr = stats.corr_table(ars, sales)
    head = stats.headline_corr(corr)
    assert head and max(head.values()) > 0.5

    infl = inflow_mod.build_inflow(ars, sales)
    vrb = vip.compute_vrb(infl, data["demo"])
    vts = vip.compute_vts(infl, vrb, sales)
    calendar = vip.campaign_calendar(vts)
    assert len(calendar) == S.TOP_N_VTS_DAYS
    assert calendar["vts"].is_monotonic_decreasing

    seg = vip.segment_table(data["demo"])
    assert set(seg["segment"]) <= {"VIP(40~60대)", "준VIP(30대)", "기타"}

    # 차트 생성이 예외 없이 되는지
    assert charts.inflow_stack(infl).data
    assert charts.corr_bars(head).data
    assert charts.segment_donut(seg).data
    assert charts.age_gender_bars(data["demo"]).data


def test_fnb_pipeline(period_a):
    """탭2 — 히트맵 / CSM / 트리맵 / 번들."""
    ars, sales = period_a
    infl = inflow_mod.build_inflow(ars, sales)
    csm = crosssell.compute_csm(sales, infl)
    assert not csm.empty
    assert csm["csm"].between(0, 100).all()
    assert csm["csm"].is_monotonic_decreasing

    pivot = crosssell.heatmap_dow_month(sales, S.CH_ROOM)
    assert list(pivot.columns) == list(S.DOW_NAMES)
    assert charts.dow_month_heatmap(pivot).data

    frame = crosssell.treemap_frame(sales, S.CH_CASINO)
    assert not frame.empty
    assert charts.venue_treemap(frame).data

    top = crosssell.top_items(csm, S.CH_ROOM, n=15, exclude_comp=True)
    assert not top.empty
    assert charts.top_items_bar(top).data

    bundles = crosssell.recommend_bundles(csm, sales, top_n=3)
    assert len(bundles) == 3
    for b in bundles:
        assert b["co_days"] > 0


def test_timing_pipeline(period_a):
    """탭3 — 래그 행렬 / 상품별 시차 / 처방."""
    ars, sales = period_a
    infl = inflow_mod.compute_cii(ars).rename(columns={"cii": "inflow"})
    matrix = lag.lag_matrix(infl, sales, max_lag=3)
    best = lag.best_lags(matrix)
    assert len(best) == 3
    assert charts.lag_heatmap(matrix).data

    dow = inflow_mod.dow_profile_table(ars, sales)
    assert len(dow) == 4        # ARS + 3채널
    assert charts.dow_profile_lines(dow).data

    month = inflow_mod.month_profile_table(sales)
    assert charts.month_profile_lines(month).data

    item_lags = lag.best_lag_by_item(infl, sales, max_lag=3)
    assert not item_lags.empty
    assert set(item_lags["best_lag"].unique()) <= {0, 1, 2, 3}

    cards = lag.prescriptions(best, dow, month)
    assert cards
    for card in cards:
        assert card["title"] and card["evidence"]


def test_local_curation_pipeline(period_a, bundle):
    """탭4 — 가맹점 스코어 / 지도 / 패키지."""
    data, _ = bundle
    ars, sales = period_a
    deck = maps.build_merchants(data["merchants"], radius_km=30.0)
    assert not deck.empty
    assert (deck["dist_km"] <= 30.0).all()
    assert deck["fit_score"].between(0, 100).all()
    assert deck["fit_score"].is_monotonic_decreasing
    assert deck["group"].isin(["음식", "소매", "숙박", S.CATEGORY_GROUP_OTHER]).all()

    # 업종 적합도는 공개된 표(S.CATEGORY_FIT)의 값이거나 중립값이어야 한다.
    # 임의의 숫자가 섞이면 '공개한 판단'이라는 근거가 무너진다.
    allowed = set(S.CATEGORY_FIT.values()) | {S.CATEGORY_FIT_NEUTRAL}
    assert set(deck["fit"].round(3)) <= {round(v, 3) for v in allowed}
    # 업종을 모르는 행은 반드시 중립값이다 (0 으로 눕히면 없는 근거를 만든다)
    unknown = ~deck["fit_known"].to_numpy(dtype=bool)
    if unknown.any():
        assert (deck.loc[unknown, "fit"] == S.CATEGORY_FIT_NEUTRAL).all()

    # pydeck Deck 생성 (외부 타일 요청 없음)
    deck_obj = maps.merchant_deck(deck, 30.0)
    assert deck_obj.layers

    infl = inflow_mod.build_inflow(ars, sales)
    csm = crosssell.compute_csm(sales, infl)
    local_top = crosssell.top_items(csm, S.CH_LOCAL, n=6, exclude_comp=True)
    packages = maps.curation_packages(deck, local_top, top_n=3)
    assert len(packages) == 3

    daily = stats.daily_qty(
        data["sales"].loc[data["sales"]["channel"] == S.CH_LOCAL]
    )
    assert charts.season_line(daily).data


def test_merchants_without_coords_are_counted_not_dropped(bundle):
    """좌표를 못 찾은 가맹점을 조용히 지우지 않는다.

    가맹점 좌표는 상호·주소 매칭이라 100% 가 되지 않는다(실측 91.6%). 나머지를
    말없이 버리면 '반경 내 N곳'이 전수인 것처럼 읽힌다. 로더는 행을 남기고,
    지도는 제외하되 **제외 건수를 attrs 로 돌려줘야** 화면에 표기할 수 있다.
    """
    data, _ = bundle
    merchants = data["merchants"]
    assert "has_coord" in merchants.columns

    deck = maps.build_merchants(merchants, radius_km=30.0)
    assert deck.attrs.get("no_coord") is not None
    expected = int((~merchants["has_coord"].to_numpy(dtype=bool)).sum())
    assert deck.attrs["no_coord"] == expected
    # 지도에 올라간 행은 전부 좌표가 있다
    assert deck["lat"].notna().all() and deck["lon"].notna().all()


def test_point_limit_is_not_a_score_signal(bundle):
    """포인트 사용 한도액을 스코어에 쓰지 않는다.

    실측 1,681곳 중 1,578곳이 정확히 400만원(CV 0.052)이라 변별력이 없다.
    ARS 의 winners(정원 캡)를 유입 지표에서 뺀 것과 같은 이유다.
    """
    data, _ = bundle
    deck = maps.build_merchants(data["merchants"], radius_km=30.0)
    if "point_limit" not in deck.columns:
        return                      # 내장 폴백에는 이 컬럼이 없다
    limit = pd.to_numeric(deck["point_limit"], errors="coerce").dropna()
    if len(limit) > 10 and limit.mean():
        assert limit.std() / limit.mean() < 0.2, "한도액이 상수가 아니게 됐다면 재검토"
    # 상수 컬럼이 점수에 섞이지 않았는지 — 적합도는 거리·업종만으로 재현된다
    proximity = np.exp(-deck["dist_km"] / S.PROXIMITY_DECAY_KM)
    expected = 100.0 * (S.CURATION_WEIGHTS["proximity"] * proximity
                        + S.CURATION_WEIGHTS["fit"] * deck["fit"])
    assert np.allclose(deck["fit_score"], expected)


def test_haversine_sanity():
    """거리 계산 위생 — 기준점 자기 자신은 0km."""
    assert maps.haversine_km(*S.CASINO_LATLON) == pytest.approx(0.0, abs=1e-6)
    # 정선 아리랑시장(37.3805, 128.6606) 은 대략 25km 권
    d = float(maps.haversine_km(37.3805, 128.6606))
    assert 15 < d < 35


# ── 엣지케이스 방어 ───────────────────────────────────────────────────
def test_empty_inputs_do_not_raise(bundle):
    """빈 DataFrame 을 넣어도 예외 없이 빈 결과를 돌려주는지."""
    data, _ = bundle
    empty_sales = data["sales"].iloc[0:0]
    empty_ars = data["ars"].iloc[0:0]

    assert stats.corr_table(empty_ars, empty_sales).empty
    assert stats.headline_corr(pd.DataFrame()) == {}
    assert inflow_mod.compute_cii(empty_ars).empty
    assert inflow_mod.compute_cai(empty_sales).empty
    assert inflow_mod.build_inflow(empty_ars, empty_sales).empty
    assert crosssell.compute_csm(empty_sales, pd.DataFrame()).empty
    assert crosssell.heatmap_dow_month(empty_sales).empty
    assert crosssell.recommend_bundles(pd.DataFrame(), empty_sales) == []
    assert lag.lag_matrix(pd.DataFrame(), empty_sales).empty
    assert lag.best_lags(pd.DataFrame()).empty
    assert maps.build_merchants(data["merchants"].iloc[0:0]).empty
    assert vip.compute_vrb(pd.DataFrame(), data["demo"]).empty


def test_single_row_does_not_raise(bundle):
    """단일 행 — 정규화가 0으로 나누지 않는지."""
    data, _ = bundle
    one_ars = data["ars"].head(1)
    one_sales = data["sales"].head(1)
    cii = inflow_mod.compute_cii(one_ars)
    assert len(cii) == 1
    assert cii["cii"].between(0, 100).all()
    csm = crosssell.compute_csm(one_sales, cii.rename(columns={"cii": "inflow"}))
    assert len(csm) == 1
    assert csm["csm"].between(0, 100).all()


def test_period_c_has_no_sales(bundle):
    """구간 C(2026-05)에는 판매 데이터가 없다 — 뷰가 안내로 처리해야 하는 조건."""
    data, _ = bundle
    start, end = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-31")
    ars_c = data["ars"].loc[data["ars"]["date"].between(start, end)]
    sales_c = data["sales"].loc[data["sales"]["date"].between(start, end)]
    if ars_c.empty:
        pytest.skip("구간 C ARS 없음")
    assert not ars_c.empty
    assert sales_c.empty
    # ARS 만으로도 유입 지수는 계산돼야 한다
    infl = inflow_mod.build_inflow(ars_c, sales_c)
    assert not infl.empty
    assert (infl["source"] == "CII").all()


def test_all_tiers_filtered_out(bundle):
    """모든 티어를 제외해도 빈 결과로 안전하게 떨어지는지."""
    data, _ = bundle
    none = data["sales"].loc[data["sales"]["tier"] == "__없는티어__"]
    assert none.empty
    assert crosssell.compute_csm(none, pd.DataFrame()).empty
    assert stats.daily_qty(none).empty
