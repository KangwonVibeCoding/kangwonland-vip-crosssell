"""실측값 회귀 테스트.

기대값은 원본 CSV 를 직접 분석해 얻은 수치다. 전처리·집계 로직을 리팩터링하다
수치가 흔들리면 여기서 즉시 잡힌다 — 이 프로젝트에서 가장 값진 테스트다.

실데이터(data/raw)가 없으면 해당 테스트는 skip 된다 (CI·클론 환경 대응).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import settings as S
from src.analysis import crosssell, inflow as inflow_mod, lag, stats, vip
from src.data import loaders

# ── 실측 기준값 ────────────────────────────────────────────────────────
# 판매 데이터는 2023-01-01 ~ 2024-12-31 (2년). 2024년분만 보면 각각
# 90,983 / 35,653 / 18,514 로 이전과 동일하다 — 2023년이 통째로 더해진 것이다.
SALES_START = pd.Timestamp("2023-01-01")
SALES_END = pd.Timestamp("2024-12-31")
EXPECTED_ROWS = {S.CH_CASINO: 179_287, S.CH_LOCAL: 69_353, S.CH_ROOM: 36_226}
EXPECTED_ROWS_2024 = {S.CH_CASINO: 90_983, S.CH_LOCAL: 35_653, S.CH_ROOM: 18_514}

# ARS 는 두 파일이 합쳐진다:
#   ars_20241201_20241231.csv  2024-12  31행 — 구간 A 의 유일한 근거
#   ars_merged.csv             2026 상반기 181행 (recv_total 계열 없음)
EXPECTED_ARS_ROWS = 212
EXPECTED_ARS_RECV_TOTAL_DAYS = 31     # recv_total 이 실제로 있는 날 수

# 무상 제공 상품 = '(V)' 접두 30,560개 + '무료' 포함 414,912개 (2023~2024 합산).
# 후자가 13배 크고 전부 카지노 식음의 컴플리멘터리 음료다('18)헛개차(무료)' 등).
EXPECTED_COMP_QTY = 445_472
EXPECTED_COMP_V_ONLY = 30_560
EXPECTED_COMP_FREE = 414_912

# 특산품은 2024-12-17 하루가 원본에 없다 (2023~2024 730일 중 유일한 결측일).
# 결측일을 0 으로 채우지 않고 내부 조인으로 제외하기 때문에 특산품만 n=30 이다.
# 일별 최소 판매량이 206 이므로 '판매 0인 날'이 아니라 데이터 누락으로 본다 —
# 0 을 채워 넣으면 없는 사실을 만드는 셈이고, 상관·요일지수가 모두 왜곡된다.
LOCAL_MISSING_DAY = pd.Timestamp("2024-12-17")
N_DAYS_FULL = 31
N_DAYS_LOCAL = 30

# (pearson, spearman, n_days)
EXPECTED_CORR = {
    ("recv_total", S.CH_CASINO): (0.801, 0.860, N_DAYS_FULL),
    ("recv_total", S.CH_ROOM): (0.740, 0.827, N_DAYS_FULL),
    ("recv_total", S.CH_LOCAL): (0.380, 0.463, N_DAYS_LOCAL),
    ("tickets", S.CH_CASINO): (0.796, 0.896, N_DAYS_FULL),
    ("tickets", S.CH_ROOM): (0.733, 0.783, N_DAYS_FULL),
    ("tickets", S.CH_LOCAL): (0.370, 0.568, N_DAYS_LOCAL),
}

# VIP 상관은 앱의 2티어 정의(프리미엄 + 선물번들, 컴프 제외) 기준이다. 탐색
# 단계의 좁은 정규식보다 넓지만 유입 반응을 더 잘 잡는다:
#   카지노 0.841(vs 0.750) · 특산품 0.513(vs 0.365) · 룸서비스 0.668(vs 0.684)
EXPECTED_VIP_CORR = {S.CH_CASINO: 0.841, S.CH_ROOM: 0.668, S.CH_LOCAL: 0.513}

EXPECTED_LAG_R = {S.CH_CASINO: {0: 0.796, 1: 0.295, 2: -0.296},
                  S.CH_ROOM: {0: 0.733, 1: 0.272, 2: -0.350},
                  S.CH_LOCAL: {0: 0.370, 1: 0.467, 2: 0.102}}
EXPECTED_BEST_LAG = {S.CH_CASINO: 0, S.CH_ROOM: 0, S.CH_LOCAL: 1}
EXPECTED_DOW_LOCAL = {"일": 1.629, "목": 0.642}
# 2023년 판매 데이터가 더해지면서 계절성이 2년 평균으로 희석됐다 (특산품 9월
# 1.48→1.43, 룸서비스 1월 1.23→1.21). 피크 월 자체는 그대로이므로 추석·설
# 성수기 해석은 유지된다 — 오히려 2개년에서 재현됐으니 근거가 강해진 것이다.
EXPECTED_MONTH = {S.CH_LOCAL: (9, 1.43), S.CH_ROOM: (1, 1.21)}
# 매월 1일 스파이크도 2년 평균에서 1.62→1.45 로 낮아졌다. robust 정규화가
# 필요하다는 근거(=1일이 스케일을 독점한다)는 그대로다.
EXPECTED_MONTH_FIRST_SPIKE = 1.45

TOL = 0.01          # 상관계수 허용 오차
TOL_IDX = 0.02      # 요일·월 인덱스 허용 오차

PERIOD_A_START = pd.Timestamp("2024-12-01")
PERIOD_A_END = pd.Timestamp("2024-12-31")


# ── 픽스처 ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def loaded():
    ars, ars_src = loaders._load_ars_impl()
    sales, sales_src = loaders._load_sales_impl()
    return {"ars": ars, "ars_src": ars_src, "sales": sales, "sales_src": sales_src}


@pytest.fixture(scope="module")
def real_data(loaded):
    if loaded["sales_src"] not in S.REAL_SOURCES or loaded["ars_src"] not in S.REAL_SOURCES:
        pytest.skip("실데이터(data/raw)가 없어 회귀 테스트를 건너뜁니다")
    return loaded


@pytest.fixture(scope="module")
def period_a(real_data):
    """구간 A(2024-12) — ARS 와 판매가 모두 있는 유일한 구간."""
    ars = real_data["ars"]
    sales = real_data["sales"]
    ars_a = ars.loc[ars["date"].between(PERIOD_A_START, PERIOD_A_END)]
    sales_a = sales.loc[sales["date"].between(PERIOD_A_START, PERIOD_A_END)]
    if len(ars_a) < 20 or sales_a.empty:
        pytest.skip("구간 A 데이터가 부족합니다")
    return ars_a, sales_a


# ── 로딩 / 스키마 ─────────────────────────────────────────────────────
def test_sales_row_counts(real_data):
    counts = real_data["sales"]["channel"].value_counts().to_dict()
    for channel, expected in EXPECTED_ROWS.items():
        assert counts.get(channel) == expected, f"{channel} 행 수 불일치"


def test_sales_total_rows(real_data):
    assert len(real_data["sales"]) == sum(EXPECTED_ROWS.values())


def test_sales_2024_slice_unchanged(real_data):
    """2024년분은 데이터 교체 전과 행 수가 완전히 같아야 한다.

    새 원본은 2023년을 덧붙였을 뿐 2024년을 다시 만들지 않았다는 확인이다.
    여기가 깨지면 구간 A 의 상관·래그 기준값도 더는 믿을 수 없다.
    """
    sales = real_data["sales"]
    y2024 = sales.loc[sales["date"].dt.year == 2024]
    counts = y2024["channel"].value_counts().to_dict()
    for channel, expected in EXPECTED_ROWS_2024.items():
        assert counts.get(channel) == expected, f"{channel} 2024년 행 수 불일치"


def test_sales_span_is_two_years(real_data):
    sales = real_data["sales"]
    assert sales["date"].min() == SALES_START
    assert sales["date"].max() == SALES_END


def test_ars_glob_concat(real_data):
    """ARS 로더가 여러 파일을 concat 하는지 — 구간 A 성립의 전제.

    2026년 전처리본만으로는 판매 데이터(2023~2024)와 겹치는 날이 0일이라
    실측 조인이 불가능하다. 2024-12 파일이 함께 잡혀야 한다.
    """
    ars = real_data["ars"]
    years = set(ars["date"].dt.year)
    assert {2024, 2026} <= years, f"ARS 가 여러 기간을 포함해야 합니다: {years}"
    assert len(ars) == EXPECTED_ARS_ROWS


def test_ars_recv_total_partially_present(real_data):
    """recv_total 이 2024-12 에만 있고 2026 구간에서는 결측이어야 한다.

    2026년 전처리본에서 총접수자 컬럼이 탈락했다. 0 으로 채우면 '그 날 접수자가
    0명'이라는 거짓이 정규화 스케일과 상관계수에 그대로 들어가므로, 결측을
    결측으로 유지하는 것이 요구사항이다.
    """
    ars = real_data["ars"]
    assert "recv_total" in ars.columns
    have = ars.loc[ars["recv_total"].notna()]
    assert len(have) == EXPECTED_ARS_RECV_TOTAL_DAYS
    assert set(have["date"].dt.year) == {2024}
    assert (ars.loc[ars["date"].dt.year == 2026, "recv_total"].isna()).all()


def test_buy_rate_identity(real_data):
    """구매율 == 구매건수 / 총당첨자 — 원본 정합성 (전 212행 일치)."""
    ars = real_data["ars"]
    derived = ars["tickets"] / ars["winners"]
    assert np.allclose(derived, ars["buy_rate"], atol=0.001)


def test_buy_rate_is_fraction(real_data):
    """% 표기(30~53)를 0~1 로 환산했는지."""
    assert real_data["ars"]["buy_rate"].between(0.0, 1.0).all()


def test_comp_items_isolated(real_data):
    """무상 제공 상품이 VIP 로 오탐되지 않고 별도 집계되는지."""
    sales = real_data["sales"]
    assert int(sales.loc[sales["is_comp"], "qty"].sum()) == EXPECTED_COMP_QTY
    v_only = sales.loc[sales["item"].str.startswith("(V)")]
    assert int(v_only["qty"].sum()) == EXPECTED_COMP_V_ONLY
    # '(V)산삼배양근진' 처럼 프리미엄 패턴에 걸리는 컴프가 실제로 있다 → 강등 확인
    comp = sales.loc[sales["is_comp"]]
    assert (comp["tier"] != S.TIER_PREMIUM).all(), "컴프 상품이 프리미엄으로 오탐됨"
    assert (comp["tier"] == S.TIER_STANDARD).all()


def test_free_items_are_comp(real_data):
    """'무료' 가 이름에 든 상품이 컴프로 잡히는지.

    누락되면 무상 음료(전체 수량의 5.5%)가 고마진 상품으로 집계되고, 저유입일
    판매가 0이라 lift 가 폭발해 CSM 상위를 통째로 점령한다.
    """
    sales = real_data["sales"]
    free = sales.loc[sales["item"].str.contains("무료", regex=False, na=False)]
    assert not free.empty
    assert free["is_comp"].all()
    assert int(free["qty"].sum()) == EXPECTED_COMP_FREE


def test_venue_asymmetry(real_data):
    """룸서비스·특산품은 영업장 1개, 카지노 식음만 5개 (실측)."""
    counts = real_data["sales"].groupby("channel")["venue"].nunique().to_dict()
    assert counts[S.CH_CASINO] == 5
    assert counts[S.CH_ROOM] == 1
    assert counts[S.CH_LOCAL] == 1


def test_no_hour_column(real_data):
    """주문 시각 컬럼이 없음을 명시적으로 고정한다.

    hour 가 생기면 이 테스트가 깨지고 — 그때는 히트맵을 시간축으로 되돌리면 된다.
    """
    assert "hour" not in real_data["sales"].columns


# ── 골프장 이용객 (신규 데이터셋) ──────────────────────────────────────
@pytest.fixture(scope="module")
def golf_real():
    df, source = loaders._load_golf_impl()
    if source not in S.REAL_SOURCES:
        pytest.skip("골프 실데이터(data/raw)가 없어 건너뜁니다")
    return df


def test_golf_drops_blank_padding_rows(golf_real):
    """원본 끝의 빈 패딩 행 47개가 제거되는지 (2,750 → 2,703).

    이 행들은 영업일자를 포함해 모든 값이 비어 있다. 날짜 결측 행이 하나라도
    남으면 add_date_parts 의 isocalendar().week 변환이 예외를 던진다.
    """
    assert len(golf_real) == 2_703
    assert golf_real["date"].notna().all()


def test_golf_schema(golf_real):
    """골프 표준 스키마 — 판매 데이터와 컬럼이 충돌하지 않아야 한다."""
    for col in S.GOLF_REQUIRED:
        assert col in golf_real.columns
    assert set(golf_real["golf_venue"].unique()) <= set(S.GOLF_VENUES)
    assert golf_real["date"].min() == pd.Timestamp("2021-03-19")
    assert golf_real["date"].max() == pd.Timestamp("2025-12-07")


def test_golf_closed_flag(golf_real):
    """휴장일을 행 삭제가 아니라 플래그로 남기는지 — 휴장 분포도 시즌 신호다."""
    assert golf_real["is_closed"].any()
    assert not golf_real["is_closed"].all()
    closed = golf_real.loc[golf_real["is_closed"]]
    assert set(closed["open_status"].unique()) <= set(S.GOLF_CLOSED_STATUSES)


def test_golf_is_separate_from_sales(real_data, golf_real):
    """골프가 판매 채널로 섞여 들어가지 않았는지.

    단위(이용인원)와 기간(2021~2025)이 판매 3종과 달라서 합치면 CAI·CSM 이
    전부 오염된다. 채널 집합이 3개로 유지되는 것이 경계선이다.
    """
    assert set(real_data["sales"]["channel"].unique()) == set(S.CHANNELS)
    assert "golf_venue" not in real_data["sales"].columns


# ── 상관계수 회귀 (핵심) ───────────────────────────────────────────────
def test_local_goods_missing_day(real_data):
    """특산품 결측일이 2024-12-17 하나뿐임을 고정한다.

    이 날 하나가 상관계수와 요일 인덱스를 눈에 띄게 흔든다(예: 일요일 지수
    1.629 vs 0채움 1.683). 결측 처리 방식이 바뀌면 여기서 먼저 드러난다.
    """
    sales = real_data["sales"]
    have = set(sales.loc[sales["channel"] == S.CH_LOCAL, "date"].unique())
    expected_missing = [
        d for d in pd.date_range(SALES_START, SALES_END) if d not in have
    ]
    assert expected_missing == [LOCAL_MISSING_DAY]

    # 다른 두 채널은 2년 730일이 빠짐없이 있다 — 특산품 결측이 채널 고유
    # 현상임을 고정한다 (전역 수집 누락이면 세 채널이 같이 빠졌을 것이다).
    for channel in (S.CH_CASINO, S.CH_ROOM):
        days = set(sales.loc[sales["channel"] == channel, "date"].unique())
        gaps = [d for d in pd.date_range(SALES_START, SALES_END) if d not in days]
        assert gaps == [], f"{channel} 에 예상 못한 결측일: {gaps[:5]}"


def test_corr_table_matches_measured(period_a):
    ars_a, sales_a = period_a
    corr = stats.corr_table(ars_a, sales_a)
    assert not corr.empty
    base = corr.loc[~corr["vip_only"]]
    for (metric, channel), (exp_p, exp_s, exp_n) in EXPECTED_CORR.items():
        row = base.loc[(base["metric"] == metric) & (base["channel"] == channel)]
        assert len(row) == 1, f"{metric} × {channel} 행 없음"
        assert row["n_days"].iloc[0] == exp_n
        assert row["pearson"].iloc[0] == pytest.approx(exp_p, abs=TOL)
        assert row["spearman"].iloc[0] == pytest.approx(exp_s, abs=TOL)


def test_headline_corr(period_a):
    """가설 검증 배너의 r=0.80 이 재현되는지."""
    ars_a, sales_a = period_a
    head = stats.headline_corr(stats.corr_table(ars_a, sales_a))
    assert head[S.CH_CASINO] == pytest.approx(0.801, abs=TOL)
    assert head[S.CH_ROOM] == pytest.approx(0.740, abs=TOL)


def test_vip_only_corr_holds(period_a):
    """VIP 태깅 상품만 봐도 상관이 유지되는지 (태깅이 노이즈가 아님)."""
    ars_a, sales_a = period_a
    corr = stats.corr_table(ars_a, sales_a)
    vip_rows = corr.loc[corr["vip_only"] & (corr["metric"] == "recv_total")]
    got = {r.channel: r.pearson for r in vip_rows.itertuples()}
    for channel, expected in EXPECTED_VIP_CORR.items():
        assert got[channel] == pytest.approx(expected, abs=TOL)


# ── 래그 시차 (차별화 기능) ────────────────────────────────────────────
def test_lag_matrix_matches_measured(period_a):
    ars_a, sales_a = period_a
    inflow = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    matrix = lag.lag_matrix(inflow, sales_a, max_lag=3)
    assert not matrix.empty
    for channel, per_lag in EXPECTED_LAG_R.items():
        for k, expected in per_lag.items():
            row = matrix.loc[(matrix["channel"] == channel) & (matrix["lag"] == k)]
            assert len(row) == 1
            assert row["pearson"].iloc[0] == pytest.approx(expected, abs=TOL)


def test_local_goods_peaks_next_day(period_a):
    """지역특산품은 유입 당일이 아니라 **익일**에 팔린다 — 프로젝트의 핵심 발견."""
    ars_a, sales_a = period_a
    inflow = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    best = lag.best_lags(lag.lag_matrix(inflow, sales_a, max_lag=3))
    got = {r.channel: int(r.best_lag) for r in best.itertuples()}
    assert got == EXPECTED_BEST_LAG


# ── 요일 / 월 인덱스 ──────────────────────────────────────────────────
def test_dow_index_local_goods(period_a):
    """특산품 일요일 1.68 / 목요일 0.66 — 체크아웃 선물 구매 패턴."""
    _, sales_a = period_a
    idx = stats.dow_index(stats.daily_qty(sales_a, S.CH_LOCAL))
    name_to_pos = {n: i for i, n in enumerate(S.DOW_NAMES)}
    for name, expected in EXPECTED_DOW_LOCAL.items():
        assert idx.get(name_to_pos[name]) == pytest.approx(expected, abs=TOL_IDX)


def test_month_index_full_year(real_data):
    """연간 계절성 (2023~2024 2년): 특산품 9월 1.43(추석), 룸서비스 1월 1.21."""
    sales = real_data["sales"]
    for channel, (month, expected) in EXPECTED_MONTH.items():
        idx = stats.month_index(stats.daily_qty(sales, channel))
        assert idx.get(month) == pytest.approx(expected, abs=TOL_IDX)
        assert int(idx.idxmax()) == month, f"{channel} 피크 월이 {month}월이 아님"


def test_seasonality_repeats_across_years(real_data):
    """계절성 피크가 2023·2024 두 해 모두에서 재현되는지.

    2년 평균만 보면 우연히 한 해의 이벤트가 평균을 끌어올린 것과 구분되지
    않는다. 각 연도에서 따로 확인해야 '계절성'이라고 말할 수 있다.
    """
    sales = real_data["sales"]
    for channel, (month, _) in EXPECTED_MONTH.items():
        for year in (2023, 2024):
            year_sales = sales.loc[sales["date"].dt.year == year]
            idx = stats.month_index(stats.daily_qty(year_sales, channel))
            assert idx.get(month) > 1.0, (
                f"{channel} {year}년 {month}월 지수가 평균 이하입니다: {idx.get(month):.3f}"
            )


def test_dow_profile_inflow_vs_local(period_a):
    """유입은 토요일 피크, 특산품은 일요일 피크 — 시퀀스의 근거."""
    ars_a, sales_a = period_a
    table = inflow_mod.dow_profile_table(ars_a, sales_a)
    ars_row = table.loc[table["series"] == "ars"].iloc[0]
    local_row = table.loc[table["series"] == S.CH_LOCAL].iloc[0]
    assert max(S.DOW_NAMES, key=lambda d: ars_row[d]) == "토"
    assert max(S.DOW_NAMES, key=lambda d: local_row[d]) == "일"


# ── 이상치 특성 ───────────────────────────────────────────────────────
def test_month_first_spike(real_data):
    """매월 1일 카지노 식음 판매량 스파이크 약 1.45배 — robust 정규화의 근거."""
    sales = real_data["sales"]
    daily = (
        sales.loc[sales["channel"] == S.CH_CASINO]
        .groupby(["date", "is_month_first"], as_index=False)["qty"].sum()
    )
    first = daily.loc[daily["is_month_first"], "qty"].mean()
    rest = daily.loc[~daily["is_month_first"], "qty"].mean()
    assert first / rest == pytest.approx(EXPECTED_MONTH_FIRST_SPIKE, abs=0.05)


# ── 스코어 분포 위생 ──────────────────────────────────────────────────
def test_scores_are_well_spread(period_a):
    """CII·VTS·CSM 이 0~100 범위이고 뭉치지 않는지 (정규화 버그 탐지)."""
    ars_a, sales_a = period_a
    cii = inflow_mod.compute_cii(ars_a)
    assert cii["cii"].between(0, 100).all()
    assert cii["cii"].std() >= 5, "CII 가 한 값에 뭉쳐 있음"

    infl = inflow_mod.build_inflow(ars_a, sales_a)
    vrb = vip.compute_vrb(infl, pd.DataFrame(
        {"gender": ["남", "여"], "age_band": ["40대", "40대"], "visitors": [100, 100]}
    ))
    vts = vip.compute_vts(infl, vrb, sales_a)
    assert vts["vts"].between(0, 100).all()
    assert vts["vts"].std() >= 5, "VTS 가 한 값에 뭉쳐 있음"

    csm = crosssell.compute_csm(sales_a, infl)
    assert csm["csm"].between(0, 100).all()
    assert csm["csm"].std() >= 5, "CSM 이 한 값에 뭉쳐 있음"


def test_lift_share_is_plausible(period_a):
    """lift > 1 상품이 전체의 20~80% — 100% 나 0% 면 hi/lo 분할 버그."""
    ars_a, sales_a = period_a
    infl = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    lift = crosssell.compute_lift(sales_a, infl)
    assert not lift.empty
    valid = lift["lift"].replace([np.inf, -np.inf], np.nan).dropna()
    share = float((valid > 1).mean())
    assert 0.2 <= share <= 0.8, f"lift>1 비율이 비현실적입니다: {share:.1%}"


def test_lift_is_bounded_and_gated(period_a):
    """가법 스무딩이 발산을 막고, 표본 미달 상품은 lift 를 내지 않는지.

    스무딩 전에는 저유입일 판매가 0인 상품(72종)이 hi_share/1e-9 로 계산돼 lift 가
    2,700만까지 튀었고 CSM 상위를 무상 음료가 독점했다. 표본 요건이 없을 때는
    판매일수 2일짜리 상품이 lift 20~60 으로 번들 추천에 올라왔다.
    """
    ars_a, sales_a = period_a
    infl = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    lift = crosssell.compute_lift(sales_a, infl)

    valid = lift["lift"].dropna()
    assert not valid.empty
    assert valid.max() < 10_000, "lift 가 발산했습니다 — 스무딩 확인"
    assert (valid > 0).all()

    # 표본 미달은 NaN 이어야 한다 (1.0 으로 채우면 없는 근거를 만드는 것)
    assert lift["lift"].isna().any(), "표본 요건이 적용되지 않았습니다"
    per_item = sales_a.groupby("item_id").agg(
        days=("date", "nunique"), qty=("qty", "sum")
    )
    gated = lift.merge(per_item, on="item_id", how="left")
    thin = gated.loc[(gated["days"] < S.LIFT_MIN_DAYS) | (gated["qty"] < S.LIFT_MIN_QTY)]
    assert thin["lift"].isna().all()


def test_bundles_are_regular_items(period_a):
    """번들 추천이 상시 판매 상품만 쓰는지 (판매일수 1~2일짜리 제외)."""
    ars_a, sales_a = period_a
    infl = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    csm = crosssell.compute_csm(sales_a, infl)
    bundles = crosssell.recommend_bundles(csm, sales_a, top_n=3)
    assert bundles
    days = sales_a.groupby("item")["date"].nunique()
    for b in bundles:
        assert days[b["room_item"]] >= S.BUNDLE_MIN_DAYS
        assert days[b["local_item"]] >= S.BUNDLE_MIN_DAYS
        assert b["co_days"] > 0


def test_csm_top_is_not_complimentary(period_a):
    """CSM 상위 20위 안에 무상 제공 상품이 없어야 한다."""
    ars_a, sales_a = period_a
    infl = inflow_mod.compute_cii(ars_a).rename(columns={"cii": "inflow"})
    csm = crosssell.compute_csm(sales_a, infl)
    top = csm.head(20)
    assert not top["is_comp"].any(), (
        "무상 제공 상품이 CSM 상위에 있습니다: "
        f"{top.loc[top['is_comp'], 'item'].tolist()}"
    )


def test_attach_rate_plausible(period_a):
    """attach rate 가 0 초과이고 비현실적으로 크지 않은지 (단위 불일치 탐지)."""
    ars_a, sales_a = period_a
    infl = inflow_mod.build_inflow(ars_a, sales_a)
    vrb = vip.compute_vrb(infl, pd.DataFrame(
        {"gender": ["남"], "age_band": ["40대"], "visitors": [100]}
    ))
    vts = vip.compute_vts(infl, vrb, sales_a)
    attach = vts["attach"].dropna()
    assert (attach > 0).all()
    assert attach.median() < 100, "attach rate 스케일 이상"


def test_cai_can_substitute_cii(period_a):
    """CAI 가 CII 를 대체할 근거 — 겹치는 구간 상관이 충분히 높아야 한다."""
    ars_a, sales_a = period_a
    v = inflow_mod.cai_validity(ars_a, sales_a)
    assert v["n_days"] == 31
    assert v["pearson"] > 0.6, f"CAI 대체 타당성 부족: r={v['pearson']:.3f}"


# ── 구간 C: recv_total 없는 ARS 에서의 CII 폴백 ────────────────────────
@pytest.fixture(scope="module")
def period_c(real_data):
    """구간 C(2026 상반기) — ARS 만 있고 총접수자 컬럼이 없는 구간."""
    ars = real_data["ars"]
    ars_c = ars.loc[ars["date"].dt.year == 2026]
    if len(ars_c) < 20:
        pytest.skip("구간 C 데이터가 부족합니다")
    return ars_c


def test_period_a_uses_pressure_signal(period_a):
    """구간 A 는 recv_total 이 있으므로 3축 CII 를 써야 한다."""
    ars_a, _ = period_a
    assert inflow_mod.has_pressure_signal(ars_a)
    assert bool(inflow_mod.compute_cii(ars_a)["pressure_signal"].iloc[0]) is True


def test_period_c_falls_back_without_recv_total(period_c):
    """총접수자가 없어도 CII 가 계산되고, 폴백이었음이 드러나야 한다.

    없는 컬럼을 0 으로 채워 3축 계산을 강행하면 '접수자 0명'이라는 거짓 신호가
    지수의 35% 를 차지한다. 축을 빼고 재정규화하는 것이 정직한 처리다.
    """
    assert not inflow_mod.has_pressure_signal(period_c)

    cii = inflow_mod.compute_cii(period_c)
    assert len(cii) == len(period_c)
    assert bool(cii["pressure_signal"].iloc[0]) is False
    assert cii["cii"].between(0, 100).all()
    assert cii["cii"].std() >= 5, "폴백 CII 가 한 값에 뭉쳐 있음"
    assert cii["cii"].notna().all()


def test_period_c_cii_ignores_stale_pressure_weights(period_c):
    """3축 가중치를 넘겨도 폴백 경로가 안전한지 (사이드바 슬라이더 대응).

    wsum 이 parts 에 없는 키를 버리고 남은 가중치로 재정규화하므로, 결과가
    2축 전용 가중치를 넘겼을 때와 같아야 한다.
    """
    with_stale = inflow_mod.compute_cii(period_c, S.INFLOW_WEIGHTS)
    with_clean = inflow_mod.compute_cii(period_c, S.INFLOW_WEIGHTS_NO_PRESSURE)
    assert np.allclose(with_stale["cii"], with_clean["cii"], atol=1e-9)


def test_corr_table_skips_all_nan_metric(period_c, real_data):
    """값이 전부 결측인 지표는 상관표에 실리지 않아야 한다.

    컬럼 존재만 확인하면 recv_total 행이 NaN 상관계수로 표에 남는다.
    """
    sales = real_data["sales"]
    corr = stats.corr_table(period_c, sales)
    # 구간 C 는 판매와 겹치는 날이 없으므로 표 자체가 비어야 정상이다
    if not corr.empty:
        assert "recv_total" not in set(corr["metric"])
        assert corr["pearson"].notna().all()


def test_headline_corr_falls_back_to_tickets(period_a):
    """recv_total 이 빠진 상관표에서는 tickets 로 배너 지표가 내려가는지."""
    ars_a, sales_a = period_a
    corr = stats.corr_table(ars_a, sales_a)
    assert stats.headline_metric(corr) == "recv_total"

    without = corr.loc[corr["metric"] != "recv_total"]
    assert stats.headline_metric(without) == "tickets"
    head = stats.headline_corr(without)
    assert head[S.CH_CASINO] == pytest.approx(0.796, abs=TOL)
    assert stats.headline_corr(corr.iloc[0:0]) == {}


# ── 요일 교란 통제 (README 가설 검증 절의 편상관 수치) ─────────────────
# 원 상관 0.801 은 그대로 방어되지 않는다. 토요일엔 유입도 매출도 많으므로 요일이
# 양변을 동시에 밀어올리는 공통 원인일 수 있다. 요일 더미로 회귀한 잔차끼리의
# 상관(편상관)을 고정해 둔다 — 절반 이상이 주말 효과였다는 사실이 발표의 근거다.
#
# 계산 근거: scripts/diagnose_confound.py (부트스트랩 CI·순열검정 p 포함)
EXPECTED_PARTIAL_CORR = {
    S.CH_CASINO: 0.370,   # 0.801 에서 하락. p=0.044 로 경계선
    S.CH_ROOM: 0.558,     # 0.740 에서 하락. p=0.0015 로 유일하게 견고
    S.CH_LOCAL: 0.176,    # 0.380 에서 하락. p=0.357 — 유의하지 않다
}


def _residual_on_dow(y: np.ndarray, dow: np.ndarray) -> np.ndarray:
    """요일로 설명되는 변동을 제거한 잔차. 더미는 6개(월요일이 기준 범주)."""
    X = np.column_stack([np.ones(len(y))]
                        + [(dow == d).astype(float) for d in range(1, 7)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def test_partial_corr_controlling_dow(period_a):
    """요일을 통제해도 남는 상관 — README 가 인용하는 수치."""
    ars_a, sales_a = period_a
    for channel, expected in EXPECTED_PARTIAL_CORR.items():
        x_s, y_s = stats.align(
            ars_a.set_index("date")["recv_total"], stats.daily_qty(sales_a, channel)
        )
        dow = x_s.index.dayofweek.to_numpy()
        got = stats.pearson(
            _residual_on_dow(x_s.to_numpy(float), dow),
            _residual_on_dow(y_s.to_numpy(float), dow),
        )
        assert got == pytest.approx(expected, abs=0.01), (
            f"{channel} 편상관 {got:.3f} != {expected}"
        )


def test_roomservice_is_the_robust_channel(period_a):
    """요일 통제 후에는 룸서비스가 카지노 식음보다 강하다 — 순서가 뒤집힌다.

    원 상관만 보면 카지노 식음(0.801) > 룸서비스(0.740) 이지만, 요일 효과를 빼면
    반대가 된다. 요일은 마케팅으로 바꿀 수 없는 변수이므로 통제 후에 남는 관계가
    실제 개입 여지다 — 교차판매 1순위를 룸서비스로 정한 근거.
    """
    assert EXPECTED_PARTIAL_CORR[S.CH_ROOM] > EXPECTED_PARTIAL_CORR[S.CH_CASINO]

    ars_a, sales_a = period_a
    got = {}
    for channel in (S.CH_CASINO, S.CH_ROOM):
        x_s, y_s = stats.align(
            ars_a.set_index("date")["recv_total"], stats.daily_qty(sales_a, channel)
        )
        dow = x_s.index.dayofweek.to_numpy()
        got[channel] = stats.pearson(
            _residual_on_dow(x_s.to_numpy(float), dow),
            _residual_on_dow(y_s.to_numpy(float), dow),
        )
    assert got[S.CH_ROOM] > got[S.CH_CASINO], (
        f"요일 통제 후 순서가 뒤집히지 않았습니다: {got}"
    )


def test_local_goods_sunday_spike_carries_the_d1_claim(period_a):
    """D+1 주장의 실제 근거 — 특산품만 토요일 저조 → 일요일 급등.

    래그 상관의 D+1(0.467) 대 D+0(0.370) 차이는 n≈30 에서 통계적으로 구별되지
    않는다(부트스트랩 CI 가 0을 포함). 주장을 지탱하는 것은 이 요일 대비다.
    """
    ars_a, sales_a = period_a
    SAT, SUN = 5, 6

    local = stats.dow_index(stats.daily_qty(sales_a, S.CH_LOCAL))
    assert local[SAT] < 1.0, "특산품 토요일이 평균 이하여야 한다"
    assert local[SUN] / local[SAT] > 1.5, "토→일 급등 폭이 근거의 핵심"

    # 다른 채널은 토·일 모두 높다 — 특산품만 일요일에 몰린다는 것이 요점
    for channel in (S.CH_CASINO, S.CH_ROOM):
        d = stats.dow_index(stats.daily_qty(sales_a, channel))
        assert d[SAT] > 1.0, f"{channel} 토요일은 평균 이상이어야 한다"
        assert d[SUN] / d[SAT] < 1.2, f"{channel} 은 특산품 같은 급등이 없어야 한다"

    # 유입은 토요일이 피크 — 판매가 하루 뒤인 구조
    ars_dow = stats.dow_index(ars_a.set_index("date")["tickets"])
    assert int(ars_dow.idxmax()) == SAT
