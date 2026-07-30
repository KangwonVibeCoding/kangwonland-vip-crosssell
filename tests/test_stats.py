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

# ── 실측 기준값 (2024-12, 31일 실제 날짜 조인) ─────────────────────────
EXPECTED_ROWS = {S.CH_CASINO: 90_983, S.CH_LOCAL: 35_653, S.CH_ROOM: 18_514}

# 무상 제공 상품 = '(V)' 접두 11종 21,870개 + '무료' 포함 38종 210,372개.
# 후자가 10배 크고 전부 카지노 식음의 컴플리멘터리 음료다('18)헛개차(무료)' 등).
EXPECTED_COMP_QTY = 232_242
EXPECTED_COMP_V_ONLY = 21_870

# 특산품은 2024-12-17 하루가 원본에 없다 (2024년 365일 중 유일한 결측일).
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
EXPECTED_MONTH = {S.CH_LOCAL: (9, 1.48), S.CH_ROOM: (1, 1.23)}

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


def test_ars_glob_concat(real_data):
    """ARS 로더가 여러 파일을 concat 하는지 — 파일 추가 시 구간 확장의 전제."""
    ars = real_data["ars"]
    years = set(ars["date"].dt.year)
    assert {2024, 2026} <= years, f"ARS 가 여러 기간을 포함해야 합니다: {years}"
    assert len(ars) == 62


def test_buy_rate_identity(real_data):
    """구매율 == 구매건수 / 총당첨자 — 원본 정합성 (실측 62/62행 일치)."""
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
    assert int(free["qty"].sum()) == 210_372


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


# ── 상관계수 회귀 (핵심) ───────────────────────────────────────────────
def test_local_goods_missing_day(real_data):
    """특산품 결측일이 2024-12-17 하나뿐임을 고정한다.

    이 날 하나가 상관계수와 요일 인덱스를 눈에 띄게 흔든다(예: 일요일 지수
    1.629 vs 0채움 1.683). 결측 처리 방식이 바뀌면 여기서 먼저 드러난다.
    """
    sales = real_data["sales"]
    have = set(sales.loc[sales["channel"] == S.CH_LOCAL, "date"].unique())
    expected_missing = [
        d for d in pd.date_range("2024-01-01", "2024-12-31") if d not in have
    ]
    assert expected_missing == [LOCAL_MISSING_DAY]


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
    """연간 계절성: 특산품 9월 1.48(추석), 룸서비스 1월 1.23(설·스키)."""
    sales = real_data["sales"]
    for channel, (month, expected) in EXPECTED_MONTH.items():
        idx = stats.month_index(stats.daily_qty(sales, channel))
        assert idx.get(month) == pytest.approx(expected, abs=TOL_IDX)


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
    """매월 1일 카지노 식음 판매량 스파이크 약 1.6배 — robust 정규화의 근거."""
    sales = real_data["sales"]
    daily = (
        sales.loc[sales["channel"] == S.CH_CASINO]
        .groupby(["date", "is_month_first"], as_index=False)["qty"].sum()
    )
    first = daily.loc[daily["is_month_first"], "qty"].mean()
    rest = daily.loc[~daily["is_month_first"], "qty"].mean()
    assert first / rest == pytest.approx(1.62, abs=0.05)


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
