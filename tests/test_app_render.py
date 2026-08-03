"""app.py 실제 렌더 검증 — Streamlit AppTest 로 스크립트를 통째로 실행한다.

브라우저 없이 위젯 생성·뷰 렌더·예외를 전부 잡는다. `app.py` 는 탭 단위로 예외를
격리하므로, 뷰가 터져도 앱은 살아남는다 — 그래서 "예외 없음" 뿐 아니라
**st.error 가 하나도 없음**까지 확인해야 실제로 정상 렌더된 것이다.
"""

from __future__ import annotations

import pytest

from config import settings as S

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(S.ROOT / "app.py")
TIMEOUT = 120


def _run(**session_state):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for key, value in session_state.items():
        at.session_state[key] = value
    return at.run()


@pytest.fixture(scope="module")
def app():
    return _run()


def test_app_runs_without_exception(app):
    assert not app.exception, f"앱 실행 중 예외: {app.exception}"


def test_no_tab_render_errors(app):
    """탭 단위 예외 격리가 삼킨 오류가 없는지 — st.error 가 곧 렌더 실패다."""
    errors = [e.value for e in app.error]
    assert not errors, f"탭 렌더 오류: {errors}"


def test_four_tabs_present(app):
    assert len(app.tabs) == 4, f"탭 4개가 아님: {len(app.tabs)}"


def test_title_and_kpis_rendered(app):
    """헤더와 KPI 카드가 실제로 그려졌는지 (markdown 으로 렌더된다)."""
    blob = " ".join(m.value for m in app.markdown)
    assert "VIP 카지노–리조트 교차판매 대시보드" in blob
    assert "kl-kpi" in blob, "KPI 카드가 렌더되지 않았습니다"
    assert "kl-hero" in blob, "가설 검증 배너가 렌더되지 않았습니다"


def test_period_badge_rendered(app):
    """구간 배지 — 어느 근거로 그린 값인지 화면에 드러나야 한다."""
    blob = " ".join(m.value for m in app.markdown)
    assert "kl-badge" in blob
    assert any(b in blob for b in ("실측 조인", "요일 브릿지", "최신 유입"))


def test_headline_correlation_is_real(app):
    """배너의 r 값이 모킹이 아닌 실측(0.80)으로 뜨는지."""
    blob = " ".join(m.value for m in app.markdown)
    if "데이터가 겹치지 않습니다" in blob:
        pytest.skip("기본 구간에 조인 가능한 데이터가 없음")
    assert "상관 r = 0.80" in blob, "가설 검증 배너의 상관계수가 실측값이 아닙니다"


def test_sidebar_widgets_have_defaults(app):
    """로그인·초기설정 관문 없이 즉시 조작 가능해야 한다 (심사 요구사항)."""
    assert app.sidebar.radio, "사이드바 라디오가 없습니다"
    assert app.sidebar.multiselect, "사이드바 multiselect 가 없습니다"
    # 기본값이 비어 있으면 첫 화면이 빈 상태로 뜬다
    period = app.sidebar.radio(key="flt_period")
    assert period.value is not None
    assert list(app.sidebar.multiselect(key="flt_channels").value), "채널 기본값 없음"
    assert list(app.sidebar.multiselect(key="flt_age").value), "연령대 기본값 없음"
    assert list(app.sidebar.multiselect(key="flt_tiers").value), "티어 기본값 없음"


def test_charts_rendered(app):
    """plotly 차트가 실제로 생성됐는지."""
    assert len(app.get("plotly_chart")) >= 4, "차트가 충분히 렌더되지 않았습니다"


def test_dataframes_rendered(app):
    """캠페인 캘린더 / CSM 테이블 등 표가 렌더됐는지."""
    assert len(app.dataframe) >= 1


# ── 엣지케이스: 필터를 극단으로 밀어도 앱이 살아남는지 ──────────────────
def test_all_age_bands_cleared():
    """연령대를 전부 해제 → 에러 대신 안내(st.info)."""
    at = _run(flt_age=[])
    assert not at.exception
    assert not [e.value for e in at.error]
    assert at.info, "빈 세그먼트 안내가 표시되지 않았습니다"


def test_all_channels_cleared():
    """채널을 전부 해제 → 판매 기반 탭이 안내로 처리돼야 한다."""
    at = _run(flt_channels=[])
    assert not at.exception
    assert not [e.value for e in at.error]
    assert at.info


def test_single_channel_keeps_entity_colors():
    """채널 하나만 남겨도 렌더되는지 (색상=엔티티 고정은 charts 단에서 보장)."""
    at = _run(flt_channels=[S.CH_LOCAL])
    assert not at.exception
    assert not [e.value for e in at.error]


def test_period_c_has_no_sales_data():
    """구간 C(2026 상반기)는 판매 데이터가 없다 → 안내로 graceful 처리.

    이 구간은 ARS 전처리본에 총 접수자 컬럼도 없어 CII 가 2축으로 폴백한다.
    지수 계산 경로가 통째로 달라지므로 렌더까지 확인해야 한다.
    """
    at = _run(flt_period=S.PERIOD_C)
    assert not at.exception
    assert not [e.value for e in at.error]
    assert at.info, "판매 데이터 부재 안내가 없습니다"


def test_period_b_full_year():
    """구간 B(2023~2024) — 요일 브릿지 경로."""
    at = _run(flt_period=S.PERIOD_B)
    assert not at.exception
    assert not [e.value for e in at.error]
    blob = " ".join(m.value for m in at.markdown)
    assert "요일 브릿지" in blob


def test_toggles_do_not_break():
    """컴프 제외 + 월초 제외 토글 동시 적용."""
    at = _run(flt_comp=True, flt_mfirst=True)
    assert not at.exception
    assert not [e.value for e in at.error]


def test_only_premium_tier():
    """프리미엄 티어만 남겨도 렌더되는지."""
    at = _run(flt_tiers=[S.TIER_PREMIUM])
    assert not at.exception
    assert not [e.value for e in at.error]
