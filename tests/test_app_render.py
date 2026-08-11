"""app.py 실제 렌더 검증 — Streamlit AppTest 로 스크립트를 통째로 실행한다.

브라우저 없이 위젯 생성·뷰 렌더·예외를 전부 잡는다. `app.py` 는 탭 단위로 예외를
격리하므로, 뷰가 터져도 앱은 살아남는다 — 그래서 "예외 없음" 뿐 아니라
**st.error 가 하나도 없음**까지 확인해야 실제로 정상 렌더된 것이다.
"""

from __future__ import annotations

import datetime as dt

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


def test_all_tabs_present(app):
    """탭 5개 — 2026-08-11 에 'AI VIP 큐레이션'이 붙었다."""
    assert len(app.tabs) == 5, f"탭 5개가 아님: {len(app.tabs)}"


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
    assert "r = 0.80" in blob, "가설 검증 배너의 상관계수가 실측값이 아닙니다"


def test_headline_shows_confound_controlled_value(app):
    """배너가 원 상관만 팔지 않는지 — 요일 통제 후 값이 같은 줄에 있어야 한다.

    배너에서 0.80 만 보고 스크롤하지 않은 사람이 반쪽짜리 근거를 가져가지 않도록,
    통제 후 값을 같은 줄에 띄운다.

    ⚠ 기대값은 **기본 창(구간 B · 731일)** 의 값이다. 2026-08-11 에 사이드바가
    `DEFAULT_PERIOD` 를 무시하고 index=0(구간 A · 31일 창)을 고르던 버그를 고치면서
    바뀌었다 — 31일 창에서는 0.801→0.370, 룸서비스가 편상관 1위(0.558)라 이 테스트가
    그 값으로 통과하고 있었다. 두 창이 다른 답을 낸다는 사실 자체가 회귀 대상이므로
    **어느 창의 값인지 확인하지 않고 기대값만 고치지 말 것.**
    """
    blob = " ".join(m.value for m in app.markdown)
    if "데이터가 겹치지 않습니다" in blob:
        pytest.skip("기본 구간에 조인 가능한 데이터가 없음")
    assert "요일을 통제하면 0.62" in blob, "배너에 요일 통제 후 상관이 없습니다"
    assert "요일 교란 통제" in blob, "요일 교란 통제 절이 렌더되지 않았습니다"
    # 편상관 1위(731일 창에서는 카지노 식음)가 배너 칩 맨 앞 = 통제 후 우선순위가
    # 화면에 드러난다. 31일 창에서는 룸서비스(0.740 → 0.558)가 1위였다.
    assert "0.801 → 0.617" in blob, "채널별 원 상관 → 편상관 병기가 없습니다"


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


def test_sales_free_range_is_handled_gracefully():
    """판매가 없는 범위를 직접 지정해도 안내로 graceful 처리.

    구간 C(2025~2026 상반기)는 2026-08-11 에 제거했다 — 그 구간 판매 데이터를
    확보할 수 없어 탭 2·3·4·5 가 늘 안내 화면이었고 심사자가 고장으로 읽었다.
    다만 '직접 지정'으로는 여전히 판매 없는 범위를 고를 수 있으므로, 그 경로가
    빈 화면이 아니라 안내로 끝나는지는 계속 고정한다.
    """
    at = _run(flt_period=S.PERIOD_CUSTOM,
              flt_dates=(dt.date(2025, 1, 1), dt.date(2025, 3, 31)))
    assert not at.exception
    assert not [e.value for e in at.error]
    assert at.info or at.warning, "판매 데이터 부재 안내가 없습니다"


def test_period_b_full_year():
    """구간 B(2023~2024, 731일) — 기본 창이자 ARS 실측 조인 경로.

    2026-08-11 통합본 이전에는 판매만 있어 '요일 브릿지'(CAI 프록시) 배지가
    떴다. 이제 ARS 가 731일 전부 붙으므로 '실측 조인' 이어야 한다 — 프록시
    배지가 다시 뜨면 ARS 파일이 빠졌다는 뜻이다.
    """
    at = _run(flt_period=S.PERIOD_B)
    assert not at.exception
    assert not [e.value for e in at.error]
    blob = " ".join(m.value for m in at.markdown)
    assert "실측 조인" in blob
    assert "요일 브릿지" not in blob


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
