"""CSS 선택자 건전성 회귀 테스트.

`assets/styles.css` 에는 Streamlit 내부 DOM 을 겨냥한 규칙이 몇 개 있다. 그런
규칙은 **틀리면 에러가 아니라 침묵**이다 — 매치가 0 이어도 CSS 는 조용히 넘어가고,
화면은 "왜인지 스타일이 안 먹네" 상태로 남는다. 실제로 그런 일이 있었다:

  .stTabs [data-baseweb="tab"]        Streamlit 1.60 에서 탭이 baseweb →
  .stTabs [data-baseweb="tab-list"]   react-aria 로 바뀌며 매치 0
  [data-testid=stDataFrame] thead th  st.dataframe 은 캔버스 그리드라 thead 없음

셋 다 규칙은 파일에 있었지만 화면에 적용된 적이 없었다(실측 계산값: 탭
font-weight 400 = 브라우저 기본). 브라우저를 띄우지 않고 잡을 수 있는 형태로
고정해 둔다 — playwright 를 개발 의존성에 넣으면 pytest 가 무거워진다.

전체 DOM 대조가 필요하면 브라우저로 직접 확인한다(선택자별 매치 수를 세면 된다).
"""

from __future__ import annotations

import re

from config import settings as S

CSS = S.CSS_PATH.read_text(encoding="utf-8")
CSS_NO_COMMENTS = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)


def test_no_emotion_class_selectors():
    """`.st-emotion-cache-*` 는 빌드마다 바뀐다 — 절대 겨냥하지 않는다."""
    hits = [ln.strip() for ln in CSS_NO_COMMENTS.splitlines()
            if "st-emotion-cache" in ln]
    assert not hits, (
        "emotion 클래스 선택자가 들어왔습니다. 빌드마다 해시가 바뀌어 다음 배포에서 "
        "조용히 죽습니다 — data-testid 나 role 을 쓰세요:\n  " + "\n  ".join(hits)
    )


# Streamlit 1.60 에서 **실제로 존재함을 브라우저로 확인한** baseweb 선택자.
# baseweb 속성은 role/data-testid 보다 수명이 짧다(탭은 1.60 에서 react-aria 로
# 갈아타며 사라졌다). 그래서 전면 금지 대신 **확인된 것만 허용**한다.
# ⚠ Streamlit 을 올릴 때는 이 목록을 브라우저로 다시 확인할 것.
VERIFIED_BASEWEB = {
    "tag",       # 멀티셀렉트 선택 태그 (1.60 에서 14개 매치 확인)
}
# 1.60 에서 매치 0 인 것으로 확인된 선택자 — 되돌아오면 안 된다
DEAD_BASEWEB = {"tab", "tab-list"}


def test_baseweb_selectors_are_verified_ones_only():
    used = set(re.findall(r'data-baseweb="([^"]+)"', CSS_NO_COMMENTS))
    revived = used & DEAD_BASEWEB
    assert not revived, (
        f"1.60 에서 매치 0 인 baseweb 선택자가 다시 들어왔습니다: {sorted(revived)}. "
        "탭은 react-aria 로 바뀌었으니 `[role=\"tab\"]` 을 쓰세요."
    )
    unknown = used - VERIFIED_BASEWEB
    assert not unknown, (
        f"확인되지 않은 baseweb 선택자입니다: {sorted(unknown)}. "
        "브라우저로 매치 수를 세어 본 뒤 VERIFIED_BASEWEB 에 추가하세요 — "
        "틀린 선택자는 에러가 아니라 침묵이라 화면에서 발견이 늦습니다."
    )


def test_dataframe_header_is_styled_via_config_not_css():
    """캔버스 그리드에는 `thead` 가 없다 — CSS 로 되돌리지 못하게 막는다."""
    assert "thead" not in CSS_NO_COMMENTS, (
        "st.dataframe 은 HTML 표가 아니라 캔버스 기반 그리드라 thead 셀렉터가 "
        "닿지 않습니다. .streamlit/config.toml 의 "
        "dataframeHeaderBackgroundColor 를 쓰세요."
    )

    config = (S.CSS_PATH.parent.parent / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )
    assert "dataframeHeaderBackgroundColor" in config, (
        "표 헤더 후퇴 설정이 config.toml 에서 사라졌습니다."
    )


def test_tab_rules_target_role():
    """탭 스타일이 실제로 매치되는 선택자를 쓰는지 확인한다."""
    assert '.stTabs [role="tablist"]' in CSS_NO_COMMENTS
    assert '.stTabs [role="tab"]' in CSS_NO_COMMENTS


def test_download_button_uses_deep_gold_for_text():
    """내보내기 버튼은 이 대시보드의 유일한 액션이라 골드로 세웠다.

    ⚠ 기본 골드(#cba95c)를 **글자색**에 쓰면 밝은 지면에서 1.9:1 이라 못 읽는다.
    테두리(비텍스트)에는 기본 골드, 글자에는 gold-deep 이라는 구분이 이 규칙의
    전부이므로 뒤집히지 않게 고정한다. 명암비 자체는 contrast.py 가 잰다.
    """
    block = re.search(
        r'\[data-testid="stDownloadButton"\] button \{(.*?)\}',
        CSS_NO_COMMENTS, flags=re.S,
    )
    assert block, "내보내기 버튼 규칙이 사라졌습니다 — 액션이 중립 위젯으로 돌아갑니다."
    body = block.group(1)
    assert "color: var(--kl-gold-deep)" in body
    assert "color: var(--kl-gold)" not in body, (
        "글자색이 기본 골드입니다 — 밝은 지면에서 1.9:1 이라 AA 미달입니다."
    )


def test_kpi_accent_is_not_a_series_color():
    """KPI 좌측 액센트는 아무것도 인코딩하지 않는다 — 데이터색을 쓰면 안 된다.

    9장이 전부 같은 파랑(SERIES[0])이라 정보를 담은 척하는 장식이었다. 크롬은
    골드, 데이터는 SERIES 라는 색 언어를 여기서 깨면 읽는 사람이 "이 파랑이
    계열인가 장식인가"를 매번 판단해야 한다.
    """
    from src.ui import components, theme

    block = re.search(r"\.kl-kpi::before \{(.*?)\}", CSS_NO_COMMENTS, flags=re.S)
    assert block, "KPI 액센트 규칙이 사라졌습니다."
    assert "var(--accent, var(--kl-gold))" in block.group(1), (
        "액센트 폴백이 데이터색으로 돌아갔습니다."
    )

    import inspect

    src = inspect.getsource(components.kpi_card)
    assert "theme.SERIES" not in src, (
        "kpi_card 의 기본 액센트가 SERIES(데이터색)입니다 — GOLD 를 쓰세요."
    )
    assert theme.GOLD["base"] in src or 'GOLD["base"]' in src


def test_section_rhythm_is_top_heavy():
    """섹션 제목은 제 내용에 붙고, 섹션끼리는 떨어져야 한다(근접성).

    위아래가 대칭이면 제목이 두 블록 사이에 떠서 어느 쪽 제목인지 알 수 없다.
    그 상태를 뷰마다 `st.markdown("")` 로 메우고 있었고, 탭마다 리듬이 달라졌다.
    """
    block = re.search(r"\.kl-section \{(.*?)\}", CSS_NO_COMMENTS, flags=re.S)
    assert block
    margin = re.search(r"margin:\s*([\d.]+)rem\s+0\s+([\d.]+)rem", block.group(1))
    assert margin, "섹션 여백이 `상 0 하` 형태가 아닙니다."
    top, bottom = float(margin.group(1)), float(margin.group(2))
    assert top >= bottom * 2, (
        f"섹션 위 여백({top}rem)이 아래({bottom}rem)의 2배 미만입니다 — 제목이 "
        "제 내용에 붙지 않아 위계가 사라집니다."
    )


def test_views_do_not_hand_space_before_sections():
    """간격은 CSS 한 곳에서 정한다 — 뷰에 자리막이를 다시 넣지 못하게 막는다."""
    for path in sorted((S.CSS_PATH.parent.parent / "src" / "views").glob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.strip() != 'st.markdown("")':
                continue
            following = "\n".join(lines[i + 1:i + 4])
            assert "C.section_header(" not in following, (
                f"{path.name}:{i + 1} 에 섹션 앞 자리막이가 있습니다. "
                ".kl-section 의 margin 이 이미 간격을 만듭니다 — 두 벌로 나뉘면 "
                "탭마다 리듬이 달라집니다."
            )


def test_focus_visible_exists():
    """키보드 포커스 표시는 접근성 최소선이다 — 지우지 못하게 막는다."""
    assert ":focus-visible" in CSS_NO_COMMENTS, (
        "포커스 표시가 사라졌습니다. 사이드바 위젯이 12개라 키보드 조작이 "
        "불가능해집니다."
    )


def test_reduced_motion_block_exists():
    assert "prefers-reduced-motion" in CSS_NO_COMMENTS


def test_no_dark_mode_block():
    """라이트 단일 테마 — config.toml 이 base='light' 로 고정돼 있다.

    CSS 에만 다크 대응을 넣으면 OS 가 다크인 사용자에게 '밝은 페이지 위 검은 카드'
    가 뜬다. 실제로 그 블록이 있었고 지웠다(파일 상단 주석 참조).
    """
    assert "prefers-color-scheme" not in CSS_NO_COMMENTS, (
        "다크 대응은 config.toml · plotly 템플릿 · CSS 를 한꺼번에 바꿔야 합니다."
    )
