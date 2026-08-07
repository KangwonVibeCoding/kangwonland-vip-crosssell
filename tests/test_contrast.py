"""UI 크롬 명암비 회귀 테스트.

차트 팔레트의 색약 안전성은 검증기를 돌려 통과시켰지만, 배지·캡션 같은 **UI
크롬 텍스트의 명암비**는 오래 검증되지 않아 실제로 미달 상태였다
(badge-est 1.90:1, muted 텍스트 3.41:1). 고친 뒤 다시 흘러내리지 않도록 여기서
고정한다.

이 파일은 실데이터를 요구하지 않는다 — 순수 색 계산이라 어떤 환경에서도 돈다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from config import settings as S
from src.ui import contrast, theme


def test_every_ui_chrome_pair_meets_aa():
    """화면에 존재하는 모든 텍스트 조합이 4.5:1 이상."""
    failed = [
        f"{p.name}: {p.ratio:.2f}:1"
        for p in contrast.ui_chrome_pairs() if not p.passes
    ]
    assert not failed, (
        "WCAG AA(4.5:1) 미달 조합:\n  " + "\n  ".join(failed)
        + "\n\nscripts/check_contrast.py 로 전체 표를 확인하세요."
    )


def test_badges_are_not_large_text_exempt():
    """배지 글자 크기가 'large text' 완화 기준에 들어가지 않는지 확인한다.

    18.66px(굵은 글씨) 이상이면 3:1 로 완화되는데, 그 사실을 근거로 기준을
    낮추려면 실제 글자가 그만큼 커야 한다. 지금은 0.72rem(≈11.5px)이다 —
    나중에 누가 완화 기준을 적용하려 할 때 이 테스트가 근거를 확인시킨다.
    """
    css = S.CSS_PATH.read_text(encoding="utf-8")
    block = re.search(r"\.kl-badge\s*\{(.*?)\}", css, re.S)
    assert block, ".kl-badge 규칙을 찾지 못했습니다"
    size = re.search(r"font-size:\s*([\d.]+)rem", block.group(1))
    assert size, ".kl-badge 에 font-size 가 없습니다"
    px = float(size.group(1)) * 16
    assert px < 18.66, (
        f"배지 글자가 {px:.1f}px 로 커졌습니다. 'large text' 3:1 완화를 쓸 수 "
        "있게 됐다면 contrast.py 의 기준을 의도적으로 바꾸세요."
    )


@pytest.mark.parametrize("var_name,expected", [
    ("--kl-muted", theme.INK["muted"]),
    ("--kl-ink", theme.INK["primary"]),
    ("--kl-ink-2", theme.INK["secondary"]),
    ("--kl-surface", theme.INK["surface"]),
    ("--kl-page", theme.INK["page"]),
    ("--kl-grid", theme.INK["grid"]),
    ("--kl-axis", theme.INK["axis"]),
    ("--kl-up", theme.INK["up"]),
    ("--kl-down", theme.INK["down"]),
    ("--kl-blue", theme.SERIES[0]),
    ("--kl-orange", theme.SERIES[1]),
    ("--kl-aqua", theme.SERIES[2]),
    ("--kl-yellow", theme.SERIES[3]),
    ("--kl-on-blue", theme.ON_TINT["blue"]),
    ("--kl-on-aqua", theme.ON_TINT["aqua"]),
    ("--kl-on-orange", theme.ON_TINT["orange"]),
    ("--kl-on-yellow", theme.ON_TINT["yellow"]),
    ("--kl-on-blue-soft", theme.ON_TINT["blue_soft"]),
    ("--kl-navy-900", theme.NAVY["900"]),
    ("--kl-navy-800", theme.NAVY["800"]),
    ("--kl-navy-700", theme.NAVY["700"]),
    ("--kl-navy-600", theme.NAVY["600"]),
    ("--kl-gold", theme.GOLD["base"]),
    ("--kl-gold-bright", theme.GOLD["bright"]),
    ("--kl-gold-deep", theme.GOLD["deep"]),
    ("--kl-on-navy", theme.ON_NAVY["primary"]),
    ("--kl-on-navy-2", theme.ON_NAVY["secondary"]),
    ("--kl-on-navy-3", theme.ON_NAVY["muted"]),
])
def test_css_matches_theme_palette(var_name, expected):
    """CSS 변수와 theme.py 가 어긋나지 않게 고정한다.

    두 곳에 같은 색을 적어 두는 구조라 한쪽만 고치는 사고가 난다. 그러면
    차트와 카드가 서로 다른 회색을 쓰게 되는데, 화면에서는 '왠지 안 맞는다'로만
    보여서 발견이 늦다. 명암비 테스트가 통과해도 이건 못 잡으므로 따로 둔다.
    """
    css = S.CSS_PATH.read_text(encoding="utf-8")
    found = re.search(rf"{re.escape(var_name)}\s*:\s*([^;]+);", css)
    assert found, f"styles.css 에 {var_name} 이 없습니다"
    assert found.group(1).strip().lower() == expected.lower(), (
        f"{var_name} 이 theme.py 와 다릅니다: "
        f"CSS={found.group(1).strip()} vs theme={expected}"
    )
