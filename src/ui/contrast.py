"""WCAG 명암비 계산과 UI 크롬 색 조합 선언.

이 프로젝트는 **차트 팔레트**의 색약 안전성(ΔE)은 검증기를 돌려 통과시켰지만
(`theme.py` 상단 주석), **UI 크롬 텍스트**의 명암비는 오래 검증되지 않았다.
실측해 보니 구간 배지가 전부 미달이었다:

    badge-est / src-mock  1.90:1   ← 사실상 판독 불가
    badge-bridge          2.44:1
    badge-latest          2.74:1
    badge-real            3.70:1
    muted 텍스트 전반     3.41:1

배지는 이 프로젝트가 "어느 근거로 그린 값인지 항상 드러내는 신뢰 장치"라고
부르는 요소다. 안 읽히면 장치가 아니라 장식이 된다. 그래서 색을 고치는 데서
끝내지 않고 **검사 가능한 형태로 선언**해 회귀로 묶는다
(`tests/test_contrast.py`, `scripts/check_contrast.py`).

⚠ 글자 크기 주의. 배지·칩·caveat 는 0.68~0.76rem(≈11~12px)이라 WCAG 의
'큰 글자' 예외(3:1)에 해당하지 않는다 — 18.66px(굵은 글씨) / 24px 이상이어야
한다. 따라서 여기 있는 조합은 전부 **4.5:1** 기준으로 본다.

색 좌표 자체는 순수 계산이라 streamlit/plotly 없이 테스트된다.
"""

from __future__ import annotations

from typing import NamedTuple

from src.ui import theme

# WCAG 2.1 AA — 일반 텍스트. 이 프로젝트의 크롬 텍스트는 전부 작은 글씨라
# 'large text' 완화 기준(3.0)을 쓸 수 있는 요소가 없다.
AA_NORMAL = 4.5
AA_LARGE = 3.0

RGB = tuple[float, float, float]


def hex_to_rgb(value: str) -> RGB:
    h = value.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def composite(tint: RGB, alpha: float, base: RGB) -> RGB:
    """반투명 틴트를 불투명 지면 위에 합성한다.

    배지 배경은 `rgba(계열색, 0.12)` 형태라 실제 화면 색은 지면과 섞인 값이다.
    원색 그대로 명암비를 재면 실제보다 나쁘게 나와 과잉 대응하게 된다.
    """
    return tuple(tint[i] * alpha + base[i] * (1 - alpha) for i in range(3))  # type: ignore[return-value]


def relative_luminance(rgb: RGB) -> float:
    def channel(v: float) -> float:
        c = v / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str | RGB, bg: str | RGB) -> float:
    """WCAG 명암비 (1.0 ~ 21.0)."""
    a = relative_luminance(hex_to_rgb(fg) if isinstance(fg, str) else fg)
    b = relative_luminance(hex_to_rgb(bg) if isinstance(bg, str) else bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


class Pair(NamedTuple):
    """검사 대상 한 쌍. `tint`/`alpha` 가 있으면 지면 위에 합성해서 잰다."""
    name: str
    fg: str
    base: str
    tint: str | None = None
    alpha: float = 0.0

    @property
    def background(self) -> RGB:
        base = hex_to_rgb(self.base)
        if self.tint is None:
            return base
        return composite(hex_to_rgb(self.tint), self.alpha, base)

    @property
    def ratio(self) -> float:
        return contrast_ratio(self.fg, self.background)

    @property
    def passes(self) -> bool:
        return self.ratio >= AA_NORMAL


SURFACE = theme.INK["surface"]     # #fcfcfb — 카드·배지가 얹히는 면
PAGE = theme.INK["page"]           # #f9f9f7 — 지면
NAVY_900 = theme.NAVY["900"]       # 헤더 패널
NAVY_800 = theme.NAVY["800"]       # hero 패널


def ui_chrome_pairs() -> list[Pair]:
    """화면에 실제로 존재하는 조합만 선언한다.

    틴트·알파 값은 `assets/styles.css` 의 배지 규칙과 1:1 로 대응한다. 한쪽만
    바꾸면 `test_css_matches_theme_palette` 가 잡는다.
    """
    return [
        # ── 구간 배지 (0.72rem) ──────────────────────────────────────
        Pair("badge-real 실측 조인", theme.ON_TINT["blue"], SURFACE,
             theme.SERIES[0], 0.12),
        Pair("badge-bridge 요일 브릿지", theme.ON_TINT["aqua"], SURFACE,
             theme.SERIES[2], 0.12),
        Pair("badge-latest 최신 유입", theme.ON_TINT["orange"], SURFACE,
             theme.SERIES[1], 0.12),
        Pair("badge-est 추정", theme.ON_TINT["yellow"], SURFACE,
             theme.SERIES[3], 0.14),
        Pair("badge-neutral", theme.INK["secondary"], SURFACE,
             theme.INK["grid"], 1.0),
        # ── 출처 태그 (0.68rem) ──────────────────────────────────────
        Pair("src-real 실데이터", theme.INK["up"], SURFACE, "#0ca30c", 0.13),
        Pair("src-mock 대체 데이터", theme.ON_TINT["yellow"], SURFACE,
             theme.SERIES[3], 0.14),
        # ── 칩 (0.70rem) ────────────────────────────────────────────
        Pair("kl-chip 채널 칩", theme.ON_TINT["blue_soft"], SURFACE,
             theme.SERIES[0], 0.10),
        # ── 본문·후퇴 텍스트 ────────────────────────────────────────
        Pair("kl-caveat 한계 캡션 (0.74rem)", theme.INK["muted"], PAGE),
        Pair("kl-section-sub 섹션 부제 (0.82rem)", theme.INK["muted"], PAGE),
        Pair("kl-kpi-label KPI 라벨 (0.76rem)", theme.INK["muted"], SURFACE),
        Pair("kl-kpi-note KPI 각주 (0.70rem)", theme.INK["muted"], SURFACE),
        Pair("kl-card-evidence 근거 (0.73rem)", theme.INK["muted"], SURFACE),
        Pair("kl-insight li 인사이트 (0.84rem)", theme.INK["secondary"], SURFACE),
        Pair("kl-hero-sub 배너 부제 (0.86rem)", theme.INK["secondary"], SURFACE),
        Pair("kl-app-title 제목", theme.INK["primary"], PAGE),
        # ── 차트 텍스트 (plotly 템플릿) ──────────────────────────────
        Pair("차트 축 눈금 (size 10~11)", theme.INK["muted"], PAGE),
        Pair("차트 축 제목 (size 12)", theme.INK["secondary"], PAGE),
        Pair("KPI 상승 델타", theme.INK["up"], SURFACE),
        Pair("KPI 하락 델타", theme.INK["down"], SURFACE),

        # ── 네이비 크롬 (헤더 패널 · hero) ───────────────────────────
        # 밝은 지면용 잉크를 그대로 얹으면 전부 미달이므로 별도 팔레트를 쓴다.
        # 두 벌을 유지하는 이상 두 벌 다 검사해야 한 쪽만 흘러내리지 않는다.
        Pair("헤더 제목 (on navy-900)", theme.ON_NAVY["primary"], NAVY_900),
        Pair("헤더 메타 (on navy-900)", theme.ON_NAVY["secondary"], NAVY_900),
        Pair("헤더 각주 (on navy-900)", theme.ON_NAVY["muted"], NAVY_900),
        Pair("헤더 골드 수치 (on navy-900)", theme.GOLD["bright"], NAVY_900),
        Pair("hero 헤드라인 (on navy-800)", theme.ON_NAVY["primary"], NAVY_800),
        Pair("hero 부제 (on navy-800)", theme.ON_NAVY["secondary"], NAVY_800),
        Pair("hero 칩 라벨 (on navy-800)", theme.ON_NAVY["muted"], NAVY_800),
        Pair("hero 칩 수치 골드 (on navy-800)", theme.GOLD["bright"], NAVY_800),
        Pair("골드 액센트 (on navy-800)", theme.GOLD["base"], NAVY_800),
        Pair("골드 텍스트 (밝은 지면)", theme.GOLD["deep"], PAGE),

        # ── 네이비 위 구간 배지 (틴트 22%) ──────────────────────────
        Pair("배지 실측 조인 (on navy)", theme.ON_NAVY_TINT["blue"],
             NAVY_900, theme.SERIES[0], 0.22),
        Pair("배지 요일 브릿지 (on navy)", theme.ON_NAVY_TINT["aqua"],
             NAVY_900, theme.SERIES[2], 0.22),
        Pair("배지 최신 유입 (on navy)", theme.ON_NAVY_TINT["orange"],
             NAVY_900, theme.SERIES[1], 0.22),
        Pair("배지 추정 (on navy)", theme.ON_NAVY_TINT["yellow"],
             NAVY_900, theme.SERIES[3], 0.22),
        # hero 는 navy-800 이라 배지 배경이 한 단계 밝다 — 더 불리한 쪽도 본다
        Pair("배지 실측 조인 (on hero)", theme.ON_NAVY_TINT["blue"],
             NAVY_800, theme.SERIES[0], 0.22),

        # ── 사이드바 위젯 ───────────────────────────────────────────
        # 멀티셀렉트 태그는 Streamlit 이 그리는 요소라 우리 클래스가 아니지만,
        # 사이드바에 14개까지 뜨는 **텍스트**다. 실제로 여기서 흰 글자 on 골드
        # (2.24:1)로 한 번 미끄러졌기 때문에 선언해 두고 회귀로 잡는다.
        # 배경은 입력 컨테이너(navy-700) 위의 흰색 9% 틴트.
        Pair("멀티셀렉트 선택 태그", theme.ON_NAVY["primary"],
             theme.NAVY["700"], "#ffffff", 0.09),
        Pair("멀티셀렉트 태그 삭제 아이콘", theme.ON_NAVY["secondary"],
             theme.NAVY["700"], "#ffffff", 0.09),
        Pair("사이드바 라벨 (on navy-900)", theme.ON_NAVY["primary"], NAVY_900),
    ]


def report() -> str:
    """사람이 읽는 표. `scripts/check_contrast.py` 가 그대로 출력한다."""
    pairs = ui_chrome_pairs()
    width = max(len(p.name) for p in pairs) + 2
    lines = [
        f"{'요소':<{width}}{'대비':>7}  {'AA 4.5:1':>9}",
        "-" * (width + 20),
    ]
    for p in pairs:
        lines.append(
            f"{p.name:<{width}}{p.ratio:>6.2f}  "
            f"{'PASS' if p.passes else 'FAIL':>9}"
        )
    failed = [p for p in pairs if not p.passes]
    lines.append("-" * (width + 20))
    lines.append(
        f"{len(pairs) - len(failed)}/{len(pairs)} 통과"
        + (f" · 미달: {', '.join(p.name for p in failed)}" if failed else "")
    )
    return "\n".join(lines)
