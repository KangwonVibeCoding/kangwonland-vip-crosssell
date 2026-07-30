"""색상 팔레트와 plotly 공용 템플릿.

팔레트는 색약 안전성 검증기를 실제로 돌려 통과를 확인한 값이다:
  인접쌍  worst CVD ΔE 9.1 / normal-vision 19.6 (light) — PASS
  전체쌍  앞 3색만 PASS (CVD ΔE 9.2 / normal 24.0)

규칙 (전 차트 공통):
  · 카테고리 색은 고정 순서. 절대 순환시키지 않는다
  · 도넛·트리맵·산점도처럼 모든 쌍이 동시에 보이는 형태는 앞 3색까지만
  · 시퀀셜(크기·양) 인코딩은 단일 청색 램프. 무지개 금지
  · 발산 팔레트는 부호가 의미를 갖는 차트에만 (래그 상관 — D+2 가 음수다)
  · 이중 축(두 개의 y 스케일) 금지
  · 색상은 엔티티에 고정 — 필터로 계열이 줄어도 생존 계열 색이 변하지 않는다
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from config import settings as S

# ── 카테고리 (아이덴티티) ──────────────────────────────────────────────
SERIES: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# ── 시퀀셜 (양·크기) — 단일 청색 램프 ──────────────────────────────────
SEQ_BLUE: list[str] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#2a78d6", "#256abf", "#184f95", "#0d366b",
]
# plotly colorscale 형식 (0~1 위치, 색)
SEQ_BLUE_SCALE = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]

# ── 발산 (부호가 의미를 갖는 경우 전용) ─────────────────────────────────
DIVERGING_SCALE = [
    [0.0, "#d03b3b"], [0.35, "#ec835a"], [0.5, "#f0efec"],
    [0.65, "#6da7ec"], [1.0, "#184f95"],
]

# ── 상태 (예약 색 — 계열로 재사용하지 않는다) ───────────────────────────
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

# ── 잉크 / 크롬 ───────────────────────────────────────────────────────
INK = {
    "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
    "page": "#f9f9f7", "up": "#006300", "down": "#d03b3b",
    "border": "rgba(11,11,11,0.10)",
}

# ── 엔티티 고정 색상 매핑 ──────────────────────────────────────────────
CHANNEL_COLOR = {
    S.CH_CASINO: SERIES[0],
    S.CH_ROOM: SERIES[1],
    S.CH_LOCAL: SERIES[2],
}
CHANNEL_COLOR_BY_LABEL = {
    S.CHANNEL_LABEL[ch]: color for ch, color in CHANNEL_COLOR.items()
}
TIER_COLOR = {
    S.TIER_PREMIUM: SERIES[0],
    S.TIER_BUNDLE: SERIES[1],
    S.TIER_STANDARD: INK["muted"],
}
TIER_COLOR_BY_LABEL = {
    S.TIER_LABEL[t]: color for t, color in TIER_COLOR.items()
}
GENDER_COLOR = {"남": SERIES[0], "여": SERIES[1]}
SEGMENT_COLOR = {
    "VIP(40~60대)": SERIES[0],
    "준VIP(30대)": SERIES[1],
    "기타": SERIES[2],
}
# ARS 유입은 채널이 아니므로 4번째 색을 쓴다 (요일 프로파일 4계열 차트)
INFLOW_COLOR = SERIES[3]
SERIES_COLOR_BY_LABEL = {
    "카지노 유입(ARS)": INFLOW_COLOR,
    **CHANNEL_COLOR_BY_LABEL,
}

GRADE_COLOR = {
    "피크": STATUS["critical"], "성수": STATUS["serious"],
    "보통": STATUS["warning"], "한산": INK["muted"],
}

TEMPLATE_NAME = "kangwonland"
FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", "Malgun Gothic", sans-serif'


def register_plotly_template() -> None:
    """공용 plotly 템플릿을 기본값으로 등록한다.

    격자·축은 후퇴색으로, 텍스트는 잉크색으로 고정한다. 계열색으로 글씨를
    쓰지 않는 규칙을 템플릿 차원에서 지킨다.
    """
    axis = dict(
        showgrid=True, gridcolor=INK["grid"], gridwidth=1,
        zeroline=False, linecolor=INK["axis"], linewidth=1,
        ticks="outside", tickcolor=INK["axis"], ticklen=4,
        tickfont=dict(color=INK["muted"], size=11),
        title=dict(font=dict(color=INK["secondary"], size=12)),
    )
    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_FAMILY, color=INK["primary"], size=13),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            colorway=SERIES,
            xaxis=axis,
            yaxis=axis,
            margin=dict(l=8, r=8, t=48, b=8),
            title=dict(font=dict(size=15, color=INK["primary"]), x=0, xanchor="left"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=11, color=INK["secondary"]),
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor=INK["surface"], bordercolor=INK["border"],
                font=dict(family=FONT_FAMILY, size=12, color=INK["primary"]),
            ),
            hovermode="x unified",
            colorscale=dict(sequential=SEQ_BLUE_SCALE, diverging=DIVERGING_SCALE),
        )
    )
    pio.templates[TEMPLATE_NAME] = template
    pio.templates.default = TEMPLATE_NAME


def grade_color(grade: str) -> str:
    return GRADE_COLOR.get(str(grade), INK["muted"])


def delta_color(value: float) -> str:
    """증감 색. 상승=성공 텍스트색, 하락=경고색."""
    return INK["up"] if value >= 0 else INK["down"]
