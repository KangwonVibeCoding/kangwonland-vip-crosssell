"""재사용 UI 조각 — KPI 카드, 섹션 헤더, 구간 배지, 표 뷰, 인사이트 박스."""

from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

from config import settings as S
from src.ui import theme


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def period_badge(badge: str, detail: str = "") -> str:
    """구간 배지 HTML. 어느 근거로 그린 값인지 항상 화면에 드러내기 위한 장치."""
    tone = {
        "실측 조인": "badge-real",
        "요일 브릿지": "badge-bridge",
        "최신 유입": "badge-latest",
        "추정": "badge-est",
    }.get(badge, "badge-neutral")
    suffix = f'<span class="badge-detail">{_esc(detail)}</span>' if detail else ""
    return f'<span class="kl-badge {tone}">{_esc(badge)}</span>{suffix}'


def section_header(title: str, subtitle: str = "", badge: str = "",
                   badge_detail: str = "") -> None:
    bits = [f'<div class="kl-section">',
            f'<div class="kl-section-title">{_esc(title)}']
    if badge:
        bits.append(period_badge(badge, badge_detail))
    bits.append("</div>")
    if subtitle:
        bits.append(f'<div class="kl-section-sub">{_esc(subtitle)}</div>')
    bits.append("</div>")
    st.markdown("".join(bits), unsafe_allow_html=True)


def app_header(title: str, badge: str, meta: str,
               stat_label: str = "", stat_value: str = "",
               stat_note: str = "") -> None:
    """상단 헤더 패널 — 제목·구간 배지·메타·요약 지표를 **한 덩어리**로 그린다.

    st.columns 로 좌우를 나누면 두 개의 독립 DOM 이라 하나의 패널 배경을 씌울 수
    없다(컬럼 사이 간격으로 배경이 갈라진다). 그래서 마크업 하나로 내보내고
    배치는 CSS flex 에 맡긴다 — 좁은 화면에서 자동으로 위아래로 쌓이는 것도
    여기서 온다.

    app.py 는 오케스트레이션만 하는 자리이므로 마크업은 이 층에 둔다.
    """
    stat_html = ""
    if stat_value:
        note = (f'<span class="kl-header-stat-note">{_esc(stat_note)}</span>'
                if stat_note else "")
        stat_html = (
            '<div class="kl-header-stat">'
            f'<span class="kl-header-stat-label">{_esc(stat_label)}</span>'
            f'<span class="kl-header-stat-value">{_esc(stat_value)}</span>'
            f'{note}</div>'
        )
    st.markdown(
        '<div class="kl-header">'
        '<div class="kl-header-main">'
        f'<div class="kl-app-title">{_esc(title)}</div>'
        f'<div class="kl-app-sub">{period_badge(badge)}'
        f'<span class="kl-app-meta">{_esc(meta)}</span></div>'
        '</div>'
        f'{stat_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str = "", delta_dir: int = 0,
             note: str = "", accent: str | None = None,
             compact: bool = False) -> None:
    """KPI 카드. delta_dir: 1 상승 / -1 하락 / 0 중립.

    `compact` 는 **행에 속하지 않은 단독 카드**용이다. 기본 카드는 delta 가
    없어도 `&nbsp;` 로 그 줄을 남긴다 — 5장이 나란히 선 KPI 행에서 값의 세로
    위치가 카드마다 어긋나지 않게 하려는 장치다. 하지만 헤더처럼 카드가 하나뿐인
    자리에서는 그 빈 줄이 카드를 이유 없이 키워 제목과의 정렬을 흐트러뜨린다.
    """
    color = theme.INK["up"] if delta_dir > 0 else (
        theme.INK["down"] if delta_dir < 0 else theme.INK["muted"]
    )
    arrow = "▲" if delta_dir > 0 else ("▼" if delta_dir < 0 else "")
    # 기본 액센트는 골드다. SERIES[0](파랑)은 **데이터색**이라 크롬에 쓰면
    # 읽는 사람이 "이 파랑이 계열인가 장식인가"를 매번 판단해야 한다.
    # 액센트가 실제로 무언가를 구분할 때만 호출부가 accent 를 넘긴다.
    accent = accent or theme.GOLD["base"]
    if delta:
        delta_html = (
            f'<div class="kl-kpi-delta" style="color:{color}">'
            f'{arrow} {_esc(delta)}</div>'
        )
    else:
        # 단독 카드는 자리막이를 넣지 않는다 (위 docstring 참조)
        delta_html = "" if compact else '<div class="kl-kpi-delta">&nbsp;</div>'
    note_html = f'<div class="kl-kpi-note">{_esc(note)}</div>' if note else ""
    cls = "kl-kpi kl-kpi-compact" if compact else "kl-kpi"
    st.markdown(
        f'<div class="{cls}" style="--accent:{accent}">'
        f'<div class="kl-kpi-label">{_esc(label)}</div>'
        f'<div class="kl-kpi-value">{_esc(value)}</div>'
        f'{delta_html}{note_html}</div>',
        unsafe_allow_html=True,
    )


def hero_banner(headline: str, sub: str, stats_pairs: Iterable[tuple[str, str]],
                badge: str = "", badge_detail: str = "") -> None:
    """가설 검증 배너 — 대시보드 최상단의 근거 제시."""
    chips = "".join(
        f'<div class="kl-hero-stat"><span class="kl-hero-stat-label">{_esc(k)}</span>'
        f'<span class="kl-hero-stat-value">{_esc(v)}</span></div>'
        for k, v in stats_pairs
    )
    badge_html = period_badge(badge, badge_detail) if badge else ""
    st.markdown(
        f'<div class="kl-hero">'
        f'<div class="kl-hero-top">{badge_html}</div>'
        f'<div class="kl-hero-headline">{_esc(headline)}</div>'
        f'<div class="kl-hero-sub">{_esc(sub)}</div>'
        f'<div class="kl-hero-stats">{chips}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def insight_box(lines: Iterable[str], title: str = "읽어낸 것",
                tone: str = "info") -> None:
    items = "".join(f"<li>{_esc(t)}</li>" for t in lines if t)
    if not items:
        return
    st.markdown(
        f'<div class="kl-insight kl-insight-{_esc(tone)}">'
        f'<div class="kl-insight-title">{_esc(title)}</div>'
        f'<ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def caveat(text: str) -> None:
    """분석 한계 캡션. 프록시 지표에는 항상 붙인다."""
    st.markdown(f'<div class="kl-caveat">{_esc(text)}</div>', unsafe_allow_html=True)


def prescription_card(title: str, targets: list[str], detail: str,
                      evidence: str) -> None:
    chips = "".join(f'<span class="kl-chip">{_esc(t)}</span>' for t in targets)
    st.markdown(
        f'<div class="kl-card">'
        f'<div class="kl-card-title">{_esc(title)}</div>'
        f'<div class="kl-card-chips">{chips}</div>'
        f'<div class="kl-card-body">{_esc(detail)}</div>'
        f'<div class="kl-card-evidence">{_esc(evidence)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def bundle_card(bundle: dict) -> None:
    st.markdown(
        f'<div class="kl-card">'
        f'<div class="kl-card-title">번들 {bundle["rank"]}</div>'
        f'<div class="kl-bundle-row"><span class="kl-chip">'
        f'{_esc(S.CHANNEL_LABEL[S.CH_ROOM])}</span> {_esc(bundle["room_item"])}</div>'
        f'<div class="kl-bundle-row"><span class="kl-chip">'
        f'{_esc(S.CHANNEL_LABEL[S.CH_LOCAL])}</span> {_esc(bundle["local_item"])}</div>'
        f'<div class="kl-card-evidence">'
        f'합산 CSM {bundle["combined_csm"]:.1f} · '
        f'동시 상승일 {bundle["both_up_days"]}/{bundle["co_days"]}일 · '
        f'{_esc(bundle["room_tier"])} × {_esc(bundle["local_tier"])}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """중복 컬럼명에 접미사를 붙인다.

    st.dataframe 은 arrow 변환 단계에서 중복 컬럼명을 만나면 예외를 던진다.
    한국어 컬럼명은 충돌하기 쉬워서('월'이 month 이자 Monday) 마지막 방어선을 둔다.
    """
    cols = [str(c) for c in df.columns]
    if len(set(cols)) == len(cols):
        return df
    seen: dict[str, int] = {}
    renamed: list[str] = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            renamed.append(f"{col} ({seen[col]})")
        else:
            seen[col] = 0
            renamed.append(col)
    out = df.copy()
    out.columns = renamed
    return out


def table_view(df: pd.DataFrame, label: str = "데이터 표로 보기",
               *, column_config: dict | None = None,
               hide_index: bool = True) -> None:
    """모든 차트에 붙는 접근성 표 뷰.

    색상만으로 정보를 전달하지 않기 위한 장치이며, 저대비 색을 쓴 차트에는
    특히 필수다.
    """
    if df is None or df.empty:
        return
    with st.expander(label, expanded=False):
        st.dataframe(_dedupe_columns(df), width="stretch", hide_index=hide_index,
                     column_config=column_config or {})


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV 바이트열. 인코딩은 **utf-8-sig** 로 고정한다.

    순수 utf-8 로 내보내면 한국어 Windows 의 Excel 이 기본 CP949 로 해석해
    컬럼명이 전부 깨진다 — 받는 사람이 마케터라면 그건 못 쓰는 파일이다.
    BOM 3바이트가 그 문제를 없앤다. 스트림릿 런타임 없이도 검증할 수 있도록
    위젯에서 분리해 둔다.
    """
    return df.to_csv(index=False).encode("utf-8-sig")


def download_csv(df: pd.DataFrame, label: str, filename: str, *,
                 key: str | None = None, help_text: str = "") -> None:
    """집행 대상 표를 파일로 꺼낸다.

    화면에서만 볼 수 있는 캠페인 캘린더는 실행 도구가 아니라 발표 슬라이드다.
    받는 사람(마케터)이 그대로 열어 일정에 옮길 수 있어야 제안이 집행이 된다.
    화면에 보이는 한글 컬럼명 그대로(=필터가 적용된 사본) 내보낸다.
    """
    if df is None or df.empty:
        return
    st.download_button(
        label,
        data=to_csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        key=key,
        help=help_text or None,
    )


def next_tab_card(tab: str, headline: str, detail: str, evidence: str = "") -> None:
    """다른 탭의 결론을 한 장으로 미리 보여준다.

    탭 4개 중 첫 화면에 있는 것은 탭1 의 결론뿐이다. 심사·발표에서 3분 만에
    화면을 떠나는 사람은 이 프로젝트의 핵심 발견(특산품 D+1)을 못 보고 나간다.
    Streamlit 에서 코드로 탭을 전환할 수는 없으므로 **문장으로 유도**한다.

    문구는 고정값이 아니라 지금 필터에서 다시 계산한 값이다 — 여기만 하드코딩하면
    필터를 바꿨을 때 첫 화면이 사실이 아닌 말을 하게 된다.
    """
    st.markdown(
        f'<div class="kl-card">'
        f'<div class="kl-card-chips"><span class="kl-chip">{_esc(tab)}</span></div>'
        f'<div class="kl-card-title">{_esc(headline)}</div>'
        f'<div class="kl-card-body">{_esc(detail)}</div>'
        + (f'<div class="kl-card-evidence">{_esc(evidence)}</div>' if evidence else "")
        + '</div>',
        unsafe_allow_html=True,
    )


def source_badges(sources: dict[str, str]) -> None:
    """데이터 출처 배지 — 데모 중 신뢰도 확보 + 개발 중 디버깅."""
    names = {"ars": "ARS 예약", "sales": "판매 3종",
             "demo": "성별·연령", "merchants": "가맹점"}
    for key, label in names.items():
        src = sources.get(key)
        if not src:
            continue
        text, detail = S.SOURCE_LABEL.get(src, (src, ""))
        cls = "src-real" if src in S.REAL_SOURCES else "src-mock"
        st.markdown(
            f'<div class="kl-src"><span class="kl-src-name">{_esc(label)}</span>'
            f'<span class="kl-src-tag {cls}" title="{_esc(detail)}">{_esc(text)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def empty_state(message: str, hint: str = "") -> None:
    """빈 데이터 방어 — 에러 대신 안내를 띄운다."""
    st.info(message, icon="ℹ️")
    if hint:
        st.caption(hint)


def fmt_int(value: object, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{int(round(float(value))):,}{suffix}"
    except (TypeError, ValueError):
        return "—"


def fmt_float(value: object, digits: int = 1, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "—"


def fmt_pct(value: object, digits: int = 1) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:,.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def josa(word: str, pair: str = "은는") -> str:
    """받침 유무에 맞는 조사를 고른다. `pair` 는 (받침 있음, 없음) 순서.

    문장을 데이터에서 생성하면 채널 이름이 그때그때 바뀐다 — '지역특산품는'
    같은 문장이 발표 화면에 그대로 뜨는 것을 막는 최소 장치다.
    """
    if not word:
        return pair[1]
    ch = word.strip()[-1:]
    if not ch or not ("가" <= ch <= "힣"):
        return pair[1]           # 숫자·영문 끝은 판정하지 않는다
    return pair[0] if (ord(ch) - 0xAC00) % 28 else pair[1]


def fmt_p(value: object) -> str:
    """p 값 표기. 아주 작은 값을 0 으로 반올림해 보여주지 않는다."""
    try:
        if value is None or pd.isna(value):
            return "—"
        p = float(value)
    except (TypeError, ValueError):
        return "—"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def pct_delta(current: float | None, previous: float | None) -> tuple[str, int]:
    """전기간 대비 증감률 문자열과 방향."""
    try:
        if current is None or previous is None or pd.isna(current) or pd.isna(previous):
            return "", 0
        if not previous:
            return "", 0
        change = (float(current) - float(previous)) / abs(float(previous))
        return f"{abs(change) * 100:,.1f}% vs 이전 구간", (1 if change > 0 else -1 if change < 0 else 0)
    except (TypeError, ValueError, ZeroDivisionError):
        return "", 0


def point_delta(current: float | None, previous: float | None,
                digits: int = 1, unit: str = "p") -> tuple[str, int]:
    """포인트 차이(절대 차) 문자열과 방향."""
    try:
        if current is None or previous is None or pd.isna(current) or pd.isna(previous):
            return "", 0
        diff = float(current) - float(previous)
        return f"{abs(diff):,.{digits}f}{unit} vs 이전 구간", (
            1 if diff > 0 else -1 if diff < 0 else 0
        )
    except (TypeError, ValueError):
        return "", 0
