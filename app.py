"""강원랜드 VIP 카지노-리조트 통합 교차판매 마케팅 대시보드.

엔트리포인트. 오케스트레이션만 담당한다 — 로드 → 필터 → ctx 조립 → 탭 렌더.
분석 로직은 src/analysis, 그림은 src/ui 에 있다.

로그인·초기 설정 관문이 없다: 접속 즉시 전체 대시보드가 조작 가능하다.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import settings as S
from src.data import api_client, loaders
from src.ui import components as C
from src.ui import sidebar, theme
from src.views import fnb_engine, local_curation, overview, timing

# ── 1. 페이지 설정 (반드시 최초 st.* 호출) ──────────────────────────────
st.set_page_config(
    page_title="강원랜드 VIP 교차판매 대시보드",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "about": "강원랜드 공공데이터 기반 VIP 카지노-리조트 교차판매 마케팅 대시보드"
    },
)


# ── 2. 전역 스타일 ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_css(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


if css := _load_css(str(S.CSS_PATH)):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

theme.register_plotly_template()


# ── 3. 데이터 로드 — 4계층 폴백. 어떤 실패도 앱을 죽이지 않는다 ──────────
@st.cache_data(ttl=S.CACHE_TTL, show_spinner="공공데이터 로딩 중...")
def load_all():
    """(데이터, 출처). 각 로더가 내부에서 폴백을 완결 처리한다."""
    return loaders.load_all_impl()


data, sources = load_all()

fallback_keys = [k for k, v in sources.items() if v not in S.REAL_SOURCES]
if fallback_keys:
    names = {"ars": "ARS 예약", "sales": "판매 3종",
             "demo": "성별·연령", "merchants": "가맹점"}
    st.warning(
        f"{', '.join(names.get(k, k) for k in fallback_keys)} 데이터를 데모용 "
        "데이터로 대체했습니다. `data/incoming/` 에 CSV 를 넣고 "
        "`python scripts/ingest.py` 를 실행하면 실데이터로 전환됩니다.",
        icon="⚠️",
    )

if errors := api_client.get_errors():
    with st.expander(f"오픈 API 안내 {len(errors)}건", expanded=False):
        for msg in errors:
            st.caption(f"• {msg}")
        st.caption(
            "API 호출이 실패해도 대시보드는 모킹/내장 데이터로 정상 동작합니다."
        )


# ── 4. 사이드바 필터 ───────────────────────────────────────────────────
fs = sidebar.render(data, sources)


# ── 5. 헤더 ────────────────────────────────────────────────────────────
left, right = st.columns([0.74, 0.26], vertical_alignment="bottom")
with left:
    st.markdown(
        '<div class="kl-app-title">🎰 VIP 카지노–리조트 교차판매 대시보드</div>',
        unsafe_allow_html=True,
    )
    seg = ", ".join(fs.age_bands) if fs.age_bands else "전체 연령"
    st.markdown(
        f'<div class="kl-app-sub">{C.period_badge(fs.period_badge)}'
        f'<span class="kl-app-meta">{fs.start:%Y-%m-%d} ~ {fs.end:%Y-%m-%d}'
        f' · {fs.n_days:,}일 · 세그먼트 {seg} / {fs.gender}</span></div>',
        unsafe_allow_html=True,
    )
with right:
    st.metric("분석 채널", f"{len(fs.channels)}개",
              f"{fs.n_days:,}일 커버리지", delta_color="off")

st.divider()


# ── 6. 탭 라우팅 (탭 단위 예외 격리) ───────────────────────────────────
ctx = {"data": data, "filters": fs, "sources": sources}

tabs = st.tabs([
    "📊 메인 대시보드",
    "🍽️ 고마진 F&B 매칭 엔진",
    "⏱️ 유입-소비 시차 엔진",
    "🎁 프리미엄 로컬 큐레이션",
])

for tab, view in zip(tabs, (overview, fnb_engine, timing, local_curation)):
    with tab:
        try:
            view.render(ctx)
        except Exception as exc:  # noqa: BLE001
            # 한 탭의 장애가 전체 앱을 흰 화면으로 만들지 않게 격리한다
            st.error(f"이 탭을 렌더링하는 중 문제가 발생했습니다: {exc}")
            st.caption("다른 탭은 정상 동작합니다. 필터를 변경해 다시 시도해 보세요.")
            with st.expander("기술 세부 정보", expanded=False):
                st.exception(exc)


# ── 7. 푸터 ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "출처: 공공데이터포털 (주)강원랜드 개방 데이터  ·  "
    "단가·주문시각 데이터가 제공되지 않아 고마진·시간대 지표는 프록시입니다  ·  "
    "상관관계는 인과관계를 의미하지 않습니다."
)
