"""탭 2 — 고마진 F&B 매칭 엔진.

요일×월 수요 히트맵 → VIP 상품 Top → 영업장 트리맵 → CSM 테이블 → 번들 추천.
"어떤 상품을 언제 밀 것인가"에 답한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import crosssell
from src.analysis import inflow as inflow_mod
from src.ui import charts
from src.ui import components as C


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars = fs.slice_ars(ctx["data"]["ars"])
    sales = fs.slice_sales(ctx["data"]["sales"])

    if sales.empty:
        C.empty_state(
            "선택한 구간·필터에 판매 데이터가 없습니다.",
            "'최신 유입(2026-05)' 구간에는 판매 데이터가 존재하지 않습니다. "
            "'정밀 검증' 또는 '연간 확장' 구간을 선택해 주세요.",
        )
        return

    infl = inflow_mod.build_inflow(ars, sales, fs.inflow_weights)
    csm = crosssell.compute_csm(sales, infl)

    # ── 요약 3열 ─────────────────────────────────────────────────────
    cols = st.columns(3)
    vip_items = csm.loc[csm["tier"] != S.TIER_STANDARD] if not csm.empty else csm
    with cols[0]:
        C.kpi_card("VIP 상품 수", C.fmt_int(len(vip_items), "종"), "", 0,
                   note="프리미엄 + 선물번들")
    total_qty = float(sales["qty"].sum())
    vip_qty = float(sales.loc[sales["tier"] != S.TIER_STANDARD, "qty"].sum())
    with cols[1]:
        C.kpi_card("VIP 판매 비중",
                   C.fmt_pct(vip_qty / total_qty if total_qty else None), "", 0,
                   note=f"{C.fmt_int(vip_qty)} / {C.fmt_int(total_qty)}개")
    with cols[2]:
        if csm.empty:
            C.kpi_card("최고 CSM 상품", "—", "", 0)
        else:
            top = csm.iloc[0]
            C.kpi_card("최고 CSM 상품", str(top["item"])[:18],
                       f"CSM {top['csm']:.1f}", 0,
                       note=str(top["channel_label"]))

    # ── 요일 × 월 수요 히트맵 ─────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "수요 히트맵 — 요일 × 월",
        "주간 패턴과 계절성을 한 장에서 본다.",
        badge=fs.period_badge,
    )
    channel_options = [c for c in fs.channels if c in set(sales["channel"])]
    if not channel_options:
        channel_options = sorted(set(sales["channel"]))
    picked = st.radio(
        "채널", channel_options, horizontal=True, key="fnb_hm_channel",
        format_func=lambda c: S.CHANNEL_LABEL.get(c, c),
    )
    pivot = crosssell.heatmap_dow_month(sales, picked)
    if pivot.empty or not pivot.notna().to_numpy().any():
        C.empty_state("이 채널·구간에는 히트맵을 그릴 데이터가 없습니다.")
    else:
        st.plotly_chart(
            charts.dow_month_heatmap(
                pivot,
                subtitle="원본에 주문 시각(hour) 컬럼이 없어 요일 × 월 수요 패턴으로 "
                         "분석합니다 — 심야 피크타임 분해는 불가능합니다.",
            ),
            width="stretch", key="fnb_heatmap",
        )
        # 인덱스 이름은 crosssell 에서 '기준 월' 로 지정해 두었다 (요일 '월' 과 충돌 방지)
        C.table_view(pivot.round(1).reset_index(),
                     label="히트맵 데이터 표로 보기")
        # 최고 셀을 문장으로 한 번 더 알려준다 (색상만으로 읽게 하지 않는다)
        flat = pivot.stack()
        if not flat.empty:
            (month, dow), value = flat.idxmax(), flat.max()
            C.caveat(
                f"최고 수요 지점: {month} {dow}요일 (평균 {value:,.0f}개). "
                f"{S.CHANNEL_LABEL.get(picked, picked)} 기준."
            )

    # ── VIP 상품 Top + 트리맵 ─────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "고단가 상품과 영업장 구성",
        "왼쪽은 티어 3색 수평 바, 오른쪽 트리맵은 색을 카테고리가 아니라 판매량 "
        "크기로 쓴다(조각이 수십 개라 카테고리 색은 구분 불가).",
        badge=fs.period_badge,
    )
    left, right = st.columns(2)
    with left:
        pool = crosssell.top_items(
            csm, channel=None, n=S.TOP_N_ITEMS,
            tiers=(S.TIER_PREMIUM, S.TIER_BUNDLE),
            exclude_comp=True, by="total_qty",
        )
        if pool.empty:
            C.empty_state("VIP 티어 상품이 없습니다.",
                          "사이드바 '상품 티어' 필터를 확인해 주세요.")
        else:
            st.plotly_chart(
                charts.top_items_bar(pool, value_col="total_qty",
                                     value_label="총 판매수량"),
                width="stretch", key="fnb_top",
            )
            C.table_view(
                pool[["item", "channel_label", "venue", "tier_label",
                      "total_qty", "margin_proxy", "csm"]].round(3).rename(columns={
                    "item": "상품", "channel_label": "채널", "venue": "영업장",
                    "tier_label": "티어", "total_qty": "총 판매수량",
                    "margin_proxy": "마진 프록시", "csm": "CSM"}),
                label="VIP 상품 표로 보기",
            )
    with right:
        casino_present = S.CH_CASINO in set(sales["channel"])
        if not casino_present:
            C.empty_state("카지노 식음 채널이 선택되지 않았습니다.")
        else:
            mint_only = st.toggle("민트 바만 보기", value=False, key="fnb_mint")
            venues = ("민트 바",) if mint_only else None
            frame = crosssell.treemap_frame(
                sales, S.CH_CASINO, venues=venues,
                exclude_comp=fs.exclude_comp,
            )
            if frame.empty:
                C.empty_state("트리맵을 그릴 데이터가 없습니다.")
            else:
                st.plotly_chart(charts.venue_treemap(frame), width="stretch",
                                key="fnb_treemap")
                C.table_view(
                    frame.rename(columns={"venue": "영업장", "item": "상품",
                                          "qty": "판매수량", "tier_label": "티어"}),
                    label="트리맵 데이터 표로 보기",
                )

    # ── CSM 테이블 ────────────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "교차판매 매칭 스코어 (CSM)",
        "Lift = 고유입일 판매 비중 / 저유입일 판매 비중. 1을 넘으면 카지노 유입에 "
        "반응하는 상품이다.",
        badge=fs.period_badge,
    )
    if csm.empty:
        C.empty_state("CSM 을 계산할 데이터가 없습니다.")
    else:
        has_lift = csm["lift"].notna().any()
        view = csm.head(50).assign(
            상품=csm.head(50)["item"],
            채널=csm.head(50)["channel_label"],
            영업장=csm.head(50)["venue"],
            티어=csm.head(50)["tier_label"],
            Lift=csm.head(50)["lift"].round(2),
            총판매=csm.head(50)["total_qty"],
            마진프록시=csm.head(50)["margin_proxy"].round(3),
            CSM=csm.head(50)["csm"].round(1),
            컴프=csm.head(50)["is_comp"].map({True: "컴프", False: ""}),
        )[["상품", "채널", "영업장", "티어", "Lift", "총판매", "마진프록시",
           "CSM", "컴프"]]
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={
                "CSM": st.column_config.ProgressColumn(
                    "CSM", min_value=0, max_value=100, format="%.1f"),
                "Lift": st.column_config.NumberColumn(
                    "Lift", format="%.2f",
                    help="1 초과 = 카지노 유입 증가 시 판매 비중이 커지는 상품"),
                "총판매": st.column_config.NumberColumn("총 판매수량", format="%d"),
            },
        )
        if not has_lift:
            C.caveat(
                "이 구간은 표본이 짧아 고유입일 / 저유입일을 나눌 수 없어 Lift 를 "
                "계산하지 않았습니다 (CSM 은 물량·마진 신호로만 산출)."
            )
        comp_qty = float(sales.loc[sales["is_comp"], "qty"].sum())
        if comp_qty:
            C.caveat(
                f"'(V)' 접두 상품 {C.fmt_int(comp_qty)}개는 컴프/바우처로 추정되어 "
                "마진 기여를 0으로 두고 티어를 일반으로 강등했습니다 "
                f"(전체 수량의 {comp_qty / total_qty * 100:.1f}%)."
            )

    # ── 번들 추천 ─────────────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "추천 번들 — 룸서비스 × 지역특산품",
        "두 채널을 묶는 이유: 서로 상관이 가장 느슨해 아직 같이 팔리지 않는다 = 여지가 크다.",
        badge=fs.period_badge,
    )
    bundles = crosssell.recommend_bundles(csm, sales, top_n=3)
    if not bundles:
        C.empty_state(
            "번들을 만들 두 채널 데이터가 함께 필요합니다.",
            "사이드바에서 룸서비스와 지역특산품을 모두 선택해 주세요.",
        )
    else:
        for col, bundle in zip(st.columns(len(bundles)), bundles):
            with col:
                C.bundle_card(bundle)

    C.caveat(
        "고마진 지표는 단가 데이터 미제공으로 상품명 티어 + 판매 희소도 기반 프록시입니다. "
        "할인구분여부는 수량 기준 2.3%(카지노)·0.2%(룸서비스)로 희소해 보조 신호로만 썼습니다."
    )
