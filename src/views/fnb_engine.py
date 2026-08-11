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
        # 구간명은 settings 에서 가져온다 — 문구에 박아두면 구간 정의가 바뀔 때 어긋난다
        names = " / ".join(f"'{m['label']}'" for m in S.PERIODS.values())
        C.empty_state(
            "선택한 구간·필터에 판매 데이터가 없습니다.",
            f"판매 데이터는 2023-01 ~ 2024-12 에만 존재합니다. {names} 중 하나를 "
            "고르거나, '직접 지정' 범위를 그 안으로 좁혀 주세요.",
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
        # 창에 데이터가 하나도 없는 달은 **표시에서** 뺀다.
        # `heatmap_dow_month` 는 1~12월을 항상 펼쳐서 돌려준다 — 달력 순서를
        # 고정하려는 것이고, 그건 분석 함수로서 옳다. 다만 그대로 그리면 31일
        # 구간에서 12행 중 11행이 빈 줄로 남아 높이 400px 이상을 쓰면서 아무것도
        # 말하지 않는다. 창에 없는 달을 '빈 셀'로 그리는 것은 결측을 보여주는 게
        # 아니라 필터 상태를 그린 것이라 오해를 부른다.
        # ⚠ 자르는 일은 여기(뷰)서 한다. 분석 함수의 반환을 바꾸면 그 함수를 쓰는
        #   다른 곳과 회귀 기준값까지 같이 흔들린다.
        months_all = len(pivot.index)
        pivot = pivot.loc[pivot.notna().any(axis=1)]
        months_present = len(pivot.index)
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
            # 창에 없는 달은 행에서 뺐다 — 몇 달치를 보고 있는지 밝히지 않으면
            # 이 히트맵이 '연간 계절성'인지 '한 달 주간 패턴'인지 구분되지 않는다.
            scope = ("이 구간에 데이터가 있는 달은 "
                     f"{months_present}개월이라 그만큼만 행으로 세웠습니다. "
                     if months_present < months_all else "")
            C.caveat(
                f"최고 수요 지점: {month} {dow}요일 (평균 {value:,.0f}개). "
                f"{S.CHANNEL_LABEL.get(picked, picked)} 기준. {scope}"
                "계절성까지 읽으려면 사이드바에서 '연간 확장' 구간을 선택하세요."
            )

    # ── VIP 상품 Top + 트리맵 ─────────────────────────────────────────
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
                pool.assign(
                    근거=pool["lift_measured"].map({True: "측정됨", False: "표본 부족"})
                )[["item", "channel_label", "venue", "tier_label",
                   "total_qty", "margin_proxy", "csm", "근거"]].round(3).rename(columns={
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
        head = csm.head(50)
        view = head.assign(
            상품=head["item"],
            채널=head["channel_label"],
            영업장=head["venue"],
            티어=head["tier_label"],
            Lift=head["lift"].round(2),
            근거=head["lift_measured"].map({True: "측정됨", False: "표본 부족"}),
            총판매=head["total_qty"],
            마진프록시=head["margin_proxy"].round(3),
            CSM=head["csm"].round(1),
            컴프=head["is_comp"].map({True: "컴프", False: ""}),
        )[["상품", "채널", "영업장", "티어", "Lift", "근거", "총판매", "마진프록시",
           "CSM", "컴프"]]
        st.dataframe(
            view, width="stretch", hide_index=True,
            column_config={
                "CSM": st.column_config.ProgressColumn(
                    "CSM", min_value=0, max_value=100, format="%.1f"),
                "Lift": st.column_config.NumberColumn(
                    "Lift", format="%.2f",
                    help="1 초과 = 카지노 유입 증가 시 판매 비중이 커지는 상품"),
                "근거": st.column_config.TextColumn(
                    "근거", help="'표본 부족' = Lift 를 측정하지 못해 실측 분포의 "
                                "중앙값을 대신 넣은 상품. 물량·마진으로만 순위가 매겨진다."),
                "총판매": st.column_config.NumberColumn("총 판매수량", format="%d"),
            },
        )
        if not has_lift:
            C.caveat(
                "이 구간은 표본이 짧아 고유입일 / 저유입일을 나눌 수 없어 Lift 를 "
                "계산하지 않았습니다 (CSM 은 물량·마진 신호로만 산출)."
            )
        # 몇 종을 무엇 때문에 뺐는지 밝히지 않으면 이 표가 '전 상품 순위'로 읽힌다.
        gate = csm.attrs.get("gate") or {}
        if gate.get("dropped"):
            floors = " · ".join(
                f"{S.CHANNEL_LABEL.get(ch, ch)} {qty:,.0f}개"
                for ch, qty in gate.get("min_qty", {}).items()
            )
            C.caveat(
                f"Lift 는 표본 요건을 통과한 {gate['kept']}종에만 계산했습니다 — "
                f"판매일수 {gate['min_days']}일 이상(창 {gate['window_days']}일의 "
                f"{S.LIFT_MIN_DAY_RATIO:.0%}) · 수량은 채널 중앙값 이상({floors}). "
                f"미달 {gate['dropped']}종('근거' 열의 '표본 부족')은 Lift 를 비워 "
                f"두고, 탄력성 축은 실측 분포의 중앙값({gate.get('elasticity_fill', 0):.3f})"
                "으로 채웁니다 — 0.5 를 넣으면 측정된 상품의 82%보다 높은 값이라 "
                "'표본이 부족하다'는 사실이 점수 보너스가 됩니다. 요건이 창 길이·채널 "
                "스케일을 따라가지 않으면 긴 구간에서 판매일수 몇 일짜리 상품의 Lift 가 "
                "수천까지 튀어 상위를 점령합니다(요건 도입 전 실측 최대 24,027)."
            )
        comp_qty = float(sales.loc[sales["is_comp"], "qty"].sum())
        if comp_qty:
            # 컴프는 '(V)' 접두만이 아니다 — '무료' 표기 쪽이 13배 크다.
            # 수치는 is_comp 합계인데 문구만 (V) 를 말하면 둘이 어긋난다.
            C.caveat(
                f"'(V)' 접두 · '무료' 표기 상품 {C.fmt_int(comp_qty)}개는 컴프/"
                "바우처로 추정되어 마진 기여를 0으로 두고 티어를 일반으로 "
                f"강등했습니다 (전체 수량의 {comp_qty / total_qty * 100:.1f}%). "
                "마진만 0으로 두면 물량·탄력성 신호가 남아 CSM 상위를 점령합니다."
            )

    # ── 번들 추천 ─────────────────────────────────────────────────────
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
