"""탭 4 — 프리미엄 로컬 상생 큐레이션.

지역특산품 상위 상품 + 하이원포인트 가맹점 지도 + 명절 시즌 분석 + VIP 로컬 패키지.
지역 상생과 고마진 교차판매를 같은 화면에서 잇는다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import crosssell
from src.analysis import inflow as inflow_mod
from src.analysis import stats
from src.ui import charts
from src.ui import components as C
from src.ui import maps


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars = fs.slice_ars(ctx["data"]["ars"])
    sales = fs.slice_sales(ctx["data"]["sales"])
    merchants: pd.DataFrame = ctx["data"]["merchants"]

    local_sales = sales.loc[sales["channel"] == S.CH_LOCAL] if not sales.empty else sales
    deck_df = maps.build_merchants(merchants, fs.radius_km)

    # ── 요약 3열 ─────────────────────────────────────────────────────
    cols = st.columns(3)
    total_qty = float(local_sales["qty"].sum()) if not local_sales.empty else 0.0
    with cols[0]:
        C.kpi_card("특산품 총 판매", C.fmt_int(total_qty, "개"), "", 0,
                   note="정태영삼 지역특산품점")
    prem_qty = float(
        local_sales.loc[local_sales["tier"] == S.TIER_PREMIUM, "qty"].sum()
    ) if not local_sales.empty else 0.0
    with cols[1]:
        C.kpi_card("프리미엄 티어 비중",
                   C.fmt_pct(prem_qty / total_qty if total_qty else None), "", 0,
                   note="명품·와인·홍삼 등")
    with cols[2]:
        C.kpi_card("반경 내 가맹점", C.fmt_int(len(deck_df), "곳"),
                   f"반경 {fs.radius_km:,.0f}km", 0,
                   note="하이원포인트 사용처")

    if local_sales.empty and deck_df.empty:
        C.empty_state(
            "이 구간·필터에는 특산품 판매와 가맹점 데이터가 모두 없습니다.",
            "사이드바에서 '지역특산품' 채널을 선택하고 반경을 넓혀 보세요.",
        )
        return

    # ── 특산품 Top + 지도 ─────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "특산품 상위 상품과 포인트 가맹점 분포",
        "특산품점은 상시 영업장이 1곳이라 영업장 대신 상품 티어로 구분한다. "
        "지도 색은 업종 3그룹(전체쌍 색약 검증 상한).",
        badge=fs.period_badge,
    )
    infl = inflow_mod.build_inflow(ars, sales, fs.inflow_weights)
    csm = crosssell.compute_csm(sales, infl) if not sales.empty else pd.DataFrame()
    local_top = crosssell.top_items(
        csm, channel=S.CH_LOCAL, n=S.TOP_N_LOCAL, exclude_comp=True, by="total_qty"
    ) if not csm.empty else pd.DataFrame()

    left, right = st.columns(2)
    with left:
        if local_top.empty:
            C.empty_state("특산품 판매 데이터가 없습니다.",
                          "사이드바에서 '지역특산품' 채널을 선택해 주세요.")
        else:
            st.plotly_chart(
                charts.top_items_bar(local_top, value_col="total_qty",
                                     value_label="총 판매수량"),
                width="stretch", key="lc_top",
            )
            C.table_view(
                local_top[["item", "tier_label", "total_qty", "margin_proxy",
                           "csm"]].round(3).rename(columns={
                    "item": "상품", "tier_label": "티어", "total_qty": "총 판매수량",
                    "margin_proxy": "마진 프록시", "csm": "CSM"}),
                label="특산품 표로 보기",
            )
    with right:
        if deck_df.empty:
            C.empty_state(
                "반경 내 가맹점이 없습니다.",
                "사이드바 고급 설정에서 '가맹점 반경'을 넓혀 보세요.",
            )
        else:
            st.pydeck_chart(maps.merchant_deck(deck_df, fs.radius_km),
                            width="stretch", height=430)
            legend = " · ".join(
                f"{group}" for group in deck_df["group"].unique()
            )
            st.caption(f"업종 그룹: {legend} · 마커 크기 = 큐레이션 스코어")
            C.table_view(
                deck_df[["merchant", "category", "group", "dist_km",
                         "curation"]].round(2).rename(columns={
                    "merchant": "가맹점", "category": "업종", "group": "그룹",
                    "dist_km": "거리(km)", "curation": "큐레이션"}),
                label="가맹점 표로 보기",
            )

    # ── 명절 선물 시즌 ────────────────────────────────────────────────
    full_local = ctx["data"]["sales"]
    full_local = full_local.loc[full_local["channel"] == S.CH_LOCAL]
    if not full_local.empty:
        daily = stats.daily_qty(full_local)
        months_present = pd.to_datetime(pd.Series(daily.index)).dt.month.nunique()
        if months_present >= 4:
            st.markdown("")
            C.section_header(
                "명절 선물 시즌 분석",
                "특산품은 추석 전후(9~10월)에 수요가 몰린다. 연간 데이터 전체 기준.",
                badge="요일 브릿지",
            )
            st.plotly_chart(charts.season_line(daily, highlight=(9, 10)),
                            width="stretch", key="lc_season")
            idx = stats.month_index(daily)
            peak_month = int(idx.idxmax()) if idx.notna().any() else None
            if peak_month:
                C.caveat(
                    f"최고 월은 {peak_month}월(지수 {idx[peak_month]:.2f})입니다. "
                    "선물 패키지 물량과 큐레이션은 이 시점에 앞세워야 합니다."
                )
            season = full_local.loc[full_local["month"].isin([9, 10])]
            if not season.empty:
                top_season = (
                    season.groupby("item", as_index=False)["qty"].sum()
                    .sort_values("qty", ascending=False).head(10)
                )
                C.table_view(
                    top_season.rename(columns={"item": "상품", "qty": "9~10월 판매수량"}),
                    label="명절 시즌 상위 상품 표로 보기",
                )

    # ── VIP 로컬 패키지 ───────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "VIP 로컬 패키지 제안",
        "큐레이션 상위 가맹점 × CSM 상위 특산품 조합. 특산품은 유입 익일(D+1)에 "
        "팔리므로 집행 시점을 체크아웃 동선에 맞춘다.",
        badge=fs.period_badge,
    )
    prem_top = crosssell.top_items(
        csm, channel=S.CH_LOCAL, n=6, tiers=(S.TIER_PREMIUM, S.TIER_BUNDLE),
        exclude_comp=True,
    ) if not csm.empty else pd.DataFrame()
    packages = maps.curation_packages(deck_df, prem_top, top_n=3)
    if not packages:
        C.empty_state(
            "패키지를 구성할 가맹점 또는 특산품 데이터가 부족합니다.",
            "반경을 넓히거나 상품 티어 필터를 완화해 주세요.",
        )
    else:
        for col, pkg in zip(st.columns(len(packages)), packages):
            with col:
                C.prescription_card(
                    title=f"패키지 {pkg['rank']}",
                    targets=[pkg["tier"], pkg["category"]],
                    detail=f"{pkg['item']} + {pkg['merchant']}",
                    evidence=(
                        f"상품 CSM {pkg['item_csm']:.1f} · "
                        f"가맹점 큐레이션 {pkg['curation']:.1f} · "
                        f"카지노에서 {pkg['dist_km']:.1f}km"
                    ),
                )

    C.caveat(
        "가맹점 큐레이션 스코어 = 카지노 기준점(37.2007, 128.8155)으로부터의 근접도"
        "(5km 감쇠)와 업종 프리미엄 가중의 합입니다. 업종 가중치는 도메인 가정값입니다."
    )
