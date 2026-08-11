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
        # 좌표를 못 찾은 가맹점은 지도에서 빠진다. 그 수를 카드에 그대로 적어
        # '반경 내 N곳'이 전수인 것처럼 읽히지 않게 한다.
        no_coord = int(deck_df.attrs.get("no_coord", 0))
        C.kpi_card("반경 내 가맹점", C.fmt_int(len(deck_df), "곳"),
                   f"반경 {fs.radius_km:,.0f}km", 0,
                   note=(f"좌표 미확인 {no_coord}곳 제외" if no_coord
                         else "하이원포인트 사용처"))

    if local_sales.empty and deck_df.empty:
        C.empty_state(
            "이 구간·필터에는 특산품 판매와 가맹점 데이터가 모두 없습니다.",
            "사이드바에서 '지역특산품' 채널을 선택하고 반경을 넓혀 보세요.",
        )
        return

    # ── 특산품 Top + 지도 ─────────────────────────────────────────────
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
            # 왼쪽 막대와 오른쪽 지도의 아래끝을 맞춘다 — 안 맞추면 한쪽만
            # 삐져나와 두 칸이 짝으로 안 읽힌다.
            # ⚠ 높이는 **낮은 쪽에 맞추지 않는다.** 특산품 20종을 지도의 430px
            #   에 욱여넣으면 막대 하나가 17px 로 눌린다. 정렬하자고 한쪽을
            #   찌그러뜨리는 건 손해다 — 둘 다 charts.PAIR_HEIGHT 로 올리면
            #   막대도 지도도 숨통이 트인다.
            st.plotly_chart(
                charts.top_items_bar(local_top, value_col="total_qty",
                                     value_label="총 판매수량",
                                     height=charts.PAIR_HEIGHT),
                width="stretch", key="lc_top",
            )
            C.table_view(
                local_top.assign(
                    근거=local_top["lift_measured"].map(
                        {True: "측정됨", False: "표본 부족"})
                )[["item", "tier_label", "total_qty", "margin_proxy",
                   "csm", "근거"]].round(3).rename(columns={
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
                            width="stretch", height=charts.PAIR_HEIGHT)
            legend = " · ".join(str(g) for g in deck_df["group"].unique())
            st.caption(f"업종 그룹: {legend} · 마커 크기 = 교차판매 적합도")
            C.table_view(
                deck_df[["merchant", "category", "group", "dist_km",
                         "fit", "fit_score"]].round(2).rename(columns={
                    "merchant": "가맹점", "category": "업종", "group": "그룹",
                    "dist_km": "거리(km)", "fit": "업종 적합도",
                    "fit_score": "교차판매 적합도"}),
                label="가맹점 표로 보기",
            )

            # 적합도 가중치는 데이터에서 유도한 값이 아니라 판단이다.
            # 숨기면 과장이 되고, 공개하면 검증 가능한 설계가 된다.
            unknown = int((~deck_df["fit_known"].to_numpy(dtype=bool)).sum()) \
                if "fit_known" in deck_df.columns else 0
            with st.expander("교차판매 적합도는 어떻게 계산했나"):
                st.markdown(
                    "**교차판매 적합도 = 100 × (0.6 × 근접도 + 0.4 × 업종 적합도)**\n\n"
                    f"근접도 = `exp(−거리km / {S.PROXIMITY_DECAY_KM:.0f})` — "
                    "강원랜드에서 멀어질수록 지수적으로 감소합니다."
                )
                st.dataframe(
                    pd.DataFrame(
                        [{"업종": k, "적합도": v} for k, v in S.CATEGORY_FIT.items()]
                    ).sort_values("적합도", ascending=False),
                    width="stretch", hide_index=True,
                )
                st.caption(
                    "업종은 소상공인시장진흥공단 상가(상권)정보의 상권업종대분류명"
                    "(표준산업분류 기반)입니다. **적합도 가중치는 데이터에서 유도한 "
                    "값이 아니라 '카지노 방문객 체류 동선과 겹치는가'라는 판단**이며, "
                    "그래서 표를 그대로 공개합니다. "
                    + (f"업종 미상 {unknown}곳은 중립값 "
                       f"{S.CATEGORY_FIT_NEUTRAL}로 처리했습니다 — 0으로 두면 "
                       "'부적합'이라는 없는 근거를 만들기 때문입니다."
                       if unknown else "")
                )
                st.caption(
                    "포인트 사용 한도액(PNT_USABLE_AMT)은 지표에 넣지 않았습니다 — "
                    "1,681곳 중 1,578곳이 정확히 400만원이라(CV 0.052) 변별력이 없습니다."
                )

    # ── 명절 선물 시즌 ────────────────────────────────────────────────
    full_local = ctx["data"]["sales"]
    full_local = full_local.loc[full_local["channel"] == S.CH_LOCAL]
    if not full_local.empty:
        daily = stats.daily_qty(full_local)
        months_present = pd.to_datetime(pd.Series(daily.index)).dt.month.nunique()
        if months_present >= 4:
            C.section_header(
                "명절 선물 시즌 분석",
                "특산품은 추석 전후(9~10월)에 수요가 몰린다. 연간 데이터 전체 기준.",
                badge="판매 실측",
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
    C.section_header(
        "VIP 로컬 패키지 제안",
        "적합도 상위 가맹점 × CSM 상위 특산품 조합. 특산품은 유입 익일(D+1)에 "
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
                    targets=[pkg["tier"], pkg["category"] or "업종 미상"],
                    detail=f"{pkg['item']} + {pkg['merchant']}",
                    evidence=(
                        f"상품 CSM {pkg['item_csm']:.1f} · "
                        f"가맹점 적합도 {pkg['fit_score']:.1f}"
                        + ("" if pkg["fit_known"] else "(업종 미상)")
                        + f" · 카지노에서 {pkg['dist_km']:.1f}km"
                    ),
                )
        # 카드는 읽기용이라 옮겨 적어야 한다. 제휴 담당자에게 넘길 수 있는 형태로
        # 같은 내용을 표로 준다 — '업종 미상' 여부까지 열로 남겨야 받는 쪽이
        # 적합도 점수를 그대로 믿지 않는다.
        C.download_csv(
            pd.DataFrame([{
                "패키지": pkg["rank"],
                "특산품": pkg["item"],
                "티어": pkg["tier"],
                "가맹점": pkg["merchant"],
                "업종": pkg["category"] or "업종 미상",
                "업종확인": "확인" if pkg["fit_known"] else "미상(중립값 적용)",
                "상품 CSM": round(float(pkg["item_csm"]), 1),
                "교차판매 적합도": round(float(pkg["fit_score"]), 1),
                "거리(km)": round(float(pkg["dist_km"]), 1),
            } for pkg in packages]),
            f"⬇ VIP 로컬 패키지 {len(packages)}건 CSV 내려받기",
            f"local_packages_{fs.start:%Y%m%d}_{fs.end:%Y%m%d}.csv",
            key="lc_dl_packages",
            help_text="제휴 담당자에게 그대로 전달할 수 있는 형태입니다 "
                      "(Excel 한글 호환).",
        )

    C.caveat(
        f"교차판매 적합도 = 카지노 기준점{S.CASINO_LATLON}으로부터의 근접도"
        f"({S.PROXIMITY_DECAY_KM:.0f}km 감쇠) {S.CURATION_WEIGHTS['proximity']:.0%} + "
        f"업종 적합도 {S.CURATION_WEIGHTS['fit']:.0%}. 업종 가중치는 데이터가 아니라 "
        "판단이며 위 '어떻게 계산했나'에 표로 공개했습니다. "
        "가맹점 좌표·업종은 상가(상권)정보 조인 결과로, 상호·주소 매칭이라 "
        "전수가 아닙니다."
    )
