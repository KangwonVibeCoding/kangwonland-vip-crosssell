"""탭 3 — 유입-소비 시차 엔진.

이 탭이 프로젝트의 차별화 포인트다. 실데이터에서 채널별 반응 시차가 다르다는
사실을 발견했고, 그것이 마케팅 처방을 바꾼다:

    카지노 식음 D+0 (+0.796)  당일 즉시 소비
    룸서비스    D+0 (+0.733)  당일 숙박 소비
    지역특산품  D+1 (+0.467)  익일 — 체크아웃 선물 구매

"유입 피크일 당일엔 F&B, 익일 오전엔 특산품 쿠폰."
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import lag as lag_mod
from src.ui import charts
from src.ui import components as C


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars = fs.slice_ars(ctx["data"]["ars"])
    sales = fs.slice_sales(ctx["data"]["sales"])

    if ars.empty or sales.empty:
        C.empty_state(
            "시차 분석에는 유입(ARS)과 판매 데이터가 같은 구간에 모두 필요합니다.",
            "사이드바에서 '정밀 검증 · 2024-12' 구간을 선택하면 실측 조인으로 분석됩니다.",
        )
        return

    infl = inflow_mod.compute_cii(ars, fs.inflow_weights).rename(
        columns={"cii": "inflow", "cii_ma7": "inflow_ma7"}
    )
    matrix = lag_mod.lag_matrix(infl, sales, max_lag=fs.max_lag)
    best = lag_mod.best_lags(matrix)

    if matrix.empty or best.empty:
        C.empty_state("이 구간에서는 시차 상관을 계산할 수 없습니다.",
                      "겹치는 날짜가 3일 이상 필요합니다.")
        return

    # ── hero ─────────────────────────────────────────────────────────
    delayed = best.loc[best["best_lag"] >= 1]
    if not delayed.empty:
        row = delayed.sort_values("pearson", ascending=False).iloc[0]
        headline = f"{row['channel_label']}은 유입 당일이 아니라 D+{int(row['best_lag'])}에 팔린다"
        sub = ("채널마다 유입에 반응하는 시점이 다르다. 같은 날 프로모션을 몰아주면 "
               "지연 반응 채널을 놓친다.")
    else:
        headline = "모든 채널이 유입 당일에 반응한다"
        sub = "이 구간에서는 지연 반응 채널이 관측되지 않았다."
    C.hero_banner(
        headline=headline, sub=sub,
        stats_pairs=[
            (f"{r.channel_label}", f"D+{int(r.best_lag)} · r={r.pearson:+.3f}")
            for r in best.itertuples()
        ],
        badge="실측 조인",
        badge_detail=f"{int(matrix['n_days'].max())}일 기준",
    )

    # ── 래그 상관 히트맵 ──────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "래그 상관 히트맵",
        "유입(D) 대비 판매(D+k) 상관. 부호가 의미를 갖는 유일한 차트여서 발산 "
        "팔레트를 쓴다(중앙 회색 = 무상관). 채널별 최댓값 셀에 링을 둘렀다.",
        badge="실측 조인",
    )
    st.plotly_chart(charts.lag_heatmap(matrix), width="stretch", key="tm_heatmap")
    C.table_view(
        matrix.assign(
            시차=matrix["lag"].map(lambda k: f"D+{k}"),
            Pearson=matrix["pearson"].round(4),
            Spearman=matrix["spearman"].round(4),
        )[["channel_label", "시차", "Pearson", "Spearman", "n_days"]].rename(
            columns={"channel_label": "채널", "n_days": "표본일수"}),
        label="래그 상관 표로 보기",
    )

    # ── 요일 프로파일 + 시차별 상품 그룹 ──────────────────────────────
    st.markdown("")
    C.section_header(
        "요일 프로파일과 시차별 공략 상품",
        "왼쪽: 유입과 각 채널의 요일 인덱스를 겹쳐 그렸다. 피크 요일이 어긋나는 "
        "지점이 곧 시차의 증거다.",
        badge="실측 조인",
    )
    left, right = st.columns([0.55, 0.45])
    dow_table = inflow_mod.dow_profile_table(ars, sales)
    with left:
        if dow_table.empty:
            C.empty_state("요일 프로파일을 계산할 데이터가 없습니다.")
        else:
            st.plotly_chart(charts.dow_profile_lines(dow_table),
                            width="stretch", key="tm_dow")
            C.table_view(
                dow_table[["label", *S.DOW_NAMES]].round(3).rename(
                    columns={"label": "계열"}),
                label="요일 인덱스 표로 보기",
            )
    with right:
        item_lags = lag_mod.best_lag_by_item(infl, sales, max_lag=fs.max_lag)
        if item_lags.empty:
            C.empty_state(
                "상품별 시차를 계산할 표본이 부족합니다.",
                "판매 일수 10일 이상, 총 수량 30개 이상인 상품만 대상으로 합니다.",
            )
        else:
            dropped = int(item_lags.attrs.get("dropped", 0))
            for k in (0, 1):
                group = lag_mod.lag_groups(item_lags, k, n=8)
                st.markdown(f"**D+{k} 공략 상품**")
                if group.empty:
                    st.caption(f"D+{k} 에 반응하는 상품이 없습니다.")
                    continue
                st.dataframe(
                    group.assign(
                        상품=group["item"], 채널=group["channel_label"],
                        r=group["pearson"].round(3), 총판매=group["total_qty"],
                    )[["상품", "채널", "r", "총판매"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "총판매": st.column_config.NumberColumn("총 판매", format="%d"),
                    },
                )
            if dropped:
                C.caveat(
                    f"표본 부족으로 제외된 상품 {dropped}종 "
                    "(판매 일수 10일 미만 또는 총 수량 30개 미만)."
                )

    # ── 계절성 ────────────────────────────────────────────────────────
    month_table = inflow_mod.month_profile_table(
        fs.slice_sales(ctx["data"]["sales"])
    )
    if not month_table.empty and month_table[
        [m for m in range(1, 13) if m in month_table.columns]
    ].notna().to_numpy().sum() > 3:
        st.markdown("")
        C.section_header(
            "채널별 계절성",
            "성수기가 채널마다 다르다 — 월별로 집중 채널을 바꿔야 한다.",
            badge=fs.period_badge,
        )
        st.plotly_chart(charts.month_profile_lines(month_table),
                        width="stretch", key="tm_month")
        C.table_view(
            month_table[["label", *[m for m in range(1, 13)
                                    if m in month_table.columns]]].round(3).rename(
                columns={"label": "채널", **{m: f"{m}월" for m in range(1, 13)}}),
            label="월 인덱스 표로 보기",
        )

    # ── 실행 처방 ─────────────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "실행 처방",
        "위 수치에서 자동 생성된다 — 필터를 바꾸면 처방도 함께 바뀐다.",
        badge="실측 조인",
    )
    cards = lag_mod.prescriptions(best, dow_table, month_table)
    if not cards:
        C.empty_state("처방을 생성할 근거가 부족합니다.")
    else:
        for col, card in zip(st.columns(len(cards)), cards):
            with col:
                C.prescription_card(card["title"], card["targets"],
                                    card["detail"], card["evidence"])

    C.caveat(
        "상관관계는 인과관계가 아닙니다. 표본은 유입·판매가 겹치는 "
        f"{int(matrix['n_days'].max())}일이며, 지연 상관은 주말 효과와 분리되지 "
        "않았을 수 있습니다."
    )
