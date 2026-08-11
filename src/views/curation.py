"""탭 5 — AI VIP 큐레이션.

세그먼트 페르소나 × 기준일 하루를 넣으면 상품 3슬롯(Main / Sub / Cross-sell)과
그 근거를 낸다. 로직은 전부 `src/analysis/curation.py` 에 있고 이 파일은 화면만
그린다 — 같은 함수를 CLI(`scripts/prototype_curation.py`)가 쓰므로 화면과 CLI 가
다른 답을 낼 수 없다.

⚠ 개인 데이터가 아니다. 이 저장소에는 고객ID·영수증·장바구니가 한 건도 없고
판매 최소 입도가 `일자 × 영업장 × 상품 × 수량` 이라, "고객"은 성별×연령
**세그먼트**다. 화면이 개인 추천처럼 읽히면 안 되므로 상단에 그 사실을 적고
`customer_id` 를 `SEG-...` 로 노출한다.

모든 추천에 `근거` 열을 붙인다. 값마다 `[실측]/[파생]/[가정]` 라벨이 달려 있어
어디까지가 데이터고 어디부터가 가정인지 화면에서 바로 읽힌다.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import curation as cur
from src.ui import components as C

_SLOT_NOTE = {
    "Main": "오늘 가장 먼저 안내할 한 가지",
    "Sub": "같은 채널의 결이 다른 대안",
    "Cross-sell": "채널을 넘기는 제안 (티어 밴드 면제)",
}


def _default_date(inflow_dates: pd.Series, fs) -> pd.Timestamp:
    """기준일 기본값 — 선택 구간의 **마지막 판매일**.

    "가장 최근"이 기본이어야 사용자가 날짜를 바꾸지 않고도 지금 쓸 수 있는 화면을
    본다. 구간 밖이면 전체 마지막 날로 떨어진다.
    """
    in_range = inflow_dates[(inflow_dates >= pd.Timestamp(fs.start))
                            & (inflow_dates <= pd.Timestamp(fs.end))]
    return (in_range.iloc[-1] if not in_range.empty else inflow_dates.iloc[-1])


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars = fs.slice_ars(ctx["data"]["ars"])
    sales = fs.slice_sales(ctx["data"]["sales"])
    demo = ctx["data"].get("demo", pd.DataFrame())

    if ars.empty or sales.empty:
        C.empty_state(
            "큐레이션에는 유입(ARS)과 판매 데이터가 같은 구간에 모두 필요합니다.",
            "사이드바에서 판매 데이터가 있는 구간(2023-01 ~ 2024-12)을 선택해 주세요.",
        )
        return

    C.section_header(
        "AI VIP 큐레이션",
        "세그먼트 × 기준일 → 상품 3슬롯. 모든 값에 출처 라벨이 붙습니다.",
        badge=fs.period_badge,
    )
    st.caption(
        "⚠ 개인 추천이 아닙니다. 이 데이터셋에는 고객ID·영수증·장바구니가 없고 "
        "판매 최소 입도가 `일자 × 영업장 × 상품 × 수량` 이라, 여기서 '고객'은 "
        "**성별×연령 세그먼트**입니다."
    )

    # ── 입력 ─────────────────────────────────────────────────────────
    dates = sales["date"].drop_duplicates().sort_values().reset_index(drop=True)
    left, right = st.columns([1, 1])
    with left:
        persona_id = st.selectbox(
            "세그먼트", sorted(cur.PERSONAS),
            format_func=lambda k: f"{cur.PERSONAS[k]['label']}  ({k})",
            index=sorted(cur.PERSONAS).index("SEG-F50-ROOM")
            if "SEG-F50-ROOM" in cur.PERSONAS else 0,
            key="cu_persona",
        )
    with right:
        as_of = st.date_input(
            "기준일", value=_default_date(dates, fs),
            min_value=dates.iloc[0], max_value=dates.iloc[-1], key="cu_date",
        )

    try:
        res = cur.curate(ars, sales, demo, persona_id, pd.Timestamp(as_of))
    except ValueError as exc:
        C.empty_state("추천을 만들지 못했습니다.", str(exc))
        return

    for msg in res["warnings"]:
        st.info(msg)

    payload, day = res["payload"], res["day"]
    st.markdown(f"**{payload['customer_id']}** · {payload['today_intent_summary']}")

    # ── 3슬롯 ────────────────────────────────────────────────────────
    for rec in payload["recommendations"]:
        row = next(item for _, item in res["slots"]
                   if str(item["item_id"]) == rec["product_id"])
        C.prescription_card(
            title=f"{rec['rank']}. {rec['recommendation_type']} — "
                  f"{_SLOT_NOTE[rec['recommendation_type']]}",
            targets=[S.CHANNEL_LABEL.get(str(row["channel"]), str(row["channel"])),
                     S.TIER_LABEL.get(str(row["tier"]), str(row["tier"]))],
            detail=rec["product_name"],
            evidence=rec["vip_personalized_message"],
        )
        with st.expander(f"근거 — {rec['product_name']}", expanded=False):
            st.caption(rec["internal_reason"])

    # ── 세그먼트 축 ──────────────────────────────────────────────────
    seg = res["seg"]
    if not seg.empty:
        C.section_header(
            "세그먼트 선호 축",
            "회원 성별·연령 구성비 변화 ↔ 품목 판매 비중 변화의 상관. "
            "**[파생]** — 개인 구매 기록이 아니라 집단 비율끼리의 상관이고, "
            "원 지표가 누적이라 시간 추세와 분리되지 않습니다.",
        )
        view = seg.assign(
            적용=seg["seg_used"].map({True: "적용", False: "미적용(판매 커버리지)"}),
        ).rename(columns={"seg_item": "상품", "seg_r": "r", "seg_months": "판매월"})
        C.table_view(view[["상품", "r", "판매월", "적용"]]
                     .sort_values("r", ascending=False),
                     label="세그먼트 선호 표로 보기")
        lo, hi = cur.SEGMENT_ASSUMED_WINDOW
        C.caveat(
            f"판매 커버리지 게이트: 창의 {cur.SEGMENT_MIN_MONTH_RATIO:.0%} 이상 월에서 "
            f"팔린 상품만 r 을 적용합니다. 창 중간에 생기거나 사라진 상품은 상관이 "
            f"선호가 아니라 '존재한 시기'를 재기 때문입니다. "
            f"이 표는 {lo:%Y-%m}~{hi:%Y-%m} 재계산본으로 **간주**한 값이며 "
            f"(원파일은 24개월 산출) 실제 재계산본이 아닙니다."
        )

    # ── 후보 흐름 + 상위 후보 ────────────────────────────────────────
    C.section_header("후보가 어떻게 좁혀졌나", "필터마다 몇 종이 남았는지.")
    C.table_view(
        pd.DataFrame([{"단계": label, "남은 상품": n} for label, n in res["funnel"]]),
        label="후보 필터 흐름 표로 보기",
    )

    scored = res["scored"]
    top = scored.head(8).assign(
        채널=scored.head(8)["channel"].map(S.CHANNEL_LABEL),
        티어=scored.head(8)["tier"].map(S.TIER_LABEL),
    ).rename(columns={"item": "상품", "cur_score": "점수", "intent": "의도",
                      "season": "계절계수", "base": "선호", "seg_mult": "세그계수",
                      "csm": "CSM"})
    C.table_view(
        top[["상품", "채널", "티어", "점수", "의도", "계절계수", "선호", "세그계수", "CSM"]]
        .round(3),
        label="상위 8종 점수 표로 보기",
    )

    C.download_csv(
        pd.DataFrame(payload["recommendations"]),
        "⬇ 추천 3건 CSV 내려받기",
        f"curation_{persona_id}_{pd.Timestamp(as_of):%Y%m%d}.csv",
        key="cu_dl_csv",
        help_text="근거 문자열까지 그대로 담깁니다 (Excel 한글 호환).",
    )
    st.download_button(
        "⬇ 추천 JSON 내려받기",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"curation_{persona_id}_{pd.Timestamp(as_of):%Y%m%d}.json",
        mime="application/json", key="cu_dl_json",
    )

    C.caveat(
        "점수 = 의도 0.50 + 선호 0.30 + CSM 0.20. 의도는 요일통제 편상관 × 채널 "
        "등급반응 × 상품 시차상관 × 요일·월 지수로 만들고, 선호는 페르소나 채널 "
        "가정값에 세그먼트 계수를 곱합니다. 컴프(무상 제공) 상품은 후보에서 "
        "제외되고, 3순위만 티어 밴드를 면제하되 마진 프록시 하한을 겁니다."
    )
