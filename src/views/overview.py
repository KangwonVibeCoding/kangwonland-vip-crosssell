"""탭 1 — 메인 대시보드.

가설 검증 배너 → 요일 교란 통제 → KPI → 유입 추이 → VIP 모수 → 캠페인 집행 캘린더.
상단에서 "카지노 유입이 리조트 소비를 견인한다"는 근거(상관계수)를 제시하고,
**바로 이어서 그 상관의 절반이 주말 효과였다는 통제 결과**를 같이 세운 뒤,
남은 관계를 실행 계획으로 번역한다.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings as S
from src.analysis import inflow as inflow_mod
from src.analysis import stats, vip
from src.ui import charts
from src.ui import components as C


def _render_confound(table: pd.DataFrame) -> None:
    """요일 교란 통제 절 — 반박을 각주로 밀지 않는다.

    **원칙은 그대로다**: r=0.80 을 팔고 그 한계를 저 아래 각주로 두면 안 된다.
    다만 지금은 그 반박이 **hero 헤드라인 자체**에 들어가 있다
    ("r=0.80 · 요일을 통제하면 0.56", 칩에도 0.801 → 0.370). 반박은 이미 첫 줄에서
    끝난다. 그래서 이 절은 '반박'이 아니라 **그 반박의 근거**를 펼치는 자리다.

    그 사이에 KPI 한 줄(카드 5장, 약 130px)이 들어가도 이 절은 여전히 첫 1.5 화면
    안이다 — 각주로 밀린 것이 아니다. 대신 읽는 순서가 자연스러워진다:
    주장(hero) → 규모(KPI) → 근거(여기) → 상세(차트들).

    배치도 바꿨다. 덤벨 차트와 그 해석(insight)을 **세로로 쌓지 않고 나란히** 둔다.
    "0.801 → 0.370" 을 보면서 "54%가 주말 효과였다"를 같이 읽어야 뜻이 붙는다.
    """
    if table.empty:
        return
    st.markdown("")
    n_days = int(table["n_days"].max())
    metric_label = str(table["metric_label"].iloc[0])
    C.section_header(
        "이 상관은 주말 효과인가 — 요일 교란 통제",
        "토요일엔 유입도 매출도 많다. 요일이 양변을 동시에 밀어올리는 공통 원인일 수 "
        "있어서, 요일 더미로 양변을 회귀한 잔차끼리의 상관(편상관)을 나란히 세웠다. "
        "요일은 마케팅으로 바꿀 수 없으므로, 통제 후에도 남는 관계만이 개입 여지다.",
        badge="실측 조인",
        badge_detail=(f"{n_days}일 · {metric_label} 기준 · "
                      f"부트스트랩·순열 각 {S.CONFOUND_RESAMPLES:,}회"),
    )
    lines: list[str] = []
    raw_top = table.loc[table["raw"].idxmax()]
    if raw_top["raw"]:
        shrink = 1 - (raw_top["partial"] / raw_top["raw"])
        lines.append(
            f"{raw_top['channel_label']}의 원 상관 {raw_top['raw']:.3f} 중 "
            f"{shrink:.0%}는 요일이 설명한다 — 통제 후 {raw_top['partial']:.3f} "
            f"({C.fmt_p(raw_top['p'])}, {raw_top['verdict']})."
        )
    lead = table.iloc[0]
    if stats.confound_flips_order(table):
        lines.append(
            f"통제 전 1위는 {raw_top['channel_label']}이지만 통제 후 1위는 "
            f"{lead['channel_label']}({lead['partial']:.3f}) — 순서가 뒤집힌다. "
            f"{lead['channel_label']}{C.josa(lead['channel_label'])} 요일과 무관하게 "
            "유입 그 자체를 따라간다."
        )
    weak = list(table.loc[table["verdict"] == S.CONFOUND_VERDICT_NULL, "channel_label"])
    if weak:
        lines.append(
            f"{' · '.join(weak)}{C.josa(weak[-1])} 통제 후 95% CI 가 0을 포함해 "
            "유의하지 않다 — 유입 자체보다 요일·계절 신호(탭3)로 공략할 채널이다."
        )
    # 차트와 그 해석을 나란히 — 세로로 쌓으면 "0.801 → 0.370" 을 본 뒤 스크롤해서
    # "54%가 주말 효과였다"를 읽게 된다. 그 사이에 눈이 한 번 끊긴다.
    chart_col, read_col = st.columns([0.62, 0.38], vertical_alignment="top")
    with chart_col:
        st.plotly_chart(charts.confound_dumbbell(table), width="stretch",
                        key="ov_confound")
    with read_col:
        C.insight_box(lines, title="요일을 빼고 남은 것")

    # 표와 방법론은 근거의 '뒷장'이다 — 접어 두되 없애지는 않는다
    C.table_view(
        table.assign(
            채널=table["channel_label"],
            원상관=table["raw"].round(3),
            통제후=table["partial"].round(3),
            감소=table["delta"].round(3),
            CI=[f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(table["ci_lo"], table["ci_hi"])],
            p값=table["p"].round(4),
            판정=table["verdict"],
            표본일수=table["n_days"],
        )[["채널", "원상관", "통제후", "감소", "CI", "p값", "판정", "표본일수"]].rename(
            columns={"원상관": "원 상관", "통제후": "요일 통제 후",
                     "CI": "95% CI(부트스트랩)", "p값": "p(순열검정)"}),
        label="편상관 표로 보기",
    )
    C.caveat(
        f"편상관은 요일 더미 6개로 양변을 회귀한 잔차끼리의 상관입니다. 표본 "
        f"{n_days}일에서 계산했고, CI 는 부트스트랩·p 는 순열검정"
        f"(각 {S.CONFOUND_RESAMPLES:,}회, seed {S.CONFOUND_SEED} 고정)입니다. "
        "scripts/diagnose_confound.py 로 같은 수치를 재현할 수 있습니다."
    )


def render(ctx: dict) -> None:
    fs = ctx["filters"]
    ars_all: pd.DataFrame = ctx["data"]["ars"]
    sales_all: pd.DataFrame = ctx["data"]["sales"]
    demo: pd.DataFrame = ctx["data"]["demo"]

    ars = fs.slice_ars(ars_all)
    sales = fs.slice_sales(sales_all)

    if ars.empty and sales.empty:
        C.empty_state(
            "선택한 구간에 데이터가 없습니다.",
            "사이드바에서 분석 구간을 넓히거나 채널·티어 필터를 완화해 주세요.",
        )
        return
    if not fs.channels:
        C.empty_state(
            "채널을 하나 이상 선택해 주세요.",
            "카지노 식음 · 룸서비스 · 지역특산품 중 최소 하나가 필요합니다.",
        )
        return
    if not fs.tiers:
        C.empty_state(
            "상품 티어를 하나 이상 선택해 주세요.",
            "프리미엄 · 선물번들 · 일반 중 최소 하나가 필요합니다.",
        )
        return
    if not fs.age_bands:
        C.empty_state(
            "연령대를 하나 이상 선택해 주세요.",
            "VIP 가용 모수를 계산할 세그먼트가 비어 있습니다.",
        )
        return

    # ── 가설 검증 배너 ────────────────────────────────────────────────
    corr = stats.corr_table(ars, sales)
    head = stats.headline_corr(corr)
    # 원 상관과 편상관을 **같은 화면에서 동시에** 계산한다. r=0.80 만 먼저 띄우고
    # 통제 결과를 아래에 숨기면, 배너를 보고 스크롤하지 않은 사람은 반쪽짜리
    # 근거만 가져가게 된다.
    confound = stats.confound_table(ars, sales)
    if head:
        n_days = int(corr["n_days"].max())
        top = max(head.values())
        # 기준축은 총 접수자다. 다만 ARS 전처리본에 그 컬럼이 없는 구간에서는
        # 입장권 구매 건수로 내려간다 — 어느 지표로 잰 상관인지 배지에 드러낸다.
        metric = stats.headline_metric(corr)
        metric_label = {
            "recv_total": "내국인 총 접수자",
            "tickets": "입장권 구매 건수",
        }.get(metric, "유입 지표")

        if not confound.empty:
            lead = confound.iloc[0]                      # 편상관 1위 = 통제 후 우선순위
            raw_top = confound.loc[confound["raw"].idxmax()]
            shrink = 1 - (raw_top["partial"] / raw_top["raw"]) if raw_top["raw"] else 0.0
            headline = (f"카지노 유입 ↔ 리조트 소비 r = {top:.2f} · "
                        f"요일을 통제하면 {lead['partial']:.2f}")
            sub = (
                f"원 상관 1위 {raw_top['channel_label']}"
                f"{C.josa(raw_top['channel_label'])} 요일을 빼면 "
                f"{raw_top['raw']:.3f} → {raw_top['partial']:.3f} — {shrink:.0%}가 "
                f"주말 효과였다. 통제 후에도 견고하게 남는 채널은 "
                f"{lead['channel_label']}({lead['partial']:.3f}, {C.fmt_p(lead['p'])}) — "
                "교차판매 1순위는 여기다."
            )
            # 칩도 편상관 순으로 세운다. 원 상관 순서로 두면 배너가 아직도
            # 옛 서사(카지노 식음 1위)를 말하게 된다.
            pairs = [
                (r.channel_label, f"{r.raw:.3f} → {r.partial:.3f}")
                for r in confound.itertuples()
            ]
            badge_detail = (f"{n_days}일 날짜 조인 · {metric_label} 기준 · "
                            "요일 통제 편상관 병기")
        else:
            headline = f"카지노 유입 ↔ 리조트 소비 상관 r = {top:.2f}"
            sub = ("ARS 예약 유입이 클수록 리조트 내 F&B 소비가 함께 커진다 — "
                   "교차판매 가설이 실데이터로 확인된다.")
            pairs = [(S.CHANNEL_LABEL[ch], f"{head[ch]:.3f}")
                     for ch in S.CHANNELS if ch in head]
            badge_detail = f"{n_days}일 날짜 조인 · {metric_label} 기준"

        C.hero_banner(
            headline=headline, sub=sub, stats_pairs=pairs,
            badge="실측 조인", badge_detail=badge_detail,
        )
    else:
        C.hero_banner(
            headline="이 구간은 유입·판매 데이터가 겹치지 않습니다",
            sub="ARS 예약 데이터와 판매 데이터가 함께 존재하는 구간에서만 "
                "상관을 실측할 수 있습니다. 사이드바에서 '정밀 검증' 구간을 선택해 보세요.",
            stats_pairs=[],
            badge=fs.period_badge,
        )

    # ── 지수 산출 ─────────────────────────────────────────────────────
    infl = inflow_mod.build_inflow(ars, sales, fs.inflow_weights)
    if infl.empty:
        C.empty_state("이 구간에서는 유입 지수를 계산할 수 없습니다.")
        return

    vrb = vip.compute_vrb(infl, demo, fs.age_bands, fs.gender)
    vts = vip.compute_vts(infl, vrb, sales, fs.vts_weights)

    # 이전 동일 길이 구간 (KPI 델타 비교용)
    prev_start, prev_end = fs.previous_window()
    prev_ars = ars_all.loc[ars_all["date"].between(prev_start, prev_end)]
    prev_sales = sales_all.loc[sales_all["date"].between(prev_start, prev_end)]

    # ── KPI 5열 ──────────────────────────────────────────────────────
    st.markdown("")
    cols = st.columns(5)

    tickets_now = float(ars["tickets"].sum()) if not ars.empty else None
    tickets_prev = float(prev_ars["tickets"].sum()) if not prev_ars.empty else None
    with cols[0]:
        delta, direction = C.pct_delta(tickets_now, tickets_prev)
        C.kpi_card("입장권 구매 건수", C.fmt_int(tickets_now, "건"), delta, direction,
                   note="당첨자 실구매" if tickets_now else "이 구간 ARS 없음")

    # 예약 경쟁률 = 총 접수자 / 당첨자. 2026년 ARS 전처리본에는 총 접수자
    # 컬럼이 없어 이 구간에서는 NaN 이 되고 카드가 '—' 로 뜬다. 값이 없는
    # 이유를 note 에 적어 두어야 "지표가 깨졌다"로 읽히지 않는다.
    compete_now = float(ars["compete_ratio"].mean()) if not ars.empty else None
    compete_prev = float(prev_ars["compete_ratio"].mean()) if not prev_ars.empty else None
    with cols[1]:
        delta, direction = C.point_delta(compete_now, compete_prev, 2, "배")
        has_compete = compete_now is not None and pd.notna(compete_now)
        C.kpi_card("예약 경쟁률", C.fmt_float(compete_now, 2, "배"), delta, direction,
                   note="접수자 / 당첨자" if has_compete else "이 구간 총 접수자 미제공")

    vrb_now = float(vrb["vrb"].sum()) if not vrb.empty else None
    estimated = bool(vrb["is_estimated"].any()) if not vrb.empty else False
    with cols[2]:
        ratio = float(vrb["vip_ratio"].iloc[0]) if not vrb.empty else None
        C.kpi_card("VIP 추정 모수", C.fmt_int(vrb_now, "명"),
                   f"비율 {C.fmt_pct(ratio)}", 0,
                   note="CAI 기반 추정" if estimated else "ARS 기반")

    vip_qty = float(sales.loc[sales["tier"] != S.TIER_STANDARD, "qty"].sum()) \
        if not sales.empty else 0.0
    total_qty = float(sales["qty"].sum()) if not sales.empty else 0.0
    share_now = (vip_qty / total_qty) if total_qty else None
    prev_total = float(prev_sales["qty"].sum()) if not prev_sales.empty else 0.0
    prev_vip = float(prev_sales.loc[prev_sales["tier"] != S.TIER_STANDARD, "qty"].sum()) \
        if not prev_sales.empty else 0.0
    share_prev = (prev_vip / prev_total) if prev_total else None
    with cols[3]:
        delta, direction = C.point_delta(
            (share_now or 0) * 100 if share_now is not None else None,
            (share_prev or 0) * 100 if share_prev is not None else None,
        )
        C.kpi_card("VIP 상품 비중", C.fmt_pct(share_now), delta, direction,
                   note="프리미엄 + 선물번들")

    attach_now = float(vts["attach"].median()) if "attach" in vts.columns else None
    with cols[4]:
        C.kpi_card("Attach rate", C.fmt_float(attach_now, 2), "", 0,
                   note="유입 1건당 리조트 소비량")

    # ── 요일 교란 통제 ────────────────────────────────────────────────
    # 읽는 순서: 주장(hero) → 규모(KPI) → **근거(여기)** → 상세(아래 차트들).
    # 반박 자체는 이미 hero 헤드라인에 있으므로("요일을 통제하면 0.56") 이 절이
    # KPI 뒤로 와도 각주로 밀린 것이 아니다 — 자세한 근거는 _render_confound 참조.
    _render_confound(confound)

    # ── 유입 추이 ─────────────────────────────────────────────────────
    st.markdown("")
    has_tickets = "tickets" in infl.columns and infl["tickets"].notna().any()
    C.section_header(
        "카지노 유입 추이",
        "위: 입장권 구매 건수 · 아래: 유입 강도 지수. 스케일이 달라 이중 축 대신 "
        "x축을 공유하는 2단으로 그렸다.",
        badge=fs.period_badge,
    )
    st.plotly_chart(
        charts.inflow_stack(infl, value_col="tickets"),
        width="stretch", key="ov_inflow",
    )
    table_cols = [c for c in ("date", "dow_name", "tickets", "recv_total",
                              "buy_rate", "inflow", "grade", "source")
                  if c in infl.columns]
    C.table_view(infl[table_cols].rename(columns={
        "date": "날짜", "dow_name": "요일", "tickets": "입장권 구매",
        "recv_total": "총 접수자", "buy_rate": "구매율", "inflow": "유입 지수",
        "grade": "등급", "source": "산출",
    }))
    if not has_tickets:
        C.caveat(
            "이 구간에는 ARS 데이터가 없어 카지노 식음 소비로 산출한 객장 활동 "
            "지수(CAI)를 유입 프록시로 사용했습니다."
        )
    else:
        validity = inflow_mod.cai_validity(ars, sales)
        if validity["n_days"] >= 10 and pd.notna(validity["pearson"]):
            C.caveat(
                f"참고: 같은 구간에서 판매 기반 활동지수(CAI)와 ARS 기반 유입지수(CII)의 "
                f"상관은 r={validity['pearson']:.3f} ({validity['n_days']}일)입니다 — "
                "ARS 가 없는 기간에 CAI 를 대체 지표로 쓰는 근거입니다."
            )

    # ── VIP 모수 ──────────────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "VIP 가용 모수 구성",
        "왼쪽 도넛은 3조각으로 압축했다(색약 안전성 상한). 상세 연령×성별은 오른쪽 그룹 바.",
        badge="추정" if estimated else "",
    )
    left, right = st.columns([0.4, 0.6])
    seg = vip.segment_table(demo)
    with left:
        if seg.empty:
            C.empty_state("성별·연령 데이터가 없습니다.")
        else:
            st.plotly_chart(charts.segment_donut(seg), width="stretch", key="ov_donut")
    with right:
        if demo.empty:
            C.empty_state("성별·연령 데이터가 없습니다.")
        else:
            st.plotly_chart(charts.age_gender_bars(demo), width="stretch", key="ov_ag")
    if not demo.empty:
        C.table_view(
            demo[["gender", "age_band", "visitors"]].rename(columns={
                "gender": "성별", "age_band": "연령대", "visitors": "방문고객수"}),
            label="성별·연령 데이터 표로 보기",
        )
        C.caveat(
            "성별·연령 데이터는 기간 집계로 제공되어 일별 분해가 불가능합니다. "
            "비율만 차용하고 일별 규모는 ARS 입장권 구매 건수에서 가져왔습니다."
        )

    # ── 캠페인 집행 캘린더 ────────────────────────────────────────────
    st.markdown("")
    C.section_header(
        "VIP 타겟 지수 상위일 = 캠페인 집행 캘린더",
        "유입·실증 소비·미달 기회를 합산한 우선순위. 이미 잘 파는 날보다 "
        "'유입은 많은데 아직 덜 판 날'이 높게 나온다 — 오른쪽 두 열이 그 차이다.",
        badge=fs.period_badge,
    )
    calendar = vip.campaign_calendar(vts)
    diag = vip.vts_vs_inflow(vts)
    if calendar.empty:
        C.empty_state("이 구간에서는 VTS 를 계산할 수 없습니다.")
    else:
        display = calendar.assign(
            날짜=calendar["date"].dt.strftime("%Y-%m-%d"),
            요일=calendar.get("dow_name", ""),
            유입지수=calendar["inflow"].round(1),
            유입순위=calendar["inflow_rank"],
            VIP소비=calendar["vip_qty"].fillna(0).round(0),
            미달기회=calendar["headroom"].round(3),
            VTS=calendar["vts"].round(1),
            등급=calendar["vts_grade"],
            유입밖=calendar["inflow_only_miss"],
            산출=calendar["source"],
        )[["날짜", "요일", "유입지수", "유입순위", "VIP소비", "미달기회",
           "VTS", "등급", "유입밖", "산출"]]
        st.dataframe(
            display, width="stretch", hide_index=True,
            column_config={
                "VTS": st.column_config.ProgressColumn(
                    "VTS", min_value=0, max_value=100, format="%.1f"),
                "VIP소비": st.column_config.NumberColumn("VIP 소비", format="%d"),
                "미달기회": st.column_config.NumberColumn("미달 기회", format="%.3f"),
                "유입지수": st.column_config.NumberColumn("유입 지수", format="%.1f"),
                "유입순위": st.column_config.NumberColumn(
                    "유입 순위", format="%d위",
                    help="유입 지수만으로 줄 세웠을 때의 순위"),
                "유입밖": st.column_config.CheckboxColumn(
                    "유입만 봤다면 놓침",
                    help=f"유입 상위 {diag['top_n']}일에는 들어오지 않는 날"),
            },
        )

    # VTS 가 유입 지수와 얼마나 다른지는 숨길 게 아니라 화면에 띄울 값이다.
    # 겹침이 N/N 이면 이 표를 볼 이유가 없다는 뜻이고, 사용자가 그걸 알아야 한다.
    if not calendar.empty:
        if diag["new_days"]:
            days = " · ".join(f"{d:%m-%d}" for d in diag["new_days"])
            C.insight_box(
                [f"상위 {diag['top_n']}일 중 {len(diag['new_days'])}일({days})은 "
                 "유입 지수만 봤다면 놓쳤을 날이다 — 유입은 중위권이지만 유입 대비 "
                 "판매가 덜 된 날이라 교차판매 여지가 크다.",
                 f"나머지 {diag['overlap']}일은 유입 상위일과 같다. VTS 와 유입 지수의 "
                 f"상관은 r={diag['corr']:.3f}로, 완전히 독립된 지표가 아니라 "
                 "'유입이 큰 날 중에서 아직 덜 판 날'을 앞으로 당기는 지수다."],
                title="유입만 봤다면 놓쳤을 날",
            )
        else:
            C.caveat(
                f"이 구간에서는 VTS 상위 {diag['top_n']}일이 유입 상위 "
                f"{diag['top_n']}일과 완전히 같습니다(r={diag['corr']:.3f}). "
                "판매 데이터가 없어 '미달 기회'·'실증 소비' 축이 상수로 떨어지면 "
                "VTS 는 유입 지수와 같아집니다 — 이 구간에서는 유입 지수만 보셔도 됩니다."
            )
    if "base_signal" in vts.columns and vts["base_signal"].iloc[0] != "VRB":
        C.caveat(
            "VIP 모수(VRB) 축은 이 지수에서 자동으로 빠졌습니다. 성별·연령이 기간 "
            "집계로만 제공되어 VIP 비율이 상수이고, 그래서 VRB 는 유입 건수의 상수배"
            f"(실측 r={stats.pearson(vts['inflow'], vts['vrb']):.4f})라 유입 축과 같은 "
            "신호이기 때문입니다. 두 축을 다 쓰면 유입에 가중치를 두 번 주는 셈이라, "
            "그 몫은 '미달 기회'로 넘겼습니다. 일자별 성별·연령 데이터가 들어오면 "
            "이 축은 자동으로 복구됩니다."
        )

    # ── 자동 인사이트 ─────────────────────────────────────────────────
    lines: list[str] = []
    if not calendar.empty:
        best = calendar.iloc[0]
        lines.append(
            f"집행 우선순위 1위는 {best['date']:%Y-%m-%d}"
            f"({best.get('dow_name', '')}요일) — VTS {best['vts']:.1f}, "
            f"유입 지수 {best['inflow']:.1f}(유입 {int(best['inflow_rank'])}위)."
        )
    dow = inflow_mod.dow_profile_table(ars, sales)
    if not dow.empty:
        bits = []
        for _, row in dow.iterrows():
            peak = max(S.DOW_NAMES, key=lambda d: (row[d] if pd.notna(row[d]) else -1))
            bits.append(f"{row['label']} {peak}요일({row[peak]:.2f})")
        lines.append("요일 피크: " + " · ".join(bits) +
                     " — 유입 피크와 채널 피크가 어긋나는 곳이 교차판매 여지다.")
    cross = stats.channel_cross_corr(sales)
    if not cross.empty:
        weakest = cross.loc[cross["pearson"].idxmin()]
        lines.append(
            f"가장 느슨하게 연결된 조합은 {weakest['a_label']} ↔ {weakest['b_label']}"
            f" (r={weakest['pearson']:.3f}) — 아직 같이 팔리지 않으니 기회가 가장 크다."
        )
    C.insight_box(lines)
    C.caveat(
        "고마진 지표는 단가 데이터가 제공되지 않아 상품명 티어(프리미엄/선물번들)와 "
        "판매 희소도를 결합한 프록시입니다."
    )
