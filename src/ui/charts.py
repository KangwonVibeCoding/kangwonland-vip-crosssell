"""plotly figure 팩토리.

시각화 규칙을 코드로 강제하는 층이다:
  · `color_discrete_map` 을 항상 명시 전달 → 필터로 계열이 줄어도 색이 안 바뀐다
  · 이중 축을 만들 수 있는 함수를 아예 두지 않는다 (2단 서브플롯으로 대체)
  · 시퀀셜은 SEQ_BLUE, 발산은 래그 상관에만
  · 계열 ≥2 면 legend, 저대비 색에는 직접 라벨
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import settings as S
from src.ui import theme

BAR_RADIUS = 4          # 데이터 엔드 라운딩
LINE_WIDTH = 2
MARKER_SIZE = 8

# 2열로 나란히 서는 차트의 공통 높이.
# 아래끝이 안 맞으면 한쪽만 삐져나와 두 칸이 짝으로 안 읽힌다. 다만 **낮은 쪽에
# 맞추면 안 된다** — 특산품 20종을 지도의 430px 에 넣으면 막대 하나가 17px 로
# 눌린다. 항목이 가장 많은 쪽(특산품 20종 = 26×20+90 ≈ 610)에 가깝게 잡아
# 양쪽 다 숨통을 준다.
PAIR_HEIGHT = 560


def _hover(*rows: str) -> str:
    return "<br>".join(rows) + "<extra></extra>"


def inflow_stack(inflow: pd.DataFrame, *, value_col: str = "tickets",
                 value_label: str = "입장권 구매 건수") -> go.Figure:
    """유입 추이 — x축을 공유하는 세로 2단.

    ⚠ 이중 축(두 개의 y 스케일)을 쓰지 않는다. 건수(수천 단위)와 지수(0~100)를
    한 축에 겹치면 둘 중 하나가 바닥에 눕고, 두 축을 만들면 교차점이 임의로
    생겨 없는 상관을 읽게 만든다. 2단 스택이 정직하다.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
        subplot_titles=(value_label, "유입 강도 지수 (0~100)"),
    )
    has_value = value_col in inflow.columns and inflow[value_col].notna().any()

    if has_value:
        fig.add_trace(
            go.Scatter(
                x=inflow["date"], y=inflow[value_col], name=value_label,
                mode="lines", line=dict(color=theme.SERIES[0], width=LINE_WIDTH),
                hovertemplate=_hover("%{x|%Y-%m-%d}", f"{value_label} %{{y:,.0f}}"),
            ),
            row=1, col=1,
        )
        ma = inflow[value_col].rolling(7, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=inflow["date"], y=ma, name="7일 이동평균", mode="lines",
                line=dict(color=theme.SERIES[0], width=LINE_WIDTH, dash="dot"),
                opacity=0.65,
                hovertemplate=_hover("%{x|%Y-%m-%d}", "7일 평균 %{y:,.0f}"),
            ),
            row=1, col=1,
        )
        # 정원 소진일 표시 — 당첨자가 캡에 닿은 날은 수요가 공급을 넘은 날이다
        if "is_capped" in inflow.columns:
            capped = inflow.loc[inflow["is_capped"].fillna(False).astype(bool)]
            if not capped.empty:
                fig.add_trace(
                    go.Scatter(
                        x=capped["date"], y=capped[value_col], mode="markers",
                        name="정원 소진일",
                        marker=dict(color=theme.STATUS["critical"], size=MARKER_SIZE,
                                    line=dict(color=theme.INK["surface"], width=2)),
                        hovertemplate=_hover("%{x|%Y-%m-%d}", "정원 소진"),
                    ),
                    row=1, col=1,
                )

    row2 = 2 if has_value else 1
    # 등급 구간 배경 밴드 (연회색) — 후퇴색으로만
    for lo, hi in ((75, 100), (25, 50)):
        fig.add_hrect(y0=lo, y1=hi, row=row2, col=1,
                      fillcolor=theme.INK["grid"], opacity=0.35,
                      line_width=0, layer="below")
    fig.add_trace(
        go.Scatter(
            x=inflow["date"], y=inflow["inflow"], name="유입 강도",
            mode="lines", line=dict(color=theme.SERIES[2], width=LINE_WIDTH),
            customdata=np.stack([
                inflow.get("grade", pd.Series([""] * len(inflow))).astype(str),
                inflow.get("source", pd.Series([""] * len(inflow))).astype(str),
            ], axis=-1),
            hovertemplate=_hover("%{x|%Y-%m-%d}", "지수 %{y:.1f}",
                                 "등급 %{customdata[0]}", "산출 %{customdata[1]}"),
        ),
        row=row2, col=1,
    )
    fig.update_yaxes(range=[0, 100], row=row2, col=1)
    fig.update_layout(height=460, hovermode="x unified",
                      margin=dict(l=8, r=8, t=64, b=8))
    for ann in fig.layout.annotations:
        ann.font.size = 12
        ann.font.color = theme.INK["secondary"]
        ann.x = 0
        ann.xanchor = "left"
    return fig


def corr_bars(head: dict[str, float]) -> go.Figure:
    """채널별 상관계수 수평 바 — 가설 검증 배너의 시각 요소."""
    rows = [(S.CHANNEL_LABEL[ch], v) for ch, v in head.items() if pd.notna(v)]
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [theme.CHANNEL_COLOR_BY_LABEL.get(l, theme.SERIES[0]) for l in labels]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker=dict(color=colors),
            text=[f"{v:.3f}" for v in values], textposition="outside",
            textfont=dict(color=theme.INK["secondary"], size=12),
            hovertemplate=_hover("%{y}", "Pearson r %{x:.3f}"),
        )
    )
    fig.update_layout(
        height=34 * max(len(labels), 1) + 70, showlegend=False,
        xaxis=dict(range=[0, 1.08], title="Pearson 상관계수"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=8, r=48, t=8, b=32),
    )
    fig.update_traces(marker_cornerradius=BAR_RADIUS)
    return fig


def confound_dumbbell(table: pd.DataFrame) -> go.Figure:
    """요일 교란 통제 — 원 상관에서 편상관으로 이동한 거리를 채널마다 하나의 선으로.

    두 값을 나란한 막대 두 개로 그리면 "둘 다 크다"로 읽힌다. 여기서 보여야 할
    것은 크기가 아니라 **얼마나 깎였고 순서가 어떻게 뒤집혔는가**이므로, 이동을
    선분으로 그리고 도착점(편상관)에 부트스트랩 CI 를 붙인다.

    입력은 `stats.confound_table()` 의 반환(편상관 내림차순)이다. 정렬을 그대로
    쓴다 — y축 위에서 아래로가 곧 통제 후 우선순위다.
    """
    df = table.dropna(subset=["raw", "partial"]).iloc[::-1]   # 수평 축은 아래→위
    labels = df["channel_label"].tolist()
    colors = [theme.CHANNEL_COLOR_BY_LABEL.get(l, theme.SERIES[0]) for l in labels]

    fig = go.Figure()
    for label, color, raw, partial in zip(labels, colors, df["raw"], df["partial"]):
        fig.add_trace(go.Scatter(
            x=[raw, partial], y=[label, label], mode="lines",
            line=dict(color=color, width=3), opacity=0.35,
            hoverinfo="skip", showlegend=False,
        ))
    # ⚠ 범례는 **모양**만 설명한다. 점의 색은 채널 아이덴티티(파랑=카지노 식음,
    # 주황=룸서비스, 초록=지역특산품)라 범례 한 칸에 담기지 않는다. 그런데
    # `marker.color` 에 리스트를 주면 plotly 는 **리스트의 첫 색만** 범례에
    # 그린다 — 그래서 범례가 "요일 통제 후 = 초록"이라고 말하는데 화면의 점은
    # 주황·파랑·초록 3색인 상태였다. 범례가 틀린 것은 색을 잘못 쓴 것보다 나쁘다.
    #
    # 실제 점 트레이스는 범례에서 빼고, 중립색 더미 두 개로 모양만 설명한다.
    for name, marker in (
        ("원 상관", dict(size=MARKER_SIZE + 2, color=theme.INK["surface"],
                        line=dict(color=theme.INK["muted"], width=2))),
        ("요일 통제 후", dict(size=MARKER_SIZE + 4, color=theme.INK["secondary"])),
    ):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=name,
            marker=marker, hoverinfo="skip", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=df["raw"], y=labels, mode="markers", name="원 상관",
        showlegend=False,
        marker=dict(size=MARKER_SIZE + 2, color=theme.INK["surface"],
                    line=dict(color=theme.INK["muted"], width=2)),
        hovertemplate=_hover("%{y}", "원 상관 r %{x:.3f}"),
    ))
    fig.add_trace(go.Scatter(
        x=df["partial"], y=labels, mode="markers+text", name="요일 통제 후",
        showlegend=False,
        marker=dict(size=MARKER_SIZE + 4, color=colors),
        error_x=dict(
            type="data", symmetric=False,
            array=(df["ci_hi"] - df["partial"]).to_numpy(),
            arrayminus=(df["partial"] - df["ci_lo"]).to_numpy(),
            color=theme.INK["axis"], thickness=1.5, width=5,
        ),
        text=[f"{v:.3f}" for v in df["partial"]],
        textposition="bottom center",
        textfont=dict(color=theme.INK["secondary"], size=11),
        customdata=np.column_stack([df["p"].to_numpy(), df["ci_lo"].to_numpy(),
                                    df["ci_hi"].to_numpy()]),
        hovertemplate=_hover("%{y}", "편상관 r %{x:.3f}",
                             "95% CI [%{customdata[1]:.3f}, %{customdata[2]:.3f}]",
                             "p %{customdata[0]:.4f}"),
    ))
    fig.add_vline(x=0, line=dict(color=theme.INK["axis"], width=1, dash="dot"))
    fig.update_layout(
        height=64 * max(len(labels), 1) + 110,
        xaxis=dict(range=[-0.35, 1.05], title="Pearson 상관계수 (0 = 무관)"),
        yaxis=dict(title=""),
        margin=dict(l=8, r=32, t=8, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def segment_donut(seg: pd.DataFrame) -> go.Figure:
    """VIP 모수 구성 도넛.

    ⚠ 3조각만 쓴다. 도넛은 모든 조각이 동시에 보이므로 색약 안전성이 전체쌍
    기준으로 검증돼야 하는데, 검증기에서 통과한 것은 앞 3색까지다. 상세
    연령×성별은 옆의 그룹 바가 담당한다.
    """
    agg = seg.groupby("segment", as_index=False)["visitors"].sum()
    order = ["VIP(40~60대)", "준VIP(30대)", "기타"]
    agg["_o"] = agg["segment"].map({k: i for i, k in enumerate(order)}).fillna(9)
    agg = agg.sort_values("_o")
    total = float(agg["visitors"].sum()) or 1.0
    vip_share = float(
        agg.loc[agg["segment"] == "VIP(40~60대)", "visitors"].sum()
    ) / total

    fig = go.Figure(
        go.Pie(
            labels=agg["segment"], values=agg["visitors"], hole=0.62,
            sort=False, direction="clockwise",
            marker=dict(
                colors=[theme.SEGMENT_COLOR.get(s, theme.INK["muted"])
                        for s in agg["segment"]],
                line=dict(color=theme.INK["surface"], width=2),   # 2px 표면 간격
            ),
            textinfo="percent", textposition="inside",
            texttemplate="%{percent:.1%}",
            insidetextfont=dict(color="#ffffff", size=12),
            hovertemplate=_hover("%{label}", "%{value:,}명", "%{percent:.1%}"),
        )
    )
    fig.add_annotation(
        text=f"<b>{vip_share * 100:.1f}%</b><br><span style='font-size:11px'>VIP 비중</span>",
        showarrow=False, font=dict(size=26, color=theme.INK["primary"]),
    )
    fig.update_layout(height=300, showlegend=True,
                      margin=dict(l=8, r=8, t=8, b=8))
    return fig


def age_gender_bars(demo: pd.DataFrame) -> go.Figure:
    """연령대 × 성별 그룹 바. 성별 2계열이므로 legend + 직접 값 라벨."""
    order = [b for b in S.AGE_BANDS_ALL if b in set(demo["age_band"])]
    extra = [b for b in demo["age_band"].unique() if b not in order]
    order = order + sorted(extra)
    fig = px.bar(
        demo, x="visitors", y="age_band", color="gender", barmode="group",
        orientation="h", text="visitors",
        color_discrete_map=theme.GENDER_COLOR,       # 엔티티 고정
        category_orders={"age_band": order, "gender": ["남", "여"]},
        labels={"visitors": "방문고객수", "age_band": "연령대", "gender": "성별"},
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="outside",
        textfont=dict(color=theme.INK["secondary"], size=11),
        marker_cornerradius=BAR_RADIUS,
        marker_line=dict(color=theme.INK["surface"], width=2),   # 인접 바 2px 간격
        hovertemplate=_hover("%{y} · %{fullData.name}", "%{x:,}명"),
    )
    fig.update_layout(height=300, hovermode="closest",
                      yaxis=dict(autorange="reversed"),
                      margin=dict(l=8, r=56, t=36, b=8))
    return fig


def dow_month_heatmap(pivot: pd.DataFrame, *, subtitle: str = "") -> go.Figure:
    """요일 × 월 수요 히트맵 (시퀀셜 단일 청색 램프).

    원본에 주문 시각(hour)이 없어 원래 계획한 심야 피크 분석을 대체하는 축이다.
    주간 패턴과 계절성을 한 장에서 읽을 수 있다.
    """
    z = pivot.to_numpy(dtype="float64")
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(pivot.columns), y=list(pivot.index),
            colorscale=theme.SEQ_BLUE_SCALE,
            hovertemplate=_hover("%{y} %{x}요일", "평균 %{z:,.0f}개"),
            colorbar=dict(title=dict(text="평균<br>판매량", side="right"),
                          thickness=12, len=0.85,
                          tickfont=dict(color=theme.INK["muted"], size=10)),
            xgap=2, ygap=2,       # 셀 사이 2px 표면 간격
        )
    )
    # 최대 셀에 직접 라벨 — 색상만으로 최고점을 읽게 하지 않는다
    if np.isfinite(z).any():
        idx = np.unravel_index(np.nanargmax(z), z.shape)
        fig.add_annotation(
            x=list(pivot.columns)[idx[1]], y=list(pivot.index)[idx[0]],
            text=f"<b>{z[idx]:,.0f}</b>", showarrow=False,
            font=dict(color="#ffffff", size=11),
        )
    # 행이 1개(31일 구간)든 12개(2년 구간)든 셀이 납작해지거나 늘어나지 않게
    # 행 수에 비례시킨다. 예전엔 430 고정이라 한 달짜리 창에서 셀 하나가
    # 세로로 400px 가까이 늘어났다.
    height = max(46 * len(pivot.index) + 120, 200)
    # ⚠ x축이 `side="top"` 이라 눈금 라벨이 그림 위쪽을 쓴다. 부제를 y=1.10 에
    # 두고 위 여백을 40 만 주면 **요일 라벨과 부제가 같은 자리에 겹쳐 찍힌다**
    # (실제로 그랬다). 부제가 있을 때는 여백을 늘리고 라벨 위로 밀어 올린다.
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=76 if subtitle else 8, b=8),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False, side="top"),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle, xref="paper", yref="paper", x=0, y=1,
            yshift=52, yanchor="bottom",
            showarrow=False, xanchor="left",
            font=dict(size=11, color=theme.INK["muted"]),
        )
    return fig


def top_items_bar(items: pd.DataFrame, *, value_col: str = "total_qty",
                  value_label: str = "총 판매수량",
                  color_by_tier: bool = True,
                  height: int | None = None) -> go.Figure:
    """상위 상품 수평 바. 티어 3색 또는 단일색 + 값 직접 라벨.

    `height` 를 주면 자동 높이(항목 수 비례) 대신 그 값을 쓴다. 이 차트가 지도처럼
    **높이가 고정된 요소와 2열로 나란히 설 때** 아래끝을 맞추기 위한 것이다 —
    맞추지 않으면 한쪽만 길게 삐져나와 두 칸이 짝으로 안 읽힌다.
    """
    df = items.iloc[::-1]        # 수평 바는 아래에서 위로 그려진다
    if color_by_tier and "tier_label" in df.columns:
        fig = px.bar(
            df, x=value_col, y="item", orientation="h", text=value_col,
            color="tier_label", color_discrete_map=theme.TIER_COLOR_BY_LABEL,
            category_orders={"tier_label": [S.TIER_LABEL[t] for t in S.TIERS]},
            labels={value_col: value_label, "item": "", "tier_label": "티어"},
        )
    else:
        fig = px.bar(df, x=value_col, y="item", orientation="h", text=value_col,
                     labels={value_col: value_label, "item": ""})
        fig.update_traces(marker_color=theme.SERIES[0])

    is_float = df[value_col].dtype.kind == "f"
    fig.update_traces(
        texttemplate="%{text:,.1f}" if is_float else "%{text:,}",
        textposition="outside",
        textfont=dict(color=theme.INK["secondary"], size=11),
        marker_cornerradius=BAR_RADIUS,
        marker_line=dict(color=theme.INK["surface"], width=2),
        hovertemplate=_hover("%{y}", f"{value_label} %{{x:,}}"),
    )
    # ⚠ x축 상한을 직접 늘린다. `textposition="outside"` 라벨은 막대 **밖**에
    # 그려지는데, 축이 데이터 최댓값에 딱 맞으면 1위 막대의 라벨이 플롯 경계에서
    # 잘린다 (실측: "3,461" 이 "3," 로 잘려 나왔다). 오른쪽 여백(r=76)은 축
    # 바깥 영역이라 이 잘림을 막지 못한다 — 축 자체에 여유를 줘야 한다.
    vmax = float(pd.to_numeric(df[value_col], errors="coerce").max() or 0)
    fig.update_layout(
        height=height or max(26 * len(df) + 90, 220), hovermode="closest",
        margin=dict(l=8, r=76, t=36, b=8), bargap=0.25,
        xaxis=dict(range=[0, vmax * 1.18]) if vmax > 0 else {},
    )
    return fig


def venue_treemap(frame: pd.DataFrame) -> go.Figure:
    """영업장 → 상품 트리맵. 크기=수량, 색=수량(시퀀셜).

    ⚠ 색을 카테고리로 쓰지 않는다. 트리맵은 모든 조각이 동시에 보여 전체쌍
    색약 검증을 통과해야 하는데, 상품 수십 개에 카테고리 색을 배정하면 불가능하다.
    """
    fig = px.treemap(
        frame, path=[px.Constant("전체"), "venue", "item"], values="qty",
        color="qty", color_continuous_scale=theme.SEQ_BLUE,
        custom_data=["tier_label"],
    )
    fig.update_traces(
        marker=dict(cornerradius=BAR_RADIUS,
                    line=dict(color=theme.INK["surface"], width=2)),
        texttemplate="%{label}<br>%{value:,}",
        textfont=dict(size=11),
        hovertemplate=_hover("%{label}", "판매수량 %{value:,}", "티어 %{customdata[0]}"),
    )
    fig.update_layout(
        height=430, margin=dict(l=4, r=4, t=8, b=4),
        coloraxis_colorbar=dict(title=dict(text="판매량", side="right"),
                                thickness=12, len=0.85,
                                tickfont=dict(color=theme.INK["muted"], size=10)),
    )
    return fig


def lag_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """래그 상관 히트맵 — 발산 팔레트를 쓰는 유일한 차트.

    D+2 상관이 −0.35 까지 내려가므로 부호가 실제 의미를 갖는다. 중앙(0)은
    회색 중립이고, 채널별 최댓값 셀에 링을 둘러 최적 시차를 색상 없이도 읽게 한다.
    """
    pivot = matrix.pivot_table(index="channel_label", columns="lag",
                               values="pearson", aggfunc="first")
    order = [S.CHANNEL_LABEL[c] for c in S.CHANNELS if S.CHANNEL_LABEL[c] in pivot.index]
    pivot = pivot.reindex(order)
    z = pivot.to_numpy(dtype="float64")
    x_labels = [f"D+{c}" for c in pivot.columns]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=x_labels, y=list(pivot.index),
            colorscale=theme.DIVERGING_SCALE, zmid=0, zmin=-1, zmax=1,
            text=[[f"{v:+.3f}" if np.isfinite(v) else "" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=12),
            hovertemplate=_hover("%{y}", "시차 %{x}", "Pearson r %{z:+.3f}"),
            colorbar=dict(title=dict(text="Pearson r", side="right"),
                          thickness=12, len=0.85,
                          tickfont=dict(color=theme.INK["muted"], size=10)),
            xgap=2, ygap=2,
        )
    )
    # 채널별 최적 시차에 링 강조 (색상 외 2차 인코딩)
    for r, channel in enumerate(pivot.index):
        row = z[r]
        if not np.isfinite(row).any():
            continue
        c = int(np.nanargmax(row))
        fig.add_shape(
            type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
            line=dict(color=theme.INK["primary"], width=2.5), fillcolor="rgba(0,0,0,0)",
        )
    fig.update_layout(
        height=250, margin=dict(l=8, r=8, t=8, b=8),
        yaxis=dict(autorange="reversed", showgrid=False),
        xaxis=dict(showgrid=False, side="top"),
    )
    return fig


def dow_profile_lines(table: pd.DataFrame) -> go.Figure:
    """요일 인덱스 다계열 라인 — ARS 유입 + 3채널.

    4계열이므로 legend + 각 계열 최댓값 지점에 직접 라벨을 둔다.
    """
    fig = go.Figure()
    for _, row in table.iterrows():
        label = str(row["label"])
        color = theme.SERIES_COLOR_BY_LABEL.get(label, theme.INK["muted"])
        values = [row[d] for d in S.DOW_NAMES]
        fig.add_trace(go.Scatter(
            x=list(S.DOW_NAMES), y=values, name=label, mode="lines+markers",
            line=dict(color=color, width=LINE_WIDTH),
            marker=dict(size=MARKER_SIZE, color=color,
                        line=dict(color=theme.INK["surface"], width=2)),
            hovertemplate=_hover(f"{label}", "%{x}요일", "지수 %{y:.2f}"),
        ))
        finite = [v for v in values if pd.notna(v)]
        if finite:
            peak = int(np.nanargmax(np.array(values, dtype="float64")))
            fig.add_annotation(
                x=S.DOW_NAMES[peak], y=values[peak], text=f"{values[peak]:.2f}",
                showarrow=False, yshift=16,
                font=dict(size=11, color=theme.INK["secondary"]),
            )
    fig.add_hline(y=1.0, line=dict(color=theme.INK["axis"], width=1, dash="dash"))
    fig.update_layout(
        height=340, hovermode="x unified",
        yaxis=dict(title="요일 인덱스 (전체 평균 = 1.0)"),
        margin=dict(l=8, r=8, t=44, b=8),
    )
    return fig


def month_profile_lines(table: pd.DataFrame) -> go.Figure:
    """월 인덱스 다계열 라인 — 채널별 계절성."""
    months = [m for m in range(1, 13) if m in table.columns]
    fig = go.Figure()
    for _, row in table.iterrows():
        label = str(row["label"])
        color = theme.SERIES_COLOR_BY_LABEL.get(label, theme.INK["muted"])
        values = [row[m] for m in months]
        fig.add_trace(go.Scatter(
            x=[f"{m}월" for m in months], y=values, name=label,
            mode="lines+markers", line=dict(color=color, width=LINE_WIDTH),
            marker=dict(size=7, color=color,
                        line=dict(color=theme.INK["surface"], width=2)),
            hovertemplate=_hover(f"{label}", "%{x}", "지수 %{y:.2f}"),
        ))
        finite = np.array(values, dtype="float64")
        if np.isfinite(finite).any():
            peak = int(np.nanargmax(finite))
            fig.add_annotation(
                x=f"{months[peak]}월", y=values[peak], text=f"{values[peak]:.2f}",
                showarrow=False, yshift=16,
                font=dict(size=11, color=theme.INK["secondary"]),
            )
    fig.add_hline(y=1.0, line=dict(color=theme.INK["axis"], width=1, dash="dash"))
    fig.update_layout(height=340, hovermode="x unified",
                      yaxis=dict(title="월 인덱스 (전체 평균 = 1.0)"),
                      margin=dict(l=8, r=8, t=44, b=8))
    return fig


def season_line(daily: pd.Series, *, highlight: tuple[int, ...] = (9, 10),
                label: str = "판매수량") -> go.Figure:
    """월별 추이 + 시즌 강조 밴드 (특산품 명절 선물 시즌 등)."""
    if daily.empty:
        return go.Figure()
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(daily.index)),
                          "qty": daily.to_numpy()})
    monthly = frame.groupby(frame["date"].dt.month, as_index=False)["qty"].sum()
    monthly.columns = ["month", "qty"]
    fig = go.Figure(go.Scatter(
        x=[f"{m}월" for m in monthly["month"]], y=monthly["qty"],
        mode="lines+markers", name=label,
        line=dict(color=theme.SERIES[2], width=LINE_WIDTH),
        marker=dict(size=MARKER_SIZE, color=theme.SERIES[2],
                    line=dict(color=theme.INK["surface"], width=2)),
        hovertemplate=_hover("%{x}", f"{label} %{{y:,}}"),
    ))
    present = [m for m in highlight if m in set(monthly["month"])]
    if present:
        fig.add_vrect(
            x0=f"{min(present)}월", x1=f"{max(present)}월",
            fillcolor=theme.INK["grid"], opacity=0.45, line_width=0, layer="below",
            annotation_text="명절 선물 시즌", annotation_position="top left",
            annotation=dict(font=dict(size=11, color=theme.INK["muted"])),
        )
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=36, b=8),
                      yaxis=dict(title=label))
    return fig
