"""AI VIP 큐레이션 추천 JSON 프로토타입 — 기존 분석 함수만 조합한 단독 실행 스크립트.

무엇을 만드는가:
  기준일 하루에 대해 세그먼트 페르소나 1명분의 상품 추천 3슬롯(Main / Sub /
  Cross-sell)을 만들어 stdout 과 JSON 파일로 뽑는다. 앱(app.py)과는 무관하며
  이 스크립트는 저장소의 어떤 파일도 고치지 않는다.

⚠ 이 저장소에는 개인 단위 데이터가 한 건도 없다:
  고객ID·영수증·장바구니·검색어·단가 컬럼이 전부 없고, 판매의 최소 입도는
  `일자 × 영업장 × 상품 × 수량` 이다. 따라서 여기서 말하는 "고객"은 개인이 아니라
  **성별×연령 세그먼트 페르소나**이고 `customer_id` 는 `SEG-...` 로 시작한다.
  개인 행동(직전 구매 이력·동선·체류시간)을 지어내면 근거 없는 값이 발표에 올라간다.

값의 출처를 문자열로 드러낸다 — `internal_reason` 의 `[실측]/[파생]/[가정]` 라벨이
그것이다. 이 프로젝트가 `pressure_signal`·`base_signal`·`lift_measured` 로 쓰는
규약과 같다: 어느 경로로 나온 값인지 결과 자체가 말하게 둔다.

LLM 을 호출하지 않는다. 문구는 결정론적 한국어 템플릿이고, D+0/D+1 표현은
**그 채널의 실측 best_lag 로 분기**한다. 하드코딩하면 화면이 근거와 다른 말을 한다.

날짜가 추천을 바꾸는 축은 둘이다. 하나만 두면 기준일을 바꿔도 결과가 그대로다:
  · 채널 축 — `channel_grade_response()`. 유입 등급별 채널 반응 지수.
  · 상품 축 — `season_factors()`. 기준일의 요일지수 × 월지수.
편상관·래그·CSM·페르소나 선호는 전부 창 전체에서 한 번 계산되는 상수이고, 등급계수는
모든 후보에 같은 수를 곱하므로 **어느 것도 순위를 바꾸지 못한다.** 이 두 축을 빼면
10일 스윕에서 고유 추천 조합이 2개로 주저앉는다(실측 → 도입 후 10개).

사용법:
    $env:PYTHONIOENCODING="utf-8"; & ".\\.venv\\Scripts\\python.exe" scripts\\prototype_curation.py `
        --persona SEG-F50-ROOM --date 2024-12-21 --out out\\curation.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                     # noqa: E402

from config import settings as S                        # noqa: E402
from src.analysis import crosssell, lag, scoring, stats, vip   # noqa: E402
from src.analysis import inflow as inflow_mod           # noqa: E402
from src.data import loaders                            # noqa: E402

# ── 페르소나 ──────────────────────────────────────────────────────────
# customer_id 가 `SEG-` 로 시작하는 이유는 위 모듈 docstring 참고 — 개인이 아니다.
#
# ⚠ `pref_channels` 는 **가정**이다. 실측할 방법이 없다:
#    성별·연령(demo)은 2026-06-10~08-06 보유분뿐이고 판매는 2023~2024 라서
#    **겹치는 날이 0일**이다. 세그먼트별 채널 선호를 데이터로 뽑을 수 없다.
#    (같은 이유로 VTS 의 base 축이 아직 복구되지 않는다 — CLAUDE.md 의 demo 절.)
#    실측인 것은 `share`(구성비)뿐이고, 그것도 원본 `CUST_WHOL_CNT` 가 누적
#    고객 수라 **규모가 아니라 비율로만** 쓸 수 있다.
DEFAULT_PREF_CHANNELS = {S.CH_ROOM: 0.5, S.CH_LOCAL: 0.3, S.CH_CASINO: 0.2}
DEFAULT_PREF_TIERS = (S.TIER_PREMIUM, S.TIER_BUNDLE)

# 세그먼트는 12개(성별 2 × 연령 6)를 다 만든다. 채널 선호는 위 공통 가정을 쓰고,
# **세그먼트를 실제로 갈라놓는 것은 상품 단위 선호(`segment_preference.csv`)** 다.
# ID 의 `-ROOM` 접미는 "룸서비스 우위 채널선호 프로필"을 뜻한다 — 그 프로필이
# 가정이라는 사실을 이름에 남긴다.
PERSONAS: dict[str, dict] = {
    f"SEG-{code}{age[:2]}-ROOM": {
        "label": f"{age} {gender_kr}성",
        "gender": gender_kr,
        "age_band": age,
        "segment_group": f"{age} {gender_kr}",   # segment_preference.csv 의 `그룹` 값
        "pref_channels": DEFAULT_PREF_CHANNELS,
        "pref_tiers": DEFAULT_PREF_TIERS,
    }
    for code, gender_kr in (("F", "여"), ("M", "남"))
    for age in ("20대", "30대", "40대", "50대", "60대", "70대")
}

# ── 스코어 상수 ───────────────────────────────────────────────────────
CUR_WEIGHTS = {"intent": 0.50, "base": 0.30, "csm": 0.20}
INTENT_MIX = {"channel": 0.6, "item": 0.4}
# 유입 등급 → 집행 강도 계수. build_inflow 의 grade 라벨(S.GRADE_BINS)과 같은 어휘다.
# 이 계수는 **모든 후보에 똑같이 곱해지므로 순위를 바꾸지 못한다.** 오늘 얼마나
# 세게 밀지(강도)만 정하고, 무엇을 밀지는 아래 두 축이 정한다.
GRADE_FACTOR = {"피크": 1.0, "성수": 0.8, "보통": 0.6, "한산": 0.4}

# 상품 계절 계수(요일지수 × 월지수)의 상하한. 판매일수가 적은 요일·월 조합에서
# 지수가 튀는 것을 막는다 — `nrm(robust=True)` 가 5~95 분위로 클리핑하는 것과
# 같은 취지다. 클리핑이 없으면 한 상품의 특정 요일 하나가 스케일을 독점한다.
SEASON_FACTOR_CLIP = (0.5, 2.0)
LAG_PENALTY = 0.7          # 채널 best_lag 가 당일(D+0)이 아니면 당일 집행 신호를 깎는다
ITEM_SIGNAL_MIN_R = 0.10   # 이 미만이면 상품 신호로 인정하지 않고 채널 신호로 대체
RECENT_DAYS = 14           # 최근 노출 판정 창
RECENT_TOP_N = 5           # 채널별 최근 상위 N종은 후보에서 뺀다
STOCK_MIN_RATIO = 0.25     # 판매일수/창일수 — 재고 프록시
TIER_BAND = 1              # 선호 티어에서 몇 칸까지 허용할지
SUB_POOL_TOP_N = 8         # 2순위는 이 순위 안에서만 고른다
COOC_MIN_DAYS = 3          # crosssell.cooccurrence 가 표본으로 인정하는 최소 공통일

# ── 세그먼트 선호 (외부 추정표) ───────────────────────────────────────
# `data/raw/segment_preference.csv` — 하이원멤버스 **누적** 가입현황의 성별·연령
# 구성비 변화와 품목별 판매 비중 변화의 상관(r)으로 추정한 표. 없으면 이 축은
# 통째로 빠지고 나머지 축으로만 점수를 낸다(파일·네트워크 의존 0).
#
# ⚠ 이 r 은 `[실측]` 이 아니라 `[파생]` 이다. 두 가지 한계가 값에 박혀 있다:
#   1. **생태학적 추론이다.** 개인 구매 기록이 아니라 집단 비율끼리의 상관이라
#      "50대가 이걸 샀다"가 아니라 "50대 비중이 높은 시기에 이게 더 팔렸다"이다.
#   2. **원 지표가 누적 수치다.** 누적 구성비는 단조 변화라 사실상 시간의
#      대리변수이고, 그래서 r 은 연령 선호와 **상품의 시간 추세를 분리하지 못한다.**
#      (`demographics_api` 의 `CUST_WHOL_CNT` 를 방문객으로 오독한 전례와 같은 성질.)
#
# 그래서 판매 커버리지 게이트를 건다. 창의 대부분에서 팔리지 않은 상품은 r 을
# 쓰지 않는다 — 창 중간에 생기거나 사라진 상품은 상관이 선호가 아니라 **존재한
# 시기**를 재기 때문이다. 실측: 룸서비스 `(컨)` 상품군이 2023-07 에 통째로 등장해
# (2023-01~06 판매 0) 40~70대 선호 목록의 60~80%를 차지한다. 20~30대 목록에는
# 0%다. 같은 실패 모드로 `lift` 가 상품 수명주기를 재던 것을 `hi_lo_days_by_month`
# 로 고친 전례가 있다(최댓값 47.05 → 2.02).
# 게이트를 통과 못 한 상품은 가산도 감점도 없다(중립) — 없는 근거를 만들지 않는다.
# 로더의 계층 폴백과 같은 순서다 — raw(원천) → sample(커밋본). `data/raw` 는
# gitignore 되므로, 배포판에서도 쓰려면 축약본을 `data/sample` 에 넣어야 한다.
SEGMENT_PREF_FILES = (S.RAW_DIR / "segment_preference.csv",
                      S.SAMPLE_DIR / "segment_preference.csv")
SEGMENT_STRENGTH = 0.5          # base 축 배수 = 1 + 이 값 × r  (r=+0.86 → ×1.43)
SEGMENT_MIN_MONTH_RATIO = 0.9   # 창의 이 비율 이상 '월'에서 팔린 상품만 r 을 쓴다
SEGMENT_CHANNEL_KR = {"룸서비스": S.CH_ROOM, "특산품": S.CH_LOCAL}

# ⚠ 이 표를 **2023-07 이후 18개월 재계산본으로 간주한다** — 데이터 제공자의 지시다.
# 파일 자체는 2023-01~2024-12(24개월) 산출이라 `(컨)` 상품군이 없던 6개월이 섞여
# 있는데, 그 6개월을 뺀 값이라고 보고 쓰는 것이다. **실제로 재계산된 값은 아니다.**
# 커버리지 게이트도 반드시 같은 창에서 재야 앞뒤가 맞는다 — 24개월 창으로 재면
# 2023-07 에 개시한 상품(`(VIP) F1`·`(컨)` 전부)이 18/24 로 전부 탈락해, 간주하기로
# 한 전제와 게이트가 서로 모순된다.
# 이 간주가 값에 미치는 영향은 `internal_reason` 에 그대로 적힌다.
SEGMENT_ASSUMED_WINDOW = (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-12-31"))

# 3순위(티어 밴드 면제 슬롯)의 마진 프록시 하한 — 후보 풀 분위로 잡는다.
# 밴드를 면제하면 범용 대량 상품이 이긴다: 매일 대량으로 팔리는 상품은 총유입과
# 거의 자동으로 상관하고(콜라 r 0.564) CSM 물량 축(0.30)도 높게 받아, 실측에서
# `콜라`·`김`·`에비앙(330ml)` 이 3순위를 차지했다. VIP 컨시어지 문구에 「콜라」가
# 붙으면 화면이 스스로 격을 깎는다.
#   ⚠ CSM 가중을 올리는 안은 **반대로 뒤집힌다** — CSM 의 물량 축이 마진 축(0.25)
#     보다 커서 재가중하면 `에비앙(330ml)`·`(22)임팩트` 가 오히려 상위로 온다(실측).
#   ⚠ 티어 밴드를 '두 칸'으로 넓히는 안은 성립하지 않는다 — 티어가 3종뿐이라
#     |Δ|≤2 는 전 티어 허용과 같다.
# 절대값이 아니라 분위인 이유는 `sample_gate` 와 같다: 후보 풀 구성이 바뀌어도
# "이 풀에서 마진이 중간 이상"이라는 의미가 유지된다.
CROSS_MARGIN_QUANTILE = 0.5

# 티어 서열 standard < bundle < premium. 밴드 계산에만 쓴다.
TIER_ORD = {S.TIER_STANDARD: 0, S.TIER_BUNDLE: 1, S.TIER_PREMIUM: 2}


# ── 신호 만들기 ───────────────────────────────────────────────────────

def channel_grade_response(sales: pd.DataFrame,
                           inflow: pd.DataFrame) -> dict[str, dict[str, float]]:
    """채널이 유입 등급에 실제로 얼마나 반응하는가 — 등급계수를 채널별로 주는 근거.

    I(채널, 등급) = 그 등급 날들의 평균 일판매량 / 그 채널 전체 평균

    등급계수를 **채널 공통 스칼라 하나로 두면 순위가 절대 안 바뀐다** — 모든
    후보에 같은 수를 곱하는 것이기 때문이다. 채널마다 반응 폭이 다르다는 것은
    실측이다(731일 창: 룸서비스 피크 1.468 / 한산 0.687 = 2.14배, 특산품은
    1.193 / 0.869 = 1.37배로 가장 둔감). 이 격차를 계수에 실어야 "붐비는 날은
    어느 채널을 미는가"가 날짜에 따라 달라진다.
    """
    if inflow.empty or sales.empty:
        return {}
    grade_by_date = inflow.set_index("date")["grade"]
    out: dict[str, dict[str, float]] = {}
    for channel in S.CHANNELS:
        series = stats.daily_qty(sales, channel)
        if series.empty:
            continue
        base = float(series.mean())
        if not base:
            continue
        idx = series.groupby(grade_by_date.reindex(series.index)).mean() / base
        for grade, value in idx.items():
            out.setdefault(str(grade), {})[channel] = float(value)
    return out


def season_factors(sales: pd.DataFrame, item_ids: pd.Series,
                   as_of: pd.Timestamp) -> pd.DataFrame:
    """상품 × 기준일의 요일지수 × 월지수 — **날짜가 채널 안의 순위를 바꾸는 유일한 축.**

    편상관·래그·CSM·페르소나 선호는 전부 창 전체에서 한 번 계산되는 상수라,
    날짜를 바꿔도 후보 순서가 그대로다(실측: 10일 스윕에 고유 추천 조합 2개).
    등급계수는 채널당 스칼라 하나뿐이라 채널 3개의 순서만 바꿀 수 있고 채널
    **안에서는** 아무것도 못 바꾼다. 상품마다 요일·월 프로파일이 다르다는 사실이
    유일하게 남은 자유도다 — 실측 변동폭이 작지 않다((VIP) F1 요일 0.52~1.78 ·
    월 0.78~1.24 → 결합 0.40~2.20배).

    `stats.dow_index` / `stats.month_index` 를 그대로 쓴다. 판매가 있는 날만
    평균 내는 그 규약을 여기서 다시 정의하면 탭3 의 요일 프로파일과 값이 갈린다.
    """
    cols = ["item_id", "dow_idx", "month_idx", "season"]
    ids = list(dict.fromkeys(item_ids.tolist()))
    if not ids:
        return pd.DataFrame(columns=cols)

    dow, month = int(as_of.dayofweek), int(as_of.month)
    daily = (sales.loc[sales["item_id"].isin(ids)]
             .groupby(["item_id", "date"])["qty"].sum())
    rows: list[dict] = []
    for item_id in ids:
        try:
            series = daily.loc[item_id]
        except KeyError:
            continue
        d = float(stats.dow_index(series).get(dow, float("nan")))
        m = float(stats.month_index(series).get(month, float("nan")))
        # 지수가 없는(그 요일·그 달에 판매 이력이 없는) 상품은 1.0 으로 둔다 —
        # 0 으로 두면 "그날 안 팔린다"가 아니라 "근거 없음"이 감점이 된다.
        d = d if pd.notna(d) else 1.0
        m = m if pd.notna(m) else 1.0
        rows.append({"item_id": item_id, "dow_idx": d, "month_idx": m,
                     "season": min(max(d * m, SEASON_FACTOR_CLIP[0]),
                                   SEASON_FACTOR_CLIP[1])})
    return pd.DataFrame(rows, columns=cols)


def load_segment_preference() -> pd.DataFrame:
    """세그먼트 선호 추정표를 읽는다. 없으면 **빈 프레임** — 축이 통째로 빠진다.

    반환: group / channel / item / r  (선호는 r>0, 비선호는 r<0 로 한 열에 합친다)

    비선호를 버리지 않는 이유: 구성비 합이 1로 묶여 있어 비선호는 선호의 거울상
    이지만, **추천을 실제로 바꾸는 쪽이 비선호다.** 실측 예 — `(VIP) F1` 은
    20대 선호 1위(r=+0.71)이면서 50대 여 비선호(r=-0.56)인데, 선호만 쓰면 50대
    여에게 가산점이 0일 뿐이라 CSM 70.6·프리미엄·채널선호 0.5 로 여전히 1순위를
    지킨다. 즉 20대 취향 상품이 50대에게 계속 추천된다.
    """
    cols = ["group", "channel", "item", "r"]
    path = next((p for p in SEGMENT_PREF_FILES if p.exists()), None)
    if path is None:
        return pd.DataFrame(columns=cols)
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001  읽기 실패는 축 부재와 같게 다룬다
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for rec in raw.itertuples(index=False):
        channel = SEGMENT_CHANNEL_KR.get(str(rec[0]).strip())
        if channel is None:
            continue
        for col in (2, 3):                      # 선호 / 비선호 열
            if col >= len(rec):
                continue
            for m in re.finditer(r"([^;]+?)\(r=(-?\d+(?:\.\d+)?)\)", str(rec[col])):
                rows.append({"group": str(rec[1]).strip(), "channel": channel,
                             "item": m.group(1).strip().strip(",").strip(),
                             "r": float(m.group(2))})
    return pd.DataFrame(rows, columns=cols)


def segment_affinity(pref: pd.DataFrame, sales: pd.DataFrame,
                     persona: dict) -> pd.DataFrame:
    """페르소나 그룹의 상품별 선호 r → item_id 단위로 붙인다 + 커버리지 게이트.

    게이트: 상품이 창의 `SEGMENT_MIN_MONTH_RATIO` 이상 '월'에서 팔렸어야 r 을 쓴다.
    창 중간에 생기거나 사라진 상품은 24개월 상관이 선호가 아니라 **존재한 시기**를
    재기 때문이다(상단 상수 주석의 `(컨)` 사례). 절대 개월수가 아니라 비율인 이유는
    `sample_gate` 와 같다 — 창이 바뀌어도 의미가 유지된다.
    """
    cols = ["item_id", "seg_r", "seg_months", "seg_used", "seg_item"]
    group = persona.get("segment_group")
    if pref.empty or not group or sales.empty:
        return pd.DataFrame(columns=cols)

    sel = pref.loc[pref["group"] == group]
    if sel.empty:
        return pd.DataFrame(columns=cols)

    # 커버리지는 `SEGMENT_ASSUMED_WINDOW` 안에서 잰다 — 위 상수 주석 참고.
    lo, hi = SEGMENT_ASSUMED_WINDOW
    win = sales.loc[sales["date"].between(lo, hi)]
    if win.empty:
        win = sales
    months = win.groupby("item_id")["date"].agg(lambda s: s.dt.to_period("M").nunique())
    window_months = int(win["date"].dt.to_period("M").nunique())
    floor = window_months * SEGMENT_MIN_MONTH_RATIO

    # 상품명 → item_id. 같은 이름이 여러 id 로 존재할 수 있어 전부 붙인다.
    lookup = sales[["item_id", "item", "channel"]].drop_duplicates()
    joined = sel.merge(lookup, on=["item", "channel"], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=cols)

    joined = joined.assign(seg_months=joined["item_id"].map(months).fillna(0).astype(int))
    joined = joined.assign(seg_used=joined["seg_months"] >= floor)
    # 같은 item_id 에 여러 행이 붙으면 절대값이 큰 r 을 남긴다 (신호가 센 쪽)
    joined = (joined.assign(_abs=joined["r"].abs())
              .sort_values(["_abs"], ascending=False)
              .drop_duplicates("item_id"))
    return joined.rename(columns={"r": "seg_r", "item": "seg_item"})[cols] \
        .reset_index(drop=True)


def channel_signals(ars: pd.DataFrame, sales: pd.DataFrame,
                    inflow: pd.DataFrame, grade: str) -> dict[str, dict]:
    """채널별 당일 집행 신호 = 편상관(음수 클립) × 래그계수 × 등급계수 × 등급반응.

    원 상관이 아니라 **요일 통제 편상관**을 쓴다. 원 상관 0.801 의 상당 부분이
    주말 효과라서, 원 상관으로 채널 순서를 매기면 README·탭1 배너가 말하는 서사와
    화면이 어긋난다 (`confound_table` 은 편상관 내림차순으로 반환된다).

    래그계수는 `best_lags` 결과를 그대로 따른다 — 채널 best_lag 가 D+0 이 아니면
    "오늘 집행"의 근거가 약하므로 깎는다. 상수로 박지 않는 이유는 창 길이가 답을
    바꾸기 때문이다: 731일 창에서는 3채널 모두 D+0 이지만 31일 창에서는 특산품이
    D+1 이었다.
    """
    confound = stats.confound_table(ars, sales)
    lags = lag.best_lags(lag.lag_matrix(inflow, sales))
    factor = GRADE_FACTOR.get(str(grade), 0.6)
    response = channel_grade_response(sales, inflow).get(str(grade), {})

    partial = {r.channel: float(r.partial) for r in confound.itertuples()}
    best_lag = {r.channel: int(r.best_lag) for r in lags.itertuples()}
    lag_r = {r.channel: float(r.pearson) for r in lags.itertuples()}

    out: dict[str, dict] = {}
    for channel in S.CHANNELS:
        p = partial.get(channel, float("nan"))
        # 음수 편상관을 그대로 곱하면 "유입이 늘면 덜 팔린다"는 채널이 점수를
        # 깎는 게 아니라 부호를 뒤집는다. 0 으로 클립해 신호 없음으로 눕힌다.
        p_clipped = 0.0 if not pd.notna(p) else max(0.0, p)
        k = best_lag.get(channel)
        penalty = 1.0 if k == 0 else LAG_PENALTY
        resp = float(response.get(channel, 1.0))
        out[channel] = {
            "partial": p,
            "best_lag": k,
            "lag_pearson": lag_r.get(channel, float("nan")),
            "penalty": penalty,
            "grade_factor": factor,
            "grade_response": resp,
            "signal": p_clipped * penalty * factor * resp,
        }
    return out


def item_signals(inflow: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """상품별 시차 상관 — 채널 평균에 가려지는 상품 단위 변별을 살린다.

    ⚠ 채널 신호만 쓰면 같은 채널 안에서 모든 상품의 intent 가 동일해져 상위가
    한 채널로 독점된다. 상품 신호를 반드시 섞는다.
    """
    item_lags = lag.best_lag_by_item(inflow, sales)
    if item_lags.empty:
        return pd.DataFrame(columns=["item_id", "item_lag", "item_r"])
    return item_lags.rename(columns={"best_lag": "item_lag", "pearson": "item_r"})[
        ["item_id", "item_lag", "item_r"]
    ]


# ── 후보 필터 ─────────────────────────────────────────────────────────

def recent_top_items(sales: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, list]:
    """기준일 직전 14일간 채널별 판매 상위 5종 = 최근 노출.

    이미 충분히 노출된 상품을 또 추천하면 '큐레이션'이 아니라 베스트셀러 목록이
    된다. 기준일 당일은 제외한다 — 추천 시점에 아직 모르는 정보다.
    """
    window = sales.loc[
        (sales["date"] >= as_of - pd.Timedelta(days=RECENT_DAYS))
        & (sales["date"] < as_of)
    ]
    if window.empty:
        return {}
    ranked = window.groupby(["channel", "item_id"])["qty"].sum().reset_index()
    return {
        str(ch): block.nlargest(RECENT_TOP_N, "qty")["item_id"].tolist()
        for ch, block in ranked.groupby("channel")
    }


def build_candidates(csm: pd.DataFrame, sales: pd.DataFrame,
                     as_of: pd.Timestamp, persona: dict,
                     grade: str) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """CSM 전 상품 → 추천 후보. 각 단계에서 몇 종이 남았는지 함께 돌려준다.

    티어 밴드는 프레임에서 **지우지 않고 `in_band` 컬럼으로 표시만 한다.**
    3순위(Cross-sell) 슬롯만 이 밴드를 면제하기 때문이다 — 지우면 그 슬롯이
    고를 것이 없어진다.
    """
    funnel: list[tuple[str, int]] = [("CSM 전 상품", len(csm))]

    pool = csm.loc[~csm["is_comp"].astype(bool)]
    funnel.append(("컴프(무상 제공) 제외", len(pool)))

    recent = recent_top_items(sales, as_of)
    flat = {i for items in recent.values() for i in items}
    pool = pool.loc[~pool["item_id"].isin(flat)]
    funnel.append((f"직전 {RECENT_DAYS}일 채널 상위 {RECENT_TOP_N}종 제외", len(pool)))

    # 재고 프록시 — 창의 25% 이상 팔린 날이 있어야 "언제든 낼 수 있는 상품"이다.
    # 단가·재고 컬럼이 없어서 판매일수로 대신한다(이 프로젝트의 마진 프록시와 같은 성격).
    window_days = int(sales["date"].nunique())
    days_sold = sales.groupby("item_id")["date"].nunique()
    ratio = pool["item_id"].map(days_sold).fillna(0) / max(window_days, 1)
    pool = pool.loc[ratio >= STOCK_MIN_RATIO]
    funnel.append((f"판매일수/창일수 ≥ {STOCK_MIN_RATIO:.0%} (재고 프록시)", len(pool)))

    pref_ord = TIER_ORD[persona["pref_tiers"][0]]
    band = (pool["tier"].map(TIER_ORD).fillna(0) - pref_ord).abs() <= TIER_BAND
    # 유입이 피크인 날은 프리미엄을 밴드 밖이어도 남긴다 — 사람이 가장 많은 날
    # 고단가 상품을 티어 규칙으로 스스로 빼면 기회를 버리는 셈이다.
    if str(grade) == "피크":
        band = band | (pool["tier"] == S.TIER_PREMIUM)
    pool = pool.assign(in_band=band.to_numpy(dtype=bool))
    funnel.append((f"└ 티어 밴드 |Δ| ≤ {TIER_BAND} (1·2순위 전용)",
                   int(pool["in_band"].sum())))
    floor = float(pool["margin_proxy"].quantile(CROSS_MARGIN_QUANTILE)) \
        if not pool.empty else float("nan")
    funnel.append((f"└ 마진 프록시 ≥ {floor:.3f} (3순위 전용)",
                   int((pool["margin_proxy"] >= floor).sum())))
    return pool.reset_index(drop=True), funnel


# ── 스코어 ────────────────────────────────────────────────────────────

def score_candidates(pool: pd.DataFrame, ch_sig: dict[str, dict],
                     item_sig: pd.DataFrame, season: pd.DataFrame,
                     seg: pd.DataFrame, persona: dict) -> pd.DataFrame:
    """cur_score = wsum({intent .50, base .30, csm .20}).

    `wsum` 은 0~1 축을 기대하므로 CSM(0~100)은 `nrm` 으로 후보 풀 안에서 다시
    정규화한다 — 이 프로젝트의 다른 지수(CII·VTS·CSM)가 축을 넣는 방식과 같다.
    """
    df = pool.merge(item_sig, on="item_id", how="left") \
             .merge(season, on="item_id", how="left") \
             .merge(seg if not seg.empty else pd.DataFrame(
                 columns=["item_id", "seg_r", "seg_months", "seg_used", "seg_item"]),
                 on="item_id", how="left")

    ch_signal = df["channel"].map(lambda c: ch_sig.get(c, {}).get("signal", 0.0))
    ch_signal = pd.to_numeric(ch_signal, errors="coerce").fillna(0.0)

    # 상품 신호가 임계 미만이거나 없으면 채널 신호로 대체한다. 0 으로 채우면
    # "측정 못 했다"가 감점이 되어 없는 근거를 만드는 셈이 된다.
    item_r = pd.to_numeric(df.get("item_r"), errors="coerce")
    usable = item_r.notna() & (item_r >= ITEM_SIGNAL_MIN_R)
    item_base = item_r.where(usable, ch_signal).clip(lower=0.0)

    # 기준일의 요일·월 프로파일을 상품 신호에 싣는다. 대체 신호(채널 신호)에도
    # 똑같이 실어서 실측·대체 상품이 같은 규칙으로 날짜를 반영하게 한다.
    season_f = pd.to_numeric(df.get("season"), errors="coerce").fillna(1.0)
    item_signal = (item_base * season_f).clip(lower=0.0)

    intent = INTENT_MIX["channel"] * ch_signal + INTENT_MIX["item"] * item_signal

    pref_ch = persona["pref_channels"]
    top_tier = persona["pref_tiers"][0]
    base_channel = df["channel"].map(pref_ch).fillna(0.0).astype("float64") * (
        df["tier"].eq(top_tier).map({True: 1.0, False: 0.75})
    )

    # 세그먼트 선호를 base 축의 **배수**로 싣는다. 축을 따로 세우지 않는 이유는
    # 측정된 34종만 새 축을 받아 나머지 1,650여 종보다 구조적으로 유리해지기
    # 때문이다 — CSM 의 `CSM_ELASTICITY_FILL_QUANTILE` 주석과 같은 함정이다
    # ("표본 부족"이 점수 보너스가 된다). 배수로 두면 미측정 상품의 배수가 1.0,
    # 즉 **중립**이라 가산도 감점도 없다. `season` 을 상품 신호에 싣는 방식과 같다.
    seg_r = pd.to_numeric(df.get("seg_r"), errors="coerce")
    seg_used = df.get("seg_used")
    seg_used = (seg_used.fillna(False).astype(bool) if seg_used is not None
                else pd.Series(False, index=df.index))
    seg_mult = (1.0 + SEGMENT_STRENGTH * seg_r.where(seg_used, 0.0)).fillna(1.0)
    base = base_channel * seg_mult

    cur = scoring.wsum(
        {"intent": intent, "base": base, "csm": scoring.nrm(df["csm"])},
        CUR_WEIGHTS,
    )
    return df.assign(
        ch_signal=ch_signal.to_numpy(),
        item_signal=item_signal.to_numpy(),
        item_signal_measured=usable.to_numpy(dtype=bool),
        season=season_f.to_numpy(),
        dow_idx=pd.to_numeric(df.get("dow_idx"), errors="coerce").fillna(1.0).to_numpy(),
        month_idx=pd.to_numeric(df.get("month_idx"), errors="coerce").fillna(1.0).to_numpy(),
        intent=intent.to_numpy(),
        base=base.to_numpy(),
        base_channel=base_channel.to_numpy(),
        seg_mult=seg_mult.to_numpy(),
        seg_r=seg_r.to_numpy(),
        seg_used=seg_used.to_numpy(dtype=bool),
        seg_months=pd.to_numeric(df.get("seg_months"), errors="coerce").to_numpy(),
        cur_score=cur.to_numpy(),
    ).sort_values(
        # 동점을 item_id 로 확정한다. 같은 값이 여러 개 있을 때 정렬 순서에
        # 맡기면 날짜만 바꿔도 2순위가 바뀌어 "날짜가 추천을 바꿨다"고 오독된다.
        ["cur_score", "item_id"], ascending=[False, True], kind="mergesort",
    ).reset_index(drop=True)


# ── 슬롯 선정 ─────────────────────────────────────────────────────────

def cross_margin_floor(scored: pd.DataFrame) -> float:
    """3순위 마진 프록시 하한값. 근거 문자열과 슬롯 선정이 같은 수를 쓰게 한다."""
    if scored.empty or "margin_proxy" not in scored.columns:
        return float("nan")
    return float(scored["margin_proxy"].quantile(CROSS_MARGIN_QUANTILE))


def pick_slots(scored: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Main / Sub / Cross-sell 3슬롯.

    Sub 는 **상위 8 안에서만** 고른다. "티어가 다르면 무조건 우선"으로 두면 점수
    순서를 뒤엎어 한참 아래의 엉뚱한 상품이 2순위로 올라온다.

    Cross-sell 은 **이 슬롯만 티어 밴드를 면제**한다. 면제하지 않으면 지역특산품
    (CSM 상위가 전부 일반 티어다 — 실측 1위 `22)으랏차차` standard)이 통째로
    탈락해 채널을 넘는 추천 자체가 성립하지 않는다. 대신 마진 프록시 하한을
    걸어 범용 대량 상품을 막는다 — 근거는 `CROSS_MARGIN_QUANTILE` 주석에 있다.
    """
    in_band = scored.loc[scored["in_band"]]
    if in_band.empty:
        return []

    main = in_band.iloc[0]
    slots: list[tuple[str, pd.Series]] = [("Main", main)]

    # ── Sub
    head = in_band.head(SUB_POOL_TOP_N)
    same_ch = head.loc[
        (head["channel"] == main["channel"]) & (head["item_id"] != main["item_id"])
    ]
    if not same_ch.empty:
        ranked = same_ch.assign(
            tier_diff=(same_ch["tier"] != main["tier"]).astype(int),
            measured=same_ch["lift_measured"].astype(int),
        ).sort_values(["tier_diff", "measured", "cur_score", "item_id"],
                      ascending=[False, False, False, True], kind="mergesort")
        slots.append(("Sub", ranked.iloc[0]))
    else:
        rest = in_band.loc[in_band["item_id"] != main["item_id"]]
        if not rest.empty:
            slots.append(("Sub", rest.iloc[0]))

    # ── Cross-sell (티어 밴드 면제 + 마진 프록시 하한)
    taken = {s["item_id"] for _, s in slots}
    other = scored.loc[
        (scored["channel"] != main["channel"]) & (~scored["item_id"].isin(taken))
    ]
    floor = cross_margin_floor(scored)
    if not other.empty and pd.notna(floor):
        gated = other.loc[other["margin_proxy"] >= floor]
        # 게이트가 후보를 다 지우면 추천을 못 내는 것보다 게이트를 푸는 편이 낫다.
        # 그 사실은 `internal_reason` 에 그대로 드러난다.
        other = gated if not gated.empty else other
    if not other.empty:
        slots.append(("Cross-sell", other.iloc[0]))
    return slots


# ── 문구 ──────────────────────────────────────────────────────────────

def source_label(row: pd.Series) -> str:
    """lift 가 실측인지 대체값인지 — CSM 의 `lift_measured` 규약을 문구로 옮긴다."""
    return "실측" if bool(row["lift_measured"]) else "파생"


def segment_note(row: pd.Series, persona: dict) -> str:
    """세그먼트 선호 축의 근거 문자열.

    `[실측]` 이 아니라 `[파생]` 이다 — 개인 구매 기록이 아니라 집단 구성비끼리의
    상관이고, 원 지표가 누적이라 시간 추세와 분리되지 않는다. 그 한계를 문자열에
    그대로 적는다. 화면이 근거보다 세게 말하면 안 된다.
    """
    group = persona.get("segment_group", "?")
    r = row.get("seg_r")
    if r is None or pd.isna(r):
        return (f"[파생] 세그먼트({group}) 선호 추정표에 없는 상품 → 배수 1.00(중립)")
    months = row.get("seg_months")
    months_txt = "" if pd.isna(months) else f" · 판매월 {int(months)}"
    if not bool(row.get("seg_used", False)):
        return (f"[파생] 세그먼트({group}) r {float(r):+.2f} 이나 판매 커버리지 미달"
                f"{months_txt} → 시기 아티팩트 위험으로 미적용(배수 1.00)")
    kind = "선호" if float(r) > 0 else "비선호"
    lo, hi = SEGMENT_ASSUMED_WINDOW
    return (f"[파생] 세그먼트({group}) {kind} r {float(r):+.2f}"
            f" → base 배수 {float(row.get('seg_mult', 1.0)):.2f}{months_txt}"
            f" · 회원 누적 구성비 기반 추정이라 시간 추세와 분리되지 않음"
            f" · [가정] {lo:%Y-%m}~{hi:%Y-%m} 재계산본으로 간주(원파일은 24개월 산출)")


def internal_reason(row: pd.Series, ch_sig: dict, persona: dict,
                    grade: str, extra: str = "") -> str:
    """운영자용 근거 문자열. 모든 수치에 [실측]/[파생]/[가정] 라벨을 단다."""
    ch = str(row["channel"])
    sig = ch_sig.get(ch, {})
    partial = sig.get("partial", float("nan"))
    k = sig.get("best_lag")
    lag_txt = f"D+{k}" if k is not None else "미측정"

    lift = row["lift"]
    if bool(row["lift_measured"]) and pd.notna(lift):
        lift_txt = f"[실측] 유입 탄력성 lift {float(lift):.2f}"
    else:
        # 표본 미달을 1.0 으로 채우지 않는다 — 없는 근거를 만드는 것이다.
        lift_txt = "[파생] lift 표본 미달(NaN) → 실측 중앙값으로 대체"

    if bool(row["item_signal_measured"]):
        item_txt = f"[실측] 상품 시차상관 r {float(row['item_r']):.3f}"
    else:
        item_txt = f"[파생] 상품 상관 r < {ITEM_SIGNAL_MIN_R:.2f} → 채널 신호로 대체"
    item_txt += (f" × [실측] 요일지수 {float(row['dow_idx']):.2f}"
                 f" × 월지수 {float(row['month_idx']):.2f}"
                 f" → 상품신호 {float(row['item_signal']):.3f}")

    parts = [
        f"[실측] 요일통제 편상관 {partial:.3f} · 채널 최적시차 {lag_txt}"
        f" · 당일 유입등급 '{grade}'(계수 {sig.get('grade_factor', 0):.1f}"
        f" × 채널 등급반응 {sig.get('grade_response', 1.0):.2f})",
        item_txt,
        lift_txt,
        f"[실측] CSM {float(row['csm']):.1f} ({S.TIER_LABEL.get(str(row['tier']), row['tier'])}"
        f" · 누적 {int(row['total_qty']):,}개)",
        f"[가정] 페르소나 채널선호 {persona['pref_channels'].get(ch, 0):.2f}"
        f" — demo(2026-06~08)와 sales(2023~2024) 겹침 0일이라 실측 불가",
        segment_note(row, persona),
        f"[파생] cur_score {float(row['cur_score']):.1f}"
        f" (intent {float(row['intent']):.3f} / base {float(row['base']):.3f})",
    ]
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def personalized_message(row: pd.Series, ch_sig: dict, slot: str) -> str:
    """VIP 컨시어지 어조 2~3문장. **채널 best_lag 로 분기**한다.

    하드코딩하면 D+0 상품에 "다음 날" 문구가 붙어 화면이 근거와 다른 말을 한다.
    창이 답을 바꾼 전례가 실제로 있다 — 특산품은 31일 창에서 D+1, 731일 창에서 D+0.
    """
    name = str(row["item"])
    ch_label = S.CHANNEL_LABEL.get(str(row["channel"]), str(row["channel"]))
    k = ch_sig.get(str(row["channel"]), {}).get("best_lag")

    if k == 0:
        lead = (f"머무시는 동안 {ch_label}에서 「{name}」을(를) 곁에 두시면 어떨까요.")
        follow = "오늘 객장이 붐비는 흐름과 같은 날에 가장 많이 찾으시는 품목입니다."
    elif k == 1:
        lead = (f"돌아가시는 길에 {ch_label}의 「{name}」을(를) 살펴보시길 권해 드립니다.")
        follow = "객장이 붐빈 다음 날에 특히 많이 찾으시는 품목이라 미리 챙겨 두었습니다."
    elif k is None:
        lead = f"{ch_label}의 「{name}」을(를) 조심스럽게 권해 드립니다."
        follow = "시차 근거가 아직 충분치 않아, 상품 자체의 반응만으로 골라 두었습니다."
    else:
        lead = f"{ch_label}의 「{name}」을(를) 권해 드립니다."
        follow = f"객장이 붐빈 지 {k}일째 되는 날에 가장 많이 찾으시는 품목입니다."

    closing = {
        "Main": "오늘 하루, 가장 먼저 안내해 드리고 싶은 한 가지입니다.",
        "Sub": "앞서 권해 드린 품목과 결이 다르니, 함께 두고 고르셔도 좋겠습니다.",
        "Cross-sell": "머무시는 시간 너머까지 남을 만한 것으로 곁들여 보시길 바랍니다.",
    }[slot]
    return f"{lead} {follow} {closing}"


def gender_share(demo: pd.DataFrame, gender: str) -> float:
    """전체 고객 중 해당 성별 비중. `vip_ratio` 는 성별 **안에서의** 비중을 주므로,
    전체 구성비로 환산하려면 이 값을 곱해야 한다."""
    if demo.empty or "visitors" not in demo.columns:
        return float("nan")
    total = float(demo["visitors"].sum())
    if not total:
        return float("nan")
    return float(demo.loc[demo["gender"] == gender, "visitors"].sum()) / total


def intent_summary(as_of: pd.Timestamp, day: pd.Series, persona: dict,
                   ch_sig: dict, share: float, share_in_gender: float) -> str:
    """오늘의 의도 요약 — 페르소나 구성비는 실측, 채널 선호는 가정."""
    top_ch = max(S.CHANNELS, key=lambda c: ch_sig.get(c, {}).get("signal", 0.0))
    top = ch_sig.get(top_ch, {})
    return (
        f"{as_of:%Y-%m-%d}({S.DOW_NAMES[int(day['dow'])]}) 유입지수 "
        f"{float(day['inflow']):.1f}·등급 '{day['grade']}'({day['source']} 기준). "
        f"[실측] 요일통제 편상관 최상위 채널은 {S.CHANNEL_LABEL[top_ch]}"
        f"({top.get('partial', float('nan')):.3f}, 최적시차 D+{top.get('best_lag')}). "
        f"[실측] {persona['label']} 세그먼트는 전체 고객 구성의 {share:.1%}"
        f"(여성 내 {share_in_gender:.1%}) — 원본이 누적 고객수라 구성비로만 인용한다. "
        f"[가정] 채널 선호도는 demo·sales 겹침이 0일이라 실측 불가로 가정값이다."
    )


# ── 출력 ──────────────────────────────────────────────────────────────

def print_segment(pref: pd.DataFrame, seg: pd.DataFrame, persona: dict) -> None:
    """세그먼트 축이 실제로 무엇을 들고 왔는지 — 게이트 탈락분까지 드러낸다."""
    group = persona.get("segment_group", "?")
    if pref.empty:
        print(f"[세그먼트 선호] 표 없음 → 이 축은 비활성 (배수 전부 1.00)")
        return
    lo, hi = SEGMENT_ASSUMED_WINDOW
    print(f"[세그먼트 선호 · {group}]  출처 data/raw/segment_preference.csv "
          f"— [파생] 회원 누적 구성비 기반 추정")
    print(f"  ⚠ [가정] {lo:%Y-%m}~{hi:%Y-%m} 재계산본으로 간주 "
          f"(원파일은 24개월 산출) · 커버리지도 같은 창에서 측정")
    if seg.empty:
        print("  해당 그룹 매칭 0종 → 축 비활성")
        print()
        return
    for r in seg.sort_values("seg_r", ascending=False).itertuples():
        mark = "적용" if r.seg_used else "미적용(커버리지)"
        print(f"  {str(r.seg_item)[:30]:<32} r {r.seg_r:+.2f}"
              f" · 판매월 {int(r.seg_months):>2}  → {mark}")
    used = int(seg["seg_used"].sum())
    print(f"  적용 {used}종 / 표 매칭 {len(seg)}종")
    print()


def print_report(funnel, scored, gate, ch_sig, persona, as_of, day) -> None:
    line = "─" * 78
    print(line)
    print(f"AI VIP 큐레이션 프로토타입 — {persona['label']} · 기준일 {as_of:%Y-%m-%d}")
    print(line)
    print(f"창: {gate.get('window_days')}일 · lift 표본요건 {gate.get('min_days')}일 "
          f"· 실측 {gate.get('measured')}종 / 탈락 {gate.get('dropped')}종 "
          f"· 탄력성 대체값 {gate.get('elasticity_fill', float('nan')):.3f}")
    tickets = day.get("tickets")
    tickets_txt = f" · 입장권 {int(tickets):,}건" if pd.notna(tickets) else ""
    print(f"당일 유입: 지수 {float(day['inflow']):.1f} · 등급 '{day['grade']}'"
          f" · 근거 {day['source']}{tickets_txt}")
    print()
    print("[후보 필터 흐름]")
    for i, (label, n) in enumerate(funnel):
        arrow = "  " if i == 0 else "→ "
        print(f"  {arrow}{label:<44} {n:>6,}종")
    print()
    print("[채널 신호 — 요일통제 편상관 × 래그계수 × 등급계수 × 채널 등급반응]")
    for ch in S.CHANNELS:
        s = ch_sig[ch]
        print(f"  {S.CHANNEL_LABEL[ch]:<8} partial {s['partial']:+.3f}"
              f" · best_lag D+{s['best_lag']} (r {s['lag_pearson']:.3f})"
              f" · 계수 {s['penalty']:.1f}×{s['grade_factor']:.1f}"
              f"×{s['grade_response']:.3f}"
              f" → 신호 {s['signal']:.3f}")
    print()

    def table(title: str, frame: pd.DataFrame) -> None:
        print(title)
        print(f"  {'#':<3}{'상품':<24}{'채널':<10}{'티어':<8}"
              f"{'cur':>7}{'intent':>8}{'계절':>6}"
              f"{'base':>7}{'세그':>6}{'CSM':>7}  {'lift':<10}밴드")
        for i, r in enumerate(frame.itertuples(), 1):
            lift = (f"{r.lift:.2f}" if bool(r.lift_measured) and pd.notna(r.lift)
                    else "NaN(미달)")
            print(f"  {i:<3}{str(r.item)[:22]:<24}{S.CHANNEL_LABEL[r.channel]:<10}"
                  f"{S.TIER_LABEL.get(r.tier, r.tier):<8}"
                  f"{r.cur_score:>7.1f}{r.intent:>8.3f}"
                  f"{r.season:>6.2f}"
                  f"{r.base:>7.3f}{r.seg_mult:>6.2f}{r.csm:>7.1f}  "
                  f"{lift:<10}{'O' if r.in_band else '-'}")
        print()

    # 전체 상위 8 — 3순위(티어 밴드 면제)가 여기서 나온다
    table("[cur_score 상위 8종 · 전체]  (밴드 O = 1·2순위 자격)", scored.head(8))
    # 밴드 통과 상위 8 — 1·2순위는 이 안에서만 고른다
    table("[cur_score 상위 8종 · 티어 밴드 통과분]  (1·2순위 후보 풀)",
          scored.loc[scored["in_band"]].head(8))


# ── 메인 ──────────────────────────────────────────────────────────────

def build_payload(persona_id: str, as_of: pd.Timestamp, verbose: bool = True) -> dict:
    persona = PERSONAS[persona_id]
    data, sources = loaders.load_all_impl()
    ars, sales, demo = data["ars"], data["sales"], data["demo"]

    inflow = inflow_mod.build_inflow(ars, sales)
    row = inflow.loc[inflow["date"] == as_of]
    if row.empty:
        raise SystemExit(
            f"기준일 {as_of:%Y-%m-%d} 의 유입 데이터가 없다. "
            f"가용 구간: {inflow['date'].min():%Y-%m-%d} ~ {inflow['date'].max():%Y-%m-%d}"
        )
    day = row.iloc[0]
    if sales.loc[sales["date"] == as_of].empty:
        print(f"⚠ 기준일 {as_of:%Y-%m-%d} 에 판매 데이터가 없다 — 유입 신호만으로 추천한다.")
    grade = str(day["grade"])

    csm = crosssell.compute_csm(sales, inflow)
    gate = dict(csm.attrs.get("gate", {}))

    ch_sig = channel_signals(ars, sales, inflow, grade)
    pool, funnel = build_candidates(csm, sales, as_of, persona, grade)
    seg_pref = load_segment_preference()
    seg = segment_affinity(seg_pref, sales, persona)
    scored = score_candidates(pool, ch_sig, item_signals(inflow, sales),
                              season_factors(sales, pool["item_id"], as_of),
                              seg, persona)
    slots = pick_slots(scored)
    if not slots:
        raise SystemExit("후보가 모두 탈락했다 — 필터 상수를 확인할 것.")

    # 세그먼트 구성비는 실측이다. 단 원본 CUST_WHOL_CNT 가 누적 고객수라
    # 규모로는 쓸 수 없고 비율로만 인용한다 (CLAUDE.md 의 demo 절).
    share_gender = vip.vip_ratio(demo, (persona["age_band"],), persona["gender"])
    share = share_gender * gender_share(demo, persona["gender"])

    if verbose:
        print(f"데이터 출처: {sources}")
        print_segment(seg_pref, seg, persona)
        print_report(funnel, scored, gate, ch_sig, persona, as_of, day)

    main_id = str(slots[0][1]["item_id"])
    recommendations = []
    for rank, (slot, item) in enumerate(slots, 1):
        extra = ""
        if slot == "Cross-sell":
            co = crosssell.cooccurrence(sales, main_id, str(item["item_id"]))
            if int(co["days"]) < COOC_MIN_DAYS:
                extra = (f"[파생] 1순위와 동반 상승 근거: 공통 판매일 {co['days']}일 "
                         f"— 표본 부족({COOC_MIN_DAYS}일 미만)")
            else:
                extra = (f"[실측] 1순위와 공통 판매일 {co['days']}일 중 "
                         f"동반 상승 {co['both_up']}일")
            floor = cross_margin_floor(scored)
            passed = pd.notna(floor) and float(item["margin_proxy"]) >= floor
            extra += (f" · [파생] 티어 밴드 면제 슬롯 · 마진 프록시"
                      f" {float(item['margin_proxy']):.3f}"
                      f" {'≥' if passed else '<'} 하한 {floor:.3f}"
                      f"(후보 풀 {CROSS_MARGIN_QUANTILE:.0%} 분위)"
                      f"{'' if passed else ' — 게이트 통과 상품이 없어 해제됨'}")
        recommendations.append({
            "rank": rank,
            "recommendation_type": slot,
            "product_id": str(item["item_id"]),
            "product_name": str(item["item"]),
            "internal_reason": internal_reason(item, ch_sig, persona, grade, extra),
            "vip_personalized_message": personalized_message(item, ch_sig, slot),
        })

    return {
        "customer_id": persona_id,
        "today_intent_summary": intent_summary(as_of, day, persona, ch_sig,
                                               share, share_gender),
        "recommendations": recommendations,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="AI VIP 큐레이션 추천 JSON 프로토타입")
    ap.add_argument("--persona", default="SEG-F50-ROOM", choices=sorted(PERSONAS))
    ap.add_argument("--date", required=True, help="기준일 YYYY-MM-DD")
    ap.add_argument("--out", default="out/curation.json", help="JSON 출력 경로")
    ap.add_argument("--quiet", action="store_true", help="stdout 진단 출력을 생략한다")
    args = ap.parse_args()

    as_of = pd.Timestamp(args.date).normalize()
    payload = build_payload(args.persona, as_of, verbose=not args.quiet)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n✔ 저장: {out}  ({len(text.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
