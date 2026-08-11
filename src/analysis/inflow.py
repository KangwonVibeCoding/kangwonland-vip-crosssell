"""유입 강도 — CII(ARS 기반) + CAI(판매 기반 대체) + 요일 브릿지.

ARS 와 판매 데이터의 기간이 부분적으로만 겹치므로 두 지수를 나눠 만든다.

  CII (Casino Inflow Intensity) — ARS 가 있는 날. 진짜 유입량.
  CAI (Casino Activity Index)   — ARS 가 없는 날. 카지노 식음 소비로 추정한
                                  객장 활동 강도 + ARS 요일 프로파일 보정.

구간 A(2024-12)에서 두 지수의 상관을 측정해 대체 타당성을 화면에 수치로 제시한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings as S
from src.analysis import scoring, stats


def has_pressure_signal(ars: pd.DataFrame) -> bool:
    """recv_total(총 접수자)이 실제로 쓸 만큼 존재하는지.

    2024-12 원본에는 있고 2026년 전처리본에는 없다. 컬럼 존재 여부만 보면
    concat 된 프레임에서 항상 True 가 나오므로 **유효값 개수**로 판단한다.
    3개 미만이면 정규화가 무의미하므로 없는 것으로 취급한다.
    """
    if ars.empty or "recv_total" not in ars.columns:
        return False
    return int(pd.to_numeric(ars["recv_total"], errors="coerce").notna().sum()) >= 3


def compute_cii(ars: pd.DataFrame,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """ARS 기반 카지노 유입 강도 지수 (0~100).

    ⚠ `winners`(총 당첨자)는 쓰지 않는다. 실측 결과 일일 정원 캡(최대 2,999)에
    묶여 CV 0.178 로 거의 상수이므로, 정규화하면 의미 없는 노이즈만 만든다.
    변동을 담고 있는 신호는 recv_total(CV 0.372)과 tickets(CV 0.302)다.

    ⚠ 2026년 ARS 전처리본에는 recv_total 이 없다. 그 구간에서는 수요 압력 축을
    빼고 demand/convert 2축으로 폴백한다 — 없는 컬럼을 0 으로 채워 계산하면
    "그 날 접수자가 0명"이라는 거짓 신호가 지수에 그대로 들어간다.
    `pressure_signal` 컬럼으로 어느 방식이었는지 화면에 드러낼 수 있게 한다.
    """
    if ars.empty:
        return pd.DataFrame(columns=["date", "cii", "cii_ma7", "grade", "source"])

    df = ars.sort_values("date").reset_index(drop=True)
    use_pressure = has_pressure_signal(df)

    if use_pressure:
        weights = weights or S.INFLOW_WEIGHTS
        parts = {
            "demand": scoring.nrm(df["tickets"]),          # 실제 입장 전환 건수
            "pressure": scoring.nrm(df["recv_total"]),     # 수요 압력
            "convert": scoring.nrm(df["buy_rate"]),        # 당첨자 → 구매 전환
        }
    else:
        weights = weights or S.INFLOW_WEIGHTS_NO_PRESSURE
        parts = {
            "demand": scoring.nrm(df["tickets"]),
            "convert": scoring.nrm(df["buy_rate"]),
        }
    # 호출자(사이드바 슬라이더)가 3축 가중치를 넘겨도 안전하다 — wsum 이
    # parts 에 없는 키를 버리고 남은 가중치로 다시 정규화한다.
    cii = scoring.wsum(parts, weights)
    cols = [c for c in ("date", "dow", "dow_name", "month", "recv_total", "winners",
                        "tickets", "buy_rate", "compete_ratio", "is_capped")
            if c in df.columns]
    out = df[cols].copy()
    return out.assign(
        cii=cii.to_numpy(),
        cii_ma7=cii.rolling(7, min_periods=1).mean().to_numpy(),
        grade=scoring.grade(cii).to_numpy(),
        source="CII",
        pressure_signal=use_pressure,
    )


def ars_dow_index(ars: pd.DataFrame) -> pd.Series:
    """ARS 구매건수의 요일 인덱스 — CAI 의 요일 브릿지에 쓴다.

    실측: 2024-12 토 1.45 / 일 1.31, 2026-05 토 1.39 / 일 1.23.
    1년 반 간격에도 프로파일이 재현되므로 브릿지로 쓸 근거가 된다.
    """
    if ars.empty:
        return pd.Series([1.0] * 7, index=range(7), name="dow_index")
    series = ars.set_index("date")["tickets"]
    idx = stats.dow_index(series)
    return idx.fillna(1.0)


def compute_cai(sales: pd.DataFrame,
                dow_idx: pd.Series | None = None,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """판매 기반 객장 활동 지수 (0~100) — ARS 가 없는 구간의 유입 프록시.

    카지노 식음영업장은 객장 안에 있으므로 그 소비량이 곧 체류 활동량이다.
    거래 행 수(주문 다양성)를 함께 쓰면 '소수가 많이 사는 날'과 '다수가 조금씩
    사는 날'을 구분할 수 있다.
    """
    weights = weights or S.CAI_WEIGHTS
    qty = stats.daily_qty(sales, S.CH_CASINO)
    if qty.empty:
        # 카지노 식음이 없으면 전체 판매로 대체한다
        qty = stats.daily_qty(sales)
    if qty.empty:
        return pd.DataFrame(columns=["date", "cai", "cai_ma7", "grade", "source"])

    tx = stats.daily_tx(sales, S.CH_CASINO)
    if tx.empty:
        tx = stats.daily_tx(sales)
    tx = tx.reindex(qty.index).fillna(0)

    dow_idx = dow_idx if dow_idx is not None else pd.Series([1.0] * 7, index=range(7))
    dows = pd.to_datetime(pd.Series(qty.index)).dt.dayofweek.to_numpy()
    dow_signal = pd.Series(
        pd.Series(dows).map(dow_idx.to_dict()).fillna(1.0).to_numpy(),
        index=qty.index, dtype="float64",
    )

    parts = {
        "volume": scoring.nrm(qty),
        "breadth": scoring.nrm(tx),
        "dow": scoring.nrm(dow_signal, robust=False),
    }
    cai = scoring.wsum(parts, weights)
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(qty.index))})
    return frame.assign(
        casino_qty=qty.to_numpy(),
        casino_tx=tx.to_numpy(),
        dow=frame["date"].dt.dayofweek,
        dow_name=frame["date"].dt.dayofweek.map(dict(enumerate(S.DOW_NAMES))),
        month=frame["date"].dt.month,
        cai=cai.to_numpy(),
        cai_ma7=cai.rolling(7, min_periods=1).mean().to_numpy(),
        grade=scoring.grade(cai).to_numpy(),
        source="CAI",
    )


def build_inflow(ars: pd.DataFrame, sales: pd.DataFrame,
                 weights: dict[str, float] | None = None) -> pd.DataFrame:
    """구간에 맞는 유입 시계열을 만든다 — ARS 가 있는 날은 CII, 없으면 CAI.

    반환 컬럼: date, inflow(0~100), inflow_ma7, grade, source('CII'|'CAI'), …
    `source` 컬럼이 UI 의 구간 배지와 직결된다 — 어느 근거로 그린 값인지 항상
    화면에 드러내기 위해서다.
    """
    cii = compute_cii(ars, weights)
    dow_idx = ars_dow_index(ars)
    cai = compute_cai(sales, dow_idx)

    frames: list[pd.DataFrame] = []
    if not cii.empty:
        frames.append(
            cii.rename(columns={"cii": "inflow", "cii_ma7": "inflow_ma7"})
        )
    if not cai.empty:
        have = set(cii["date"]) if not cii.empty else set()
        extra = cai.loc[~cai["date"].isin(have)]
        if not extra.empty:
            frames.append(
                extra.rename(columns={"cai": "inflow", "cai_ma7": "inflow_ma7"})
            )
    if not frames:
        return pd.DataFrame(columns=["date", "inflow", "inflow_ma7", "grade", "source"])

    out = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    # 이동평균은 병합 후 다시 계산해야 구간 경계에서 끊기지 않는다
    return out.assign(inflow_ma7=out["inflow"].rolling(7, min_periods=1).mean())


def cai_validity(ars: pd.DataFrame, sales: pd.DataFrame) -> dict[str, float | int]:
    """CAI 가 CII 를 대체할 수 있는지 — 겹치는 구간에서 상관을 측정한다.

    이 수치를 화면에 노출해 "ARS 없는 기간의 유입 프록시"라는 주장을 방어한다.
    """
    cii = compute_cii(ars)
    cai = compute_cai(sales, ars_dow_index(ars))
    if cii.empty or cai.empty:
        return {"n_days": 0, "pearson": float("nan"), "spearman": float("nan")}
    a = cii.set_index("date")["cii"]
    b = cai.set_index("date")["cai"]
    x, y = stats.align(a, b)
    return {
        "n_days": int(len(x)),
        "pearson": stats.pearson(x, y),
        "spearman": stats.spearman(x, y),
    }


def hi_lo_days(inflow: pd.DataFrame,
               quantile: float = S.HI_LO_QUANTILE) -> tuple[set, set]:
    """고유입일 / 저유입일 집합 — lift 계산의 분모·분자 구간.

    표본이 너무 적으면(6일 미만) 분할이 무의미하므로 빈 집합을 반환하고,
    호출자가 lift 계산을 생략한다.
    """
    if inflow.empty or len(inflow) < 6:
        return set(), set()
    s = inflow.set_index("date")["inflow"].dropna()
    if s.empty:
        return set(), set()
    hi_cut = s.quantile(1 - quantile)
    lo_cut = s.quantile(quantile)
    hi = set(s.loc[s >= hi_cut].index)
    lo = set(s.loc[s <= lo_cut].index)
    return hi, lo


def hi_lo_days_by_month(inflow: pd.DataFrame,
                        quantile: float = S.HI_LO_QUANTILE) -> tuple[set, set]:
    """달마다 따로 고·저 유입일을 나눈 뒤 합친다 — **추세 제거용**.

    ⚠ 전역 분위로 한 번에 자르면 유입에 추세가 있을 때 고유입일이 특정 시기로
    몰린다. 실측(2023~2024 731일): 유입이 2023→2024 에 21.6% 감소해서
    고유입일 220일 중 대부분이 2023년, 저유입일 대부분이 2024년에 쏠렸다.
    그러면 lift 가 유입 탄력성이 아니라 **상품 수명주기**를 측정하게 된다 —
    2023 상반기에 단종된 `(22)아이스아메리카노`(2023-01~06 에만 148,961개)가
    lift 47.05 로 1위가 됐다. 유입에 민감한 게 아니라 그때만 존재한 상품이다.

    달 안에서 자르면 비교가 항상 동시대끼리라 이 교란이 사라진다
    (실측 lift 최댓값 47.05 → 2.02). 창이 한 달이면 `hi_lo_days` 와 동일하다 —
    구간 A(2024-12) 결과가 그대로 보존되는 이유다.
    """
    if inflow.empty or len(inflow) < 6:
        return set(), set()
    hi: set = set()
    lo: set = set()
    for _, block in inflow.groupby(inflow["date"].dt.to_period("M")):
        b_hi, b_lo = hi_lo_days(block, quantile)
        hi |= b_hi
        lo |= b_lo
    return hi, lo


def dow_profile_table(ars: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """ARS + 3채널 요일 인덱스 비교표 (탭3 의 4계열 차트용).

    실측 하이라이트: 유입은 토요일 피크(1.45)인데 특산품은 일요일 1.63 —
    '주말 유입 → 체크아웃 선물 구매' 시퀀스의 근거다(구간 A = 2024-12 기준).

    ⚠ 이 표는 넘겨받은 `sales` 의 창을 그대로 쓴다. 창이 바뀌면 값도 바뀐다 —
    일요일 피크는 2023·2024 두 해 모두에서 재현되지만(각 1.36), 구간 A 의
    '토요일 0.92(평균 이하)'는 2년 창에서 1.20 이라 재현되지 않는다.
    """
    rows: list[dict] = []
    if not ars.empty:
        idx = stats.dow_index(ars.set_index("date")["tickets"])
        rows.append({"series": "ars", "label": "카지노 유입(ARS)",
                     **{S.DOW_NAMES[i]: idx.get(i, np.nan) for i in range(7)}})
    for ch in S.CHANNELS:
        qty = stats.daily_qty(sales, ch)
        if qty.empty:
            continue
        idx = stats.dow_index(qty)
        rows.append({"series": ch, "label": S.CHANNEL_LABEL[ch],
                     **{S.DOW_NAMES[i]: idx.get(i, np.nan) for i in range(7)}})
    return pd.DataFrame(rows)


def month_profile_table(sales: pd.DataFrame) -> pd.DataFrame:
    """채널별 월 인덱스 (계절성). 특산품 9월 1.43 / 룸서비스 1월 1.21 실측(2년)."""
    rows: list[dict] = []
    for ch in S.CHANNELS:
        qty = stats.daily_qty(sales, ch)
        if qty.empty:
            continue
        idx = stats.month_index(qty)
        rows.append({"series": ch, "label": S.CHANNEL_LABEL[ch],
                     **{m: idx.get(m, np.nan) for m in range(1, 13)}})
    return pd.DataFrame(rows)
