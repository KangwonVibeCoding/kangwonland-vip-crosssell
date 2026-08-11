"""최종 폴백 — 코드에 임베드된 상수 데이터.

파일 시스템도 네트워크도 건드리지 않는다. pandas 만 쓴다. **절대 예외를 던지지
않는다.** 저장소에 데이터 파일이 하나도 없고 API 키도 없는 상태로 배포돼도 앱이
정상 렌더되게 하는 최후의 안전장치다.

수치는 실데이터에서 측정한 통계를 모사한다 (ARS 요일지수 토 1.45 / 일 1.31, 특산품
9월 피크 1.43, 룸서비스 1월 1.21, 티어 비율 등) — 폴백 상태에서도 차트가 의미
있는 모양으로 보여야 데모가 성립하기 때문이다.

**여기 상수는 실측을 따라가야 한다.** 판매 데이터가 2년으로 확장됐을 때 갱신을
빠뜨려 한동안 1년치 구값(9월 1.48 / 특산품 일요일 1.68)을 모사하고 있었다.
회귀 기준값(CLAUDE.md)이 바뀌면 이 파일도 같이 고친다.

난수를 쓰지 않으므로 항상 같은 결과가 나온다.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pandas as pd


def _stable_num(text: str) -> str:
    """프로세스 간에도 동일한 6자리 숫자열.

    내장 hash() 는 PYTHONHASHSEED 로 무작위화되므로 쓰지 않는다 — 실행마다
    상품 ID 가 달라지면 '결정론적'이라는 이 모듈의 약속이 깨진다.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{int(digest[:8], 16) % 1_000_000:06d}"

# ── ARS 예약현황 31일 (2024-12 실데이터 형태를 모사) ────────────────────
# (일자, 총접수자, ARS접수자, 총당첨자, 구매건수, 구매율%)
_ARS_ROWS: tuple[tuple[int, int, int, int, int, float], ...] = (
    (1, 3458, 1015, 2990, 1206, 40.33), (2, 2213, 687, 2213, 753, 34.02),
    (3, 2101, 644, 2101, 730, 34.74), (4, 2064, 631, 2064, 700, 33.91),
    (5, 2158, 660, 2158, 742, 34.38), (6, 2604, 780, 2604, 921, 35.37),
    (7, 3301, 962, 2995, 1180, 39.40), (8, 3096, 905, 2993, 1088, 36.35),
    (9, 2189, 671, 2189, 748, 34.17), (10, 2076, 636, 2076, 707, 34.06),
    (11, 2043, 626, 2043, 699, 34.21), (12, 2187, 668, 2187, 758, 34.66),
    (13, 2661, 795, 2661, 951, 35.74), (14, 3402, 988, 2997, 1215, 40.54),
    (15, 3140, 917, 2994, 1102, 36.81), (16, 2154, 660, 2154, 735, 34.12),
    (17, 1996, 613, 1996, 676, 33.87), (18, 1757, 546, 1757, 569, 32.38),
    (19, 2098, 643, 2098, 719, 34.27), (20, 2588, 775, 2588, 918, 35.47),
    (21, 3364, 977, 2996, 1201, 40.09), (22, 3178, 928, 2995, 1114, 37.20),
    (23, 2216, 678, 2216, 761, 34.34), (24, 2394, 726, 2394, 843, 35.21),
    (25, 2871, 852, 2871, 1041, 36.26), (26, 2135, 654, 2135, 731, 34.24),
    (27, 2649, 792, 2649, 947, 35.75), (28, 3519, 1018, 2999, 1222, 40.75),
    (29, 3244, 944, 2996, 1145, 38.22), (30, 2287, 698, 2287, 792, 34.63),
    (31, 2242, 579, 2242, 837, 37.33),
)

# ── 상품 마스터 (채널, 영업장, 상품명, 기준 일판매량) ───────────────────
# 실데이터의 대표 상품명을 그대로 쓴다 → 티어 분류 로직도 함께 검증된다.
_ITEMS: tuple[tuple[str, str, str, int], ...] = (
    # 카지노 식음 — 5개 영업장 (실측 존재)
    ("casino_fnb", "크리스탈라운지", "(V)에비앙", 62),
    ("casino_fnb", "크리스탈라운지", "우유", 45),
    ("casino_fnb", "크리스탈라운지", "아메리카노", 58),
    ("casino_fnb", "크리스탈라운지", "(23)한우등심구이", 7),
    ("casino_fnb", "크리스탈라운지", "(23)한우육회", 5),
    ("casino_fnb", "써미타스라운지", "(V)게토레이", 41),
    ("casino_fnb", "써미타스라운지", "아이스아메리카노", 49),
    ("casino_fnb", "써미타스라운지", "(23)송이한우갈비탕", 6),
    ("casino_fnb", "민트 바", "(21)산삼배양근진", 12),
    ("casino_fnb", "민트 바", "(22)블루베리 머핀", 22),
    ("casino_fnb", "민트 바", "(K)제주향 프리미엄 주스", 9),
    ("casino_fnb", "민트 바", "생맥주", 31),
    ("casino_fnb", "팬지", "(K)팬지 한우육회비빔밥", 8),
    ("casino_fnb", "팬지", "(K)우동김밥세트", 14),
    ("casino_fnb", "크리스탈라운지2", "콜라", 18),
    # 룸서비스 — 영업장 1개 (실측)
    ("roomservice", "In Room Dining(K)", "(VIP) A/C1", 3),
    ("roomservice", "In Room Dining(K)", "(VIP) F/C2", 3),
    ("roomservice", "In Room Dining(K)", "(VIP) F1", 2),
    ("roomservice", "In Room Dining(K)", "(VIP) D/S", 2),
    ("roomservice", "In Room Dining(K)", "1340 샴페인(프랑스)", 2),
    ("roomservice", "In Room Dining(K)", "1340 프리미엄 레드(이탈리아)", 2),
    ("roomservice", "In Room Dining(K)", "연인패키지(과일+와인)", 3),
    ("roomservice", "In Room Dining(K)", "한우 불고기 정식", 4),
    ("roomservice", "In Room Dining(K)", "한국식 조찬", 11),
    ("roomservice", "In Room Dining(K)", "간고등어정식", 10),
    ("roomservice", "In Room Dining(K)", "전복 된장찌개", 10),
    ("roomservice", "In Room Dining(K)", "공기밥", 12),
    # 지역특산품 — 영업장 1개 (실측)
    ("local_goods", "정태영삼 지역특산품점", "명품잡곡6종세트", 5),
    ("local_goods", "정태영삼 지역특산품점", "동해안 명품 오징어", 6),
    ("local_goods", "정태영삼 지역특산품점", "머루와인세트", 4),
    ("local_goods", "정태영삼 지역특산품점", "벌꿀선물세트", 7),
    ("local_goods", "정태영삼 지역특산품점", "12곡 프리미엄 파우더", 8),
    ("local_goods", "정태영삼 지역특산품점", "꿀삼진삼세트", 3),
    ("local_goods", "정태영삼 지역특산품점", "고춧가루(영월농협)", 14),
    ("local_goods", "정태영삼 지역특산품점", "정선 수리취떡", 16),
    ("local_goods", "정태영삼 지역특산품점", "삼척동자 맑은쌀", 11),
    ("local_goods", "정태영삼 지역특산품점", "오징어세트(신토불이)", 9),
)

# 실측 요일 인덱스 (월=0 … 일=6) — 구간 A(2024-12) 기준. 이 모듈이 생성하는 구간이다.
_DOW_INDEX: dict[str, tuple[float, ...]] = {
    "casino_fnb": (0.79, 0.86, 0.87, 0.80, 0.93, 1.27, 1.45),
    "roomservice": (0.78, 0.79, 0.89, 0.91, 0.93, 1.32, 1.39),
    "local_goods": (0.93, 1.08, 0.76, 0.64, 0.90, 0.92, 1.63),
}
# 실측 월 인덱스 (1월 … 12월) — 2023~2024 2년 기준. 회귀 기준값과 같은 창이다.
_MONTH_INDEX: dict[str, tuple[float, ...]] = {
    "casino_fnb": (1.02, 1.07, 0.97, 0.95, 0.99, 0.94, 1.01, 1.10, 1.02, 0.99, 0.92, 1.02),
    "roomservice": (1.21, 1.16, 1.09, 0.99, 1.02, 0.92, 0.89, 1.00, 0.90, 0.91, 0.88, 1.03),
    "local_goods": (1.04, 1.04, 0.86, 0.90, 0.96, 0.86, 0.92, 1.02, 1.43, 1.03, 0.95, 0.99),
}
# 채널별 반응 시차 — 특산품만 D+1.
# ⚠ 이것은 **구간 A(2024-12) 31일의 실측**이고, 이 모듈이 생성하는 구간도 거기다.
# 731일 창에서는 세 채널 모두 D+0 이다(특산품 D+0 0.313 > D+1 0.163). 폴백을
# 넓은 창으로 확장하게 되면 이 상수부터 D+0 으로 고쳐야 한다.
_LAG = {"casino_fnb": 0, "roomservice": 0, "local_goods": 1}

_YEAR, _MONTH = 2024, 12
_DAYS = 31


def _ars_records() -> list[dict]:
    rows = []
    for day, recv, recv_ars, win, tickets, rate in _ARS_ROWS:
        rows.append({
            "기준일자": dt.date(_YEAR, _MONTH, day).isoformat(),
            "내국인 총 접수자": recv,
            "ARS 접수자": recv_ars,
            "모바일 접수자": recv - recv_ars,
            "총 당첨자": win,
            "ARS 담청자": min(recv_ars, win),
            "모바일 담청자": max(win - recv_ars, 0),
            "당첨자 입장권 구매 건 수": tickets,
            "당첨자 입장권 구매율": rate,
        })
    return rows


def ars_df() -> pd.DataFrame:
    """ARS 예약현황 — 원본 한글 컬럼 형태 그대로 반환 (표준화는 로더가 한다)."""
    return pd.DataFrame(_ars_records())


def _inflow_by_day() -> dict[int, float]:
    """일자 → 유입 강도 배율 (평균 1.0 기준). 판매량 생성의 구동 신호."""
    tickets = {day: t for day, _, _, _, t, _ in _ARS_ROWS}
    mean = sum(tickets.values()) / len(tickets)
    return {d: v / mean for d, v in tickets.items()}


def sales_df(channel: str | None = None) -> pd.DataFrame:
    """판매현황 — 원본 한글 컬럼 형태. channel 지정 시 해당 채널만.

    유입 배율 × 요일지수 × 월지수 를 곱해 결정론적으로 생성한다. 난수가 없으므로
    호출마다 동일하며, 실데이터의 상관·시차·요일 구조가 재현된다.
    """
    inflow = _inflow_by_day()
    rows: list[dict] = []
    for ch, venue, item, base in _ITEMS:
        if channel and ch != channel:
            continue
        lag = _LAG[ch]
        dow_ix = _DOW_INDEX[ch]
        mon_ix = _MONTH_INDEX[ch][_MONTH - 1]
        for day in range(1, _DAYS + 1):
            date = dt.date(_YEAR, _MONTH, day)
            # 시차: 특산품은 전날 유입에 반응한다
            src_day = max(day - lag, 1)
            mult = inflow[src_day] * dow_ix[date.weekday()] * mon_ix
            qty = max(int(round(base * mult)), 1)
            row = {
                "영업일자": date.isoformat(),
                "영업장명": venue,
                "상품명": item,
                "상품영문명": "",
                "판매수량": qty,
            }
            if ch == "casino_fnb":
                row["영업장아이디(ID)"] = {
                    "크리스탈라운지": "2119", "써미타스라운지": "2108",
                    "민트 바": "2107", "팬지": "2109", "크리스탈라운지2": "2112",
                }[venue]
                row["상품아이디(ID)"] = f"FB{row['영업장아이디(ID)']}{_stable_num(item)}"
                row["할인구분여부"] = "N"
            elif ch == "roomservice":
                row["할인구분여부"] = "N"
            else:
                row["상품아이디(ID)"] = f"FB2118{_stable_num(item)}"
            rows.append(row)
    return pd.DataFrame(rows)


def sales_by_channel() -> dict[str, pd.DataFrame]:
    """채널별 원본형 DataFrame 딕셔너리 (로더가 채널마다 표준화한다)."""
    return {ch: sales_df(ch) for ch in ("casino_fnb", "roomservice", "local_goods")}


# ── 골프장 이용객 현황 ─────────────────────────────────────────────────
# 실측 구조를 모사한다: 영업장 4종, 시즌구분에 따른 이용객 편차, 겨울 정기휴장.
# 골프장은 4월~11월만 운영하므로(실측 최이른 개장 3-19, 최늦 폐장 12-07)
# 12~2월은 아예 행이 없다 — 이 계절 공백 자체가 데이터의 특징이다.
_GOLF_VENUE_BASE = {
    "그린피": 140, "카트대여": 35, "용품대여": 8, "드라이빙레인지": 22,
}
# (시작월, 시작일, 종료월, 종료일, 시즌명, 배율)
_GOLF_SEASONS: tuple[tuple[int, int, int, int, str, float], ...] = (
    (4, 1, 5, 15, "비수기A", 0.72),
    (5, 16, 6, 30, "준성수기A", 0.95),
    (7, 1, 8, 20, "성수기A", 1.34),
    (8, 21, 9, 30, "성수기B", 1.21),
    (10, 1, 10, 31, "성수기C", 1.28),
    (11, 1, 11, 30, "비수기B", 0.66),
)
_GOLF_DOW_INDEX = (0.82, 0.85, 0.88, 0.91, 1.18, 1.42, 1.31)   # 월…일
_GOLF_YEAR = 2024


def _golf_season(month: int, day: int) -> tuple[str, float] | None:
    for m0, d0, m1, d1, name, mult in _GOLF_SEASONS:
        if (month, day) >= (m0, d0) and (month, day) <= (m1, d1):
            return name, mult
    return None


def golf_df() -> pd.DataFrame:
    """골프장 이용객 현황 — 원본 한글 컬럼 형태 (표준화는 로더가 한다)."""
    rows: list[dict] = []
    date = dt.date(_GOLF_YEAR, 4, 1)
    end = dt.date(_GOLF_YEAR, 11, 30)
    while date <= end:
        season = _golf_season(date.month, date.day)
        if season is None:
            date += dt.timedelta(days=1)
            continue
        season_name, mult = season
        dow = date.weekday()
        # 매월 첫 월요일은 정기휴장 (실측에 '정기휴장(골프)' 시즌구분이 있다)
        closed = dow == 0 and date.day <= 7
        day_type = ("휴장" if closed
                    else "주말" if dow == 5
                    else "일요일" if dow == 6
                    else "금요일" if dow == 4
                    else "주중")
        for venue, base in _GOLF_VENUE_BASE.items():
            qty = 1 if closed else max(
                int(round(base * mult * _GOLF_DOW_INDEX[dow])), 1
            )
            rows.append({
                "영업일자": date.isoformat(),
                "시즌구분": "정기휴장(골프)" if closed else season_name,
                "주중주말": day_type,
                "요일": "월화수목금토일"[dow],
                "영업장": venue,
                "영업상태": "휴장영업" if closed else "정상영업",
                "이용인원": qty,
            })
        date += dt.timedelta(days=1)
    return pd.DataFrame(rows)


# ── 고객성별연령분석현황 ───────────────────────────────────────────────
_DEMOGRAPHICS = {
    "성별": ["남"] * 5 + ["여"] * 5,
    "연령대": ["20대", "30대", "40대", "50대", "60대이상"] * 2,
    "방문고객수": [9_400, 28_000, 31_200, 27_600, 14_900,
                   4_700, 14_900, 16_700, 15_300, 8_100],
}


def demographics_df() -> pd.DataFrame:
    """성별·연령대별 고객수. VIP 비율(40~60대) = 0.666 — 실측과 같게 맞춘 값이다.

    이 비율이 VRB(=입장권 구매 건수 × vip_ratio)의 유일한 입력이므로, 폴백에서
    어긋나면 VIP 모수 카드가 실측과 다른 규모를 말한다. 한동안 0.752 였다.
    """
    return pd.DataFrame(_DEMOGRAPHICS)


# ── 하이원포인트 가맹점 ────────────────────────────────────────────────
# 강원랜드(37.2007, 128.8155) 주변 정선·태백·영월 일대 좌표
_MERCHANTS: tuple[tuple[str, float, float, str], ...] = (
    ("하이원 그랜드호텔", 37.2015, 128.8172, "숙박"),
    ("하이원 컨벤션호텔", 37.1998, 128.8203, "숙박"),
    ("하이원 밸리호텔", 37.1876, 128.8261, "숙박"),
    ("하이원 컨트리클럽", 37.1802, 128.8397, "골프"),
    ("하이원 스키장", 37.2103, 128.8266, "레저"),
    ("정선 아리랑시장", 37.3805, 128.6606, "쇼핑"),
    ("정선 곤드레마을", 37.3812, 128.6650, "음식"),
    ("사북 한우마당", 37.2201, 128.8083, "음식"),
    ("고한 약초시장", 37.1938, 128.8501, "특산"),
    ("태백 황지연못상회", 37.1640, 128.9856, "특산"),
    ("영월 동강한과", 37.1836, 128.4617, "특산"),
    ("삼탄아트마인 카페", 37.2492, 128.7681, "카페"),
    ("정선 레일바이크", 37.3417, 128.7500, "레저"),
    ("만항재 산채정식", 37.1745, 128.8829, "음식"),
    ("사북 주유소", 37.2172, 128.8046, "주유"),
    ("고한 종합병원", 37.1912, 128.8467, "의료"),
    ("정선 로컬푸드직매장", 37.3790, 128.6588, "농산"),
    ("하이원 워터월드", 37.1969, 128.8324, "레저"),
    ("태백 고원자생식물원", 37.1502, 129.0102, "레저"),
    ("영월 청령포 기념품점", 37.1802, 128.4489, "쇼핑"),
)


def merchants_df() -> pd.DataFrame:
    """하이원포인트 가맹점 — 원본 한글 컬럼 형태."""
    return pd.DataFrame(
        [{"가맹점명": n, "위도": la, "경도": lo, "업종구분": c}
         for n, la, lo, c in _MERCHANTS]
    )
