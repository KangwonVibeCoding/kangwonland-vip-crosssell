"""프로젝트 전역 설정 — 경로, 컬럼 매핑, VIP 티어 패턴, 구간 정의, 스코어 가중치.

여기가 단일 설정 소스다. 상수를 하드코딩해야 할 일이 생기면 이 파일에 넣는다.
streamlit 을 import 하지 않으므로 pytest 에서 자유롭게 쓸 수 있다.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

# ── 경로 (전부 절대 경로. Cloud 의 작업 디렉토리 차이에 영향받지 않는다) ──
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INCOMING_DIR = DATA_DIR / "incoming"     # 원본 그대로 (한글 파일명, CP949)
RAW_DIR = DATA_DIR / "raw"               # ingest 산출물 (ASCII 파일명, UTF-8)
SAMPLE_DIR = DATA_DIR / "sample"         # 축약본 (커밋 대상 — Cloud 데모용)
MOCK_DIR = DATA_DIR / "mock"             # 합성 데이터 (커밋 대상 — 폴백)
PROCESSED_DIR = DATA_DIR / "processed"   # parquet 캐시
API_CACHE_DIR = RAW_DIR / "api_cache"    # Open API 응답 디스크 캐시
CSS_PATH = ROOT / "assets" / "styles.css"
INGEST_REPORT = RAW_DIR / "_ingest_report.txt"

# ── Open API (공공데이터포털) ──────────────────────────────────────────
# ⚠ 런타임에 이 API 를 호출하는 것에 의존하지 않는다. `scripts/fetch_api_data.py`
# 로 **빌드 타임에 한 번** 받아 data/raw/*.csv 로 떨구고, 앱은 그 파일을 L2/L3 로
# 읽는다. L4(런타임 호출)는 선택 계층으로만 남긴다.
#
# 엔드포인트는 **전체 URL** 이다. 포털 계열마다 페이지 파라미터가 다르다:
#   apis.data.go.kr  → pageNo / numOfRows   (응답: response.body.items)
#   api.odcloud.kr   → page   / perPage     (응답: data)
# 섞어 보내면 HTTP_ERROR 가 난다 (실측).
EP_DEMOGRAPHICS = (
    "https://apis.data.go.kr/B552525/CustSexdstnAgeAnlsCnt/getCustSexdstnAgeAnlsCnt"
)
EP_MERCHANTS = "https://apis.data.go.kr/B552525/pbdata/getStoreInfo"
API_TIMEOUT = 8
API_PAGE_SIZE = 1000

# ⚠ 성별연령 엔드포인트는 dateFrom/dateTo 가 **필수**다. 빠뜨리면 서버가
# `HTTP_ERROR(코드 04)` 를 돌려준다 — 04 는 보통 '제공기관 서버 다운'을 뜻해서
# 2026-08-04 에 이를 서버 장애로 오진했다. 실제로는 우리 요청이 불완전했던 것이고,
# 날짜를 넣으면 그때도 200 이 왔을 것이다(2026-08-08 확인).
# 형식은 `YYYY-MM-DD` 고정 — `20260601` 은 `DATE_STRING_PATTERN_MISSMATCH(33)`.
# pageNo/numOfRows 도 함께 보내야 한다(날짜만 보내면 빈 응답).
# 범위는 넓게 잡아도 서버가 보유분만 돌려준다. 실보유는 2026-06-10~08-06 뿐이라
# 판매(2023~2024)·구간 A(2024-12)와 겹치지 않는다 — 2024-12 조회는 totalCount 0.
DEMO_PARAMS = {"dateFrom": "2023-01-01", "dateTo": "2026-12-31"}

# ── 지리 기준점: 하이원리조트 / 강원랜드 ───────────────────────────────
CASINO_LATLON = (37.2007, 128.8155)
PROXIMITY_DECAY_KM = 5.0
DEFAULT_RADIUS_KM = 15.0

# ── 채널 ──────────────────────────────────────────────────────────────
CH_CASINO = "casino_fnb"
CH_ROOM = "roomservice"
CH_LOCAL = "local_goods"
CHANNELS = (CH_CASINO, CH_ROOM, CH_LOCAL)
CHANNEL_LABEL = {
    CH_CASINO: "카지노 식음",
    CH_ROOM: "룸서비스",
    CH_LOCAL: "지역특산품",
}
# 실측: 룸서비스·특산품은 영업장이 1개뿐이라 영업장 필터가 무의미하다.
MULTI_VENUE_CHANNELS = (CH_CASINO,)

# ── 분석 구간 (실데이터 기간 불일치를 정직하게 다루기 위한 3구간 설계) ──
#   A: ARS + 판매가 모두 있는 유일한 구간 → 진짜 날짜 조인, 상관·래그 실측
#   B: 판매만 있는 구간 → CAI(객장 활동 지수) + 요일 브릿지
#   C: ARS만 있는 구간 → 최신 유입 트렌드 참조
#
# ⚠ 구간 A 는 `data/raw/ars_20241201_20241231.csv` 에 의존한다. 2026년 ARS
#   전처리본(ars_merged.csv)에는 2024-12 이 없어서, 이 파일이 빠지면 ARS 와
#   판매가 겹치는 날이 0일이 되고 상관·래그 분석이 전부 무너진다.
#   복원: python scripts/restore_ars_2024.py
PERIOD_A = "A"
PERIOD_B = "B"
PERIOD_C = "C"
PERIOD_CUSTOM = "custom"

PERIODS: dict[str, dict] = {
    PERIOD_A: {
        "label": "정밀 검증 · 2024-12 (ARS 실측 조인)",
        "badge": "실측 조인",
        "start": dt.date(2024, 12, 1),
        "end": dt.date(2024, 12, 31),
        "has_ars": True,
        "has_sales": True,
    },
    PERIOD_B: {
        "label": "연간 확장 · 2023~2024 (요일 브릿지)",
        "badge": "요일 브릿지",
        "start": dt.date(2023, 1, 1),
        "end": dt.date(2024, 12, 31),
        "has_ars": True,     # 2024-12 분만 존재 → 부분 조인
        "has_sales": True,
    },
    PERIOD_C: {
        "label": "최신 유입 · 2026 상반기 (판매 데이터 없음)",
        "badge": "최신 유입",
        "start": dt.date(2026, 1, 1),
        "end": dt.date(2026, 6, 30),
        "has_ars": True,
        "has_sales": False,
    },
}
DEFAULT_PERIOD = PERIOD_A

# ── 컬럼 매핑 ─────────────────────────────────────────────────────────
# 키는 normalize_key() 를 통과한 값이다 (공백·괄호·특수문자 제거 + 소문자).
# 원본의 오타('ARS 담청자')와 불규칙 공백('구매 건 수')을 흡수하기 위한 설계.
COLUMN_MAP: dict[str, str] = {
    # 공통 날짜
    "기준일자": "date",
    "영업일자": "date",
    # ARS 예약현황
    "내국인총접수자": "recv_total",
    "ars접수자": "recv_ars",
    "모바일접수자": "recv_mobile",
    "총당첨자": "winners",
    "ars담청자": "win_ars",       # 원본 오타
    "ars당첨자": "win_ars",       # 정정판 대응
    "모바일담청자": "win_mobile",  # 원본 오타
    "모바일당첨자": "win_mobile",
    "당첨자입장권구매건수": "tickets",
    "당첨자입장권구매율": "buy_rate",
    # ARS 전처리본(ars_merged.csv) — 컬럼명이 이미 영문이고 축약돼 있다.
    # ⚠ 총접수자·모바일 접수자가 빠져 있어 recv_* 계열이 결측이 된다.
    #   CII 는 이 결측을 감지해 tickets/buy_rate 2축으로 폴백한다.
    "winnerstotal": "winners",
    "ticketpurchases": "tickets",
    "purchaserate": "buy_rate",
    # 이미 표준명인 컬럼을 그대로 통과시킨다 (정규화된 CSV 재입력 대응)
    "recvtotal": "recv_total",
    "recvars": "recv_ars",
    "recvmobile": "recv_mobile",
    "winners": "winners",
    "winars": "win_ars",
    "winmobile": "win_mobile",
    "tickets": "tickets",
    "buyrate": "buy_rate",
    # 판매 3종
    "영업장아이디id": "venue_id",
    "영업장명": "venue",
    "상품아이디id": "item_id",
    "상품명": "item",
    "상품영문명": "item_en",
    "할인구분여부": "is_discount",
    "판매수량": "qty",
    # 골프장 이용객 현황 (2021~2025). 판매 데이터와 컬럼명이 겹치지 않는다 —
    # '영업장'(golf) vs '영업장명'(판매) 은 normalize_key 후에도 서로 다르다.
    "시즌구분": "season",
    "주중주말": "day_type",
    "요일": "dow_label",
    "영업장": "golf_venue",
    "영업상태": "open_status",
    "이용인원": "visitors",
    # 인구통계 (Open API — apis.data.go.kr/B552525/CustSexdstnAgeAnlsCnt)
    # 실제 응답 필드는 영문 코드명이고, BSN_DT 로 **일자별**로 온다.
    "성별": "gender",
    "연령대": "age_band",
    "방문고객수": "visitors",
    "bsndt": "date",
    "sexdstnsenm": "gender",
    "custagenm": "age_band",
    "custwholcnt": "visitors",
    # 가맹점 (Open API — apis.data.go.kr/B552525/pbdata/getStoreInfo)
    # 실제 응답 필드는 영문 코드명이다. 위경도·업종 컬럼은 **응답에 없다** —
    # scripts/geocode_merchants.py 가 주소를 좌표·업종으로 바꿔 채운다.
    "frcsnm": "merchant",
    "frcsaddr": "address",
    "frcsregno": "merchant_id",
    "frcsbrno": "biz_no",
    "frcstelno": "tel",
    # ⚠ PNT_USABLE_AMT 는 가맹점별 포인트 '사용 한도액'이라 1,681건 중 1,578건
    # (93.9%)이 정확히 4,000,000 이다 (CV 0.052). ARS 의 winners 와 같은 성격의
    # 상수 컬럼이므로 스코어 신호로 쓰지 않는다 — 표시용으로만 매핑한다.
    "pntusableamt": "point_limit",
    "가맹점명": "merchant",
    "위도": "lat",
    "경도": "lon",
    "업종구분": "category",
}

# recv_total 은 필수에서 뺐다. 2026년 ARS 전처리본에는 총접수자 컬럼이 없고,
# 그 기간에도 tickets/buy_rate 만으로 유입 지수를 낼 수 있어야 하기 때문이다.
ARS_REQUIRED = ("date", "winners", "tickets", "buy_rate")
# 있으면 쓰고 없으면 결측으로 두는 컬럼. 0 으로 채우지 않는다 — 접수자 0명이라는
# 거짓을 만들고 상관·정규화를 통째로 왜곡한다.
ARS_OPTIONAL = ("recv_total", "recv_ars", "recv_mobile", "win_ars", "win_mobile")
SALES_REQUIRED = ("date", "venue", "item", "qty")
DEMO_REQUIRED = ("gender", "age_band", "visitors")
MERCHANT_REQUIRED = ("merchant", "lat", "lon", "category")
GOLF_REQUIRED = ("date", "golf_venue", "visitors")

# ── 골프장 이용객 (2021-03 ~ 2025-12) ──────────────────────────────────
# 판매 3종과 기간이 거의 겹치지 않고(판매는 2023~2024) 단위도 다르므로 채널로
# 합치지 않는다. 별도 데이터셋으로 로드만 해 두고 UI 노출은 하지 않는다.
GOLF_VENUES = ("그린피", "카트대여", "용품대여", "드라이빙레인지")
GOLF_VENUE_PRIMARY = "그린피"          # 실제 라운딩 인원 = 이용객 수의 기준
GOLF_CLOSED_STATUSES = ("휴장영업",)   # 집계에서 제외할 영업상태

# ── VIP / 마진 프록시 (실데이터 분석으로 2티어 확정) ────────────────────
# 단일 정규식이면 특산품 히트율이 31.4% 로 과대해진다(실측). '명품잡곡6종세트'와
# '간편톡 선물세트'를 구분해야 큐레이션이 의미를 갖는다.
TIER_PREMIUM_PATTERN = (
    r"VIP|V\.I\.P|프리미엄|명품|한우|와인|샴페인|캐비어|산삼|시그니처|1340|"
    r"송이|홍삼|더덕주|오가피주|육포"
)
TIER_BUNDLE_PATTERN = r"선물세트|세트|패키지|플래터|기프트|선물"

# 무상 제공(컴프/바우처) 상품 — 마진 기여가 0이므로 고마진 집계에서 반드시 분리한다.
#   '^(V)'  : '(V)에비앙' 등 11종 21,870개 (전체 수량 0.57%)
#   '무료'   : '18)헛개차(무료)', '24)사과(무료)' 등 38종 210,372개 (5.5%)
# ⚠ VIP 정규식에 단독 V 를 넣으면 대량 오탐이 발생하므로 접두 형태로만 매칭한다.
COMP_PATTERN = r"^\s*\(V\)|무료|\(무\)"

TIER_PREMIUM = "premium"
TIER_BUNDLE = "bundle"
TIER_STANDARD = "standard"
TIERS = (TIER_PREMIUM, TIER_BUNDLE, TIER_STANDARD)
TIER_LABEL = {
    TIER_PREMIUM: "프리미엄",
    TIER_BUNDLE: "선물번들",
    TIER_STANDARD: "일반",
}
TIER_MARGIN_BASE = {TIER_PREMIUM: 1.0, TIER_BUNDLE: 0.7, TIER_STANDARD: 0.4}
RARITY_BONUS = 0.2          # margin_proxy 에 더해지는 희소도 보너스 상한
DISCOUNT_PENALTY = 0.1      # 할인 판매 비중이 높은 상품에 대한 감점

# ── VIP 세그먼트 ──────────────────────────────────────────────────────
AGE_BANDS_ALL = ("20대", "30대", "40대", "50대", "60대이상")
AGE_BANDS_VIP = ("40대", "50대", "60대이상")     # 고소비층
AGE_BANDS_SEMI = ("30대",)
GENDER_OPTIONS = ("전체", "남", "여")

# ── 스코어 가중치 ─────────────────────────────────────────────────────
# CII: 원안의 winners(총 당첨자)는 일일 정원 캡(최대 2,999)으로 거의 상수라
#      제외했다. 실측 CV — recv_total 0.372 / tickets 0.302 / winners 0.178
INFLOW_WEIGHTS = {"demand": 0.45, "pressure": 0.35, "convert": 0.20}
# recv_total(수요 압력)이 없는 구간용 폴백 — 압력 축만 빼고 나머지 비율은 그대로
# 둔다. wsum 이 남은 가중치 합으로 재정규화하므로 실효 비율은 0.692:0.308 이 된다.
# 값을 직접 적지 않고 파생시키는 이유: INFLOW_WEIGHTS 를 조정했을 때 폴백이
# 따라오지 않으면 구간마다 지수의 성격이 조용히 달라진다.
# 2026년 실측 CV — tickets 0.258 / buy_rate 0.139 / winners 0.155(여전히 캡)
INFLOW_WEIGHTS_NO_PRESSURE = {
    k: v for k, v in INFLOW_WEIGHTS.items() if k != "pressure"
}
CAI_WEIGHTS = {"volume": 0.60, "breadth": 0.25, "dow": 0.15}
VTS_WEIGHTS = {"inflow": 0.35, "base": 0.25, "proven": 0.20, "headroom": 0.20}
# ⚠ base(VIP 모수) 축은 성별·연령이 **기간 집계**로만 제공되는 한 유입 축의 복사본이다.
# vip_ratio 가 상수라서 VRB = tickets × 상수 이고, 실측 corr(inflow, vrb)=0.9988 이다.
# 두 축을 함께 쓰면 유입에 0.60 을 준 것과 같아져 VTS 가 유입 지수의 재탕이 된다
# (실측 corr(inflow, vts)=0.981 · 상위 10일 중 9일이 유입 상위일과 동일).
# 그래서 중복이 감지되면 base 를 빼고 **그 가중치를 headroom 으로 넘긴다** —
# 균등 분배가 아니라 headroom 인 이유는 이 지수의 존재 이유가 "이미 잘 파는 날"이
# 아닌 날을 고르는 것이기 때문이다. 실측 corr(inflow, vts) 0.981 → 0.889.
# 성별·연령 API 가 일자별 데이터를 주면 corr 이 떨어져 base 축이 자동으로 살아난다.
VTS_BASE_REDUNDANT_R = 0.98   # |corr(inflow, vrb)| 이 이 값 이상이면 같은 신호로 본다
VTS_WEIGHTS_NO_BASE = {
    **{k: v for k, v in VTS_WEIGHTS.items() if k != "base"},
    "headroom": VTS_WEIGHTS["headroom"] + VTS_WEIGHTS["base"],
}
CSM_WEIGHTS = {"elasticity": 0.45, "scale": 0.30, "margin": 0.25}
# 교차판매 적합도 = 거리 + 업종. 두 축은 서로 독립이라 합성이 의미를 갖는다.
# (밀집도를 넣는 안은 버렸다 — 사북·고한은 가장 가깝고 동시에 가맹점이 가장
#  많아 근접도와 같은 신호가 된다. VRB 가 유입의 상수배라 VTS 를 무의미하게
#  만든 것과 같은 붕괴다.)
CURATION_WEIGHTS = {"proximity": 0.60, "fit": 0.40}

CAPACITY_CAP = 2990          # winners >= 이 값이면 정원 소진일로 본다
HI_LO_QUANTILE = 0.30        # lift 계산의 고유입/저유입 분위
MAX_LAG_DEFAULT = 3
ROBUST_CLIP = (0.05, 0.95)   # nrm(robust=True) 의 분위 클리핑 범위

# lift 는 통계량이므로 표본 요건이 필요하다. 판매일수 2일짜리 상품도 lift 20~60 이
# 나오는데, 그걸 근거로 번들을 추천하면 실행 불가능한 제안이 된다.
#
# ⚠ 요건을 **절대값으로만** 두면 창 길이가 바뀔 때 게이트가 사실상 사라진다.
# 31일 창에서 '5일·10개'는 의미 있는 문턱이지만 731일 창에서는 거의 전량 통과라
# (실측 1,688종 중 1,534종) lift 가 24,027 까지 나오고 CSM 상위를 잡음이 점령했다.
# 그래서 **창 길이에 비례한 요건**을 함께 걸고 둘 중 큰 값을 쓴다
# (`crosssell.sample_gate`). 짧은 창에서는 절대값이, 긴 창에서는 비율이 지배한다.
#
# 수량 요건은 **채널 안에서** 정한다. 채널 간 판매량 스케일이 20배 이상 다르기
# 때문에(카지노 식음 상품 평균 6,131 vs 룸서비스 613, 731일 창 실측) 전 채널 공통
# 절대값을 쓰면 한쪽은 전부 통과하고 한쪽은 전멸한다. 실제로 '창일수×1' 을 걸었더니
# 731일 창에서 특산품 프리미엄 상품(2년 622개)이 통째로 탈락했다.
# 채널 중앙값을 쓰는 이유는 평균이 히트 상품 하나에 끌려가기 때문이다
# (특산품 평균 1,505 vs 중앙값 356 — '으랏차차' 65,254개가 평균을 끌어올린다).
LIFT_MIN_DAYS = 5            # 미달 시 lift=NaN → CSM 의 elasticity 는 중립(0.5)
LIFT_MIN_QTY = 10            # 절대 하한 (짧은 창 대비)
LIFT_MIN_DAY_RATIO = 0.20    # 판매일수 ≥ 창 길이의 20%
LIFT_MIN_QTY_QUANTILE = 0.50  # 수량 ≥ 그 채널 상품 판매량의 중앙값
BUNDLE_MIN_DAYS = 8          # 번들 추천은 더 보수적으로 (상시 판매 상품만)
BUNDLE_MIN_DAY_RATIO = 0.25  # 31일 창의 8일과 같은 비율 — 긴 창에서도 상시 판매만

# 표본 미달로 lift 를 못 구한 상품의 탄력성을 무엇으로 채울 것인가.
# `nrm` 기본값 0.5 는 **중립이 아니다** — 실측 상품의 탄력성 중앙값이 0.24(731일 창)
# 라서 0.5 는 측정된 상품의 82%보다 높은 값이고, "표본이 부족하다"가 점수 보너스가
# 된다. 실측 분포의 중앙값으로 채우면 미측정 상품은 1위가 될 수도, 보너스를 받을
# 수도 없는 '평범한 상품' 대우가 된다. (축을 빼고 재정규화하는 안은 반대로
# 뒤집혔다 — 물량·마진이 최상인 미측정 상품이 100점 1위가 된다. crosssell 주석 참조)
CSM_ELASTICITY_FILL_QUANTILE = 0.50
CSM_ELASTICITY_FILL_DEFAULT = 0.5   # 측정된 상품이 하나도 없을 때만 쓰는 최후값

GRADE_BINS = ((75, "피크"), (50, "성수"), (25, "보통"), (0, "한산"))

# ── 요일 교란 통제 (편상관) ────────────────────────────────────────────
# 토요일엔 유입도 많고 매출도 많다. 즉 요일이 양변을 동시에 밀어올리는 **공통
# 원인**일 수 있어서, 원 상관 0.801 만으로는 "유입 → 소비"를 주장할 수 없다.
# 요일 더미 6개로 양변을 회귀한 **잔차끼리의 상관**(편상관)을 같이 계산해 화면에
# 나란히 세운다 — 통제 후에도 남는 관계만이 마케팅으로 개입할 수 있는 부분이다.
CONFOUND_MIN_DAYS = 14       # 절편 + 요일 더미 6개를 빼고도 자유도가 남는 최소선
CONFOUND_RESAMPLES = 10_000  # 부트스트랩 CI · 순열검정 반복수
CONFOUND_SEED = 42           # 고정 시드 — 새로고침마다 흔들리는 수치는 근거가 아니다
# p 임계값과 판정 문구. 화면·README·처방 카드가 같은 말을 쓰게 한 곳에 둔다.
CONFOUND_VERDICTS = ((0.01, "견고"), (0.05, "간신히 유의"))
CONFOUND_VERDICT_NULL = "유의하지 않음"

# ── 가맹점 업종 적합도 ─────────────────────────────────────────────────
# 하이원포인트 가맹점 API 는 업종을 주지 않는다. 업종은 소상공인시장진흥공단
# 상가(상권)정보의 **상권업종대분류명**(표준산업분류 기반 공식 체계)을 조인해
# 채운다 (scripts/join_merchant_geo.py). 아래 키는 그 분류의 실제 값이다.
#
# ⚠ 이전 판은 '골프' '스키' '특산' 처럼 **그 분류체계에 존재하지 않는** 이름에
# 가중치를 매기고 있었다. 실데이터에서는 전부 기본값으로 떨어져 지표가 사실상
# 상수였다. 실제 관측된 값(음식 579 / 소매 425 / 수리·개인 169 / 숙박 115 /
# 예술·스포츠 15 / 과학·기술 6 / 교육 2)에 맞춰 다시 정의한다.
#
# 가중치는 데이터에서 유도한 값이 아니라 **판단**이다. 그래서 이름을 '프리미엄'이
# 아니라 '적합도'로 두고, 이 표를 UI 에 그대로 노출한다 — 숨긴 판단은 과장이지만
# 공개한 판단은 검증 가능한 설계다. 기준은 '카지노 방문객의 체류 동선과 겹치는가'.
CATEGORY_FIT = {
    "소매": 1.0,          # 특산품·기념품 — 교차판매 대상 그 자체
    "음식": 0.9,          # 체류 중 식사 동선
    "숙박": 0.8,          # 연박 유도
    "예술·스포츠": 0.6,    # 레저 연계 체험
    "보건의료": 0.2,
    "수리·개인": 0.2,      # 미용·세탁 — 관광 동선이 아니다
    "과학·기술": 0.1,
    "교육": 0.1,
    "부동산": 0.1,
    "시설관리·임대": 0.1,
}
# 업종 미상(주소로만 좌표를 찾은 건)은 적합도를 **중립 0.5** 로 둔다.
# 0 으로 두면 '부적합'이라는 없는 근거를 만들고, 1 로 두면 그 반대다.
# ⚠ 여기서 0.5 가 중립인 이유는 이 축이 **0~1 을 판단으로 직접 매긴 표**(CATEGORY_FIT)
# 라서 한가운데가 실제로 한가운데이기 때문이다. lift 의 elasticity 축은 관측 분포를
# 정규화한 값이라 0.5 가 한가운데가 아니고(실측 중앙값 0.24), 그래서 그쪽은
# 실측 중앙값으로 채운다(CSM_ELASTICITY_FILL_QUANTILE). 같은 '0.5' 라도 근거가 다르다.
CATEGORY_FIT_NEUTRAL = 0.5

# 지도 마커 색 — 전체쌍 색약 검증을 통과한 3색까지만 쓴다. 실측 분포상
# 음식·소매·숙박이 전체의 92% 라 이 셋을 색으로 두고 나머지는 회색으로 눕힌다.
CATEGORY_GROUPS = {"음식": "음식", "소매": "소매", "숙박": "숙박"}
CATEGORY_GROUP_OTHER = "기타"

# ── 캐싱 / 성능 ───────────────────────────────────────────────────────
CACHE_TTL = 3600
TOP_N_ITEMS = 15
TOP_N_LOCAL = 20
TOP_N_VTS_DAYS = 10

# ── 데이터 출처 라벨 (4계층 폴백) ──────────────────────────────────────
SRC_PROCESSED = "PROCESSED"
SRC_RAW = "RAW"
SRC_SAMPLE = "SAMPLE"
SRC_API = "API"
SRC_MOCK = "MOCK"
SRC_EMBEDDED = "EMBEDDED"
SOURCE_LABEL = {
    SRC_PROCESSED: ("실데이터", "캐시된 원본 전처리 결과"),
    SRC_RAW: ("실데이터", "data/raw 원본 CSV"),
    SRC_SAMPLE: ("실데이터(축약)", "data/sample 축약본"),
    SRC_API: ("실시간 API", "data.go.kr Open API"),
    SRC_MOCK: ("모킹", "합성 데이터"),
    SRC_EMBEDDED: ("내장 데모", "코드 내장 상수 — 최종 폴백"),
}
REAL_SOURCES = (SRC_PROCESSED, SRC_RAW, SRC_SAMPLE, SRC_API)

DOW_NAMES = ("월", "화", "수", "목", "금", "토", "일")
