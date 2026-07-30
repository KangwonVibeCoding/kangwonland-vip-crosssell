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
API_BASE = "https://api.odcloud.kr/api"
EP_DEMOGRAPHICS = "15083033/v1/uddi:demographics"   # 고객성별연령분석현황
EP_MERCHANTS = "15083033/v1/uddi:merchants"         # 하이원포인트 가맹점 상세정보
API_TIMEOUT = 8
API_PAGE_SIZE = 1000

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
        "label": "연간 확장 · 2024 전체 (요일 브릿지)",
        "badge": "요일 브릿지",
        "start": dt.date(2024, 1, 1),
        "end": dt.date(2024, 12, 31),
        "has_ars": True,     # 12월분만 존재 → 부분 조인
        "has_sales": True,
    },
    PERIOD_C: {
        "label": "최신 유입 · 2026-05 (판매 데이터 없음)",
        "badge": "최신 유입",
        "start": dt.date(2026, 5, 1),
        "end": dt.date(2026, 5, 31),
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
    # 판매 3종
    "영업장아이디id": "venue_id",
    "영업장명": "venue",
    "상품아이디id": "item_id",
    "상품명": "item",
    "상품영문명": "item_en",
    "할인구분여부": "is_discount",
    "판매수량": "qty",
    # 인구통계 (Open API)
    "성별": "gender",
    "연령대": "age_band",
    "방문고객수": "visitors",
    # 가맹점 (Open API)
    "가맹점명": "merchant",
    "위도": "lat",
    "경도": "lon",
    "업종구분": "category",
}

ARS_REQUIRED = ("date", "recv_total", "winners", "tickets", "buy_rate")
SALES_REQUIRED = ("date", "venue", "item", "qty")
DEMO_REQUIRED = ("gender", "age_band", "visitors")
MERCHANT_REQUIRED = ("merchant", "lat", "lon", "category")

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
CAI_WEIGHTS = {"volume": 0.60, "breadth": 0.25, "dow": 0.15}
VTS_WEIGHTS = {"inflow": 0.35, "base": 0.25, "proven": 0.20, "headroom": 0.20}
CSM_WEIGHTS = {"elasticity": 0.45, "scale": 0.30, "margin": 0.25}
CURATION_WEIGHTS = {"proximity": 0.60, "premium": 0.40}

CAPACITY_CAP = 2990          # winners >= 이 값이면 정원 소진일로 본다
HI_LO_QUANTILE = 0.30        # lift 계산의 고유입/저유입 분위
MAX_LAG_DEFAULT = 3
ROBUST_CLIP = (0.05, 0.95)   # nrm(robust=True) 의 분위 클리핑 범위

# lift 는 통계량이므로 표본 요건이 필요하다. 판매일수 2일짜리 상품도 lift 20~60 이
# 나오는데, 그걸 근거로 번들을 추천하면 실행 불가능한 제안이 된다.
LIFT_MIN_DAYS = 5            # 미달 시 lift=NaN → CSM 의 elasticity 는 중립(0.5)
LIFT_MIN_QTY = 10
BUNDLE_MIN_DAYS = 8          # 번들 추천은 더 보수적으로 (상시 판매 상품만)

GRADE_BINS = ((75, "피크"), (50, "성수"), (25, "보통"), (0, "한산"))

# 업종별 프리미엄 가중 (가맹점 큐레이션). 부분 문자열 매칭.
CATEGORY_PREMIUM = {
    "숙박": 1.0, "호텔": 1.0, "콘도": 0.9,
    "골프": 1.0, "레저": 0.8, "스키": 0.8,
    "음식": 0.7, "식당": 0.7, "카페": 0.5,
    "특산": 0.9, "농산": 0.8, "쇼핑": 0.6, "판매": 0.6,
    "주유": 0.3, "의료": 0.3, "기타": 0.2,
}
CATEGORY_PREMIUM_DEFAULT = 0.4

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
