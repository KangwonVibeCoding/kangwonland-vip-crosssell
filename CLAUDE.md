# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트

강원랜드 공공데이터 기반 **VIP 카지노–리조트 교차판매 마케팅 대시보드** (Streamlit).
카지노 유입 → VIP 가용 모수 → 채널별 반응 시차 → 교차판매 대상 상품을 하나의
스코어링 체인으로 잇는다. 해커톤 제출물이며 Streamlit Community Cloud 배포가 목표다.

전체 구현 계획과 2주 캘린더: `C:\Users\sinsh\.claude\plans\python-fancy-lighthouse.md`

## 명령어

Windows PowerShell 기준. venv 의 인터프리터를 **직접 경로로 호출**한다 (activate 없이도 동작).

```powershell
# 의존성
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

# 원본 CSV 정규화: data/incoming (CP949, 한글 파일명) → data/raw (UTF-8, ASCII)
$env:PYTHONIOENCODING="utf-8"; & ".\.venv\Scripts\python.exe" scripts\ingest.py

# 구간 A(2024-12 ARS) 복원 — data/raw 에 ars_20241201_20241231.csv 가 없을 때
& ".\.venv\Scripts\python.exe" scripts\restore_ars_2024.py

# 배포용 축약본 재생성: data/raw → data/sample (커밋 대상)
& ".\.venv\Scripts\python.exe" scripts\make_sample.py

# 테스트
& ".\.venv\Scripts\python.exe" -m pytest tests\ -q
& ".\.venv\Scripts\python.exe" -m pytest tests\test_stats.py -q                    # 파일 하나
& ".\.venv\Scripts\python.exe" -m pytest tests\test_stats.py::test_headline_corr -v  # 테스트 하나
& ".\.venv\Scripts\python.exe" -m pytest tests\ -q -k "lag"                        # 이름 매칭

# 앱 실행
& ".\.venv\Scripts\python.exe" -m streamlit run app.py
```

**한글 출력이 깨지면** `$env:PYTHONIOENCODING="utf-8"` 을 앞에 붙인다. 콘솔 인코딩
문제일 뿐 데이터는 정상이다.

**PowerShell `-c "..."` 로 파이썬 코드를 인라인 전달하면 따옴표가 소실된다.**
임시 스크립트 파일에 쓰고 실행하는 편이 확실하다.

**스키마·전처리를 바꿨으면 `data/processed/*.parquet` 을 지운다.** 로더가 parquet
캐시를 L1 으로 먼저 읽으므로, 지우지 않으면 옛 dtype·옛 티어 분류가 계속 나온다.

```powershell
Remove-Item data\processed\*.parquet -ErrorAction SilentlyContinue
```

## 아키텍처

### 계층 경계 (반드시 지킬 것)

```
app.py                오케스트레이션만 — 로드 → 필터 → ctx → 탭 렌더. 로직 금지
src/views/*.py        render(ctx) 단일 시그니처. ctx = {data, filters, sources}
src/ui/*.py           streamlit / plotly / pydeck 를 아는 유일한 층
src/analysis/*.py     DataFrame in / DataFrame out 순수 함수. streamlit import 금지
src/data/*.py         pandas 만. streamlit import 금지
config/settings.py    단일 설정 소스 (경로·컬럼맵·패턴·가중치·구간 정의)
```

`src/data`, `src/analysis` 가 streamlit 을 import 하지 않는 것이 pytest 를 가능하게
하는 전제다. 캐싱은 `loaders.py` 의 `_cached()` 래퍼가 **지연 적용**한다 — import
시점에 `st.cache_data` 를 붙이면 런타임 밖에서 경고가 쏟아진다.

`src/views/` 를 절대 `src/pages/` 로 바꾸지 말 것 — Streamlit 자동 멀티페이지와 충돌한다.

### 시간축 3구간 설계 (이 프로젝트의 핵심 구조)

ARS(유입) 데이터와 판매 데이터의 기간이 **부분적으로만 겹친다.** 구간을 명시적으로
분리하고 각 구간에서 할 수 있는 분석만 한다. `settings.PERIODS` 에 정의.

| 구간 | 기간 | 데이터 | 분석 | UI 배지 |
|---|---|---|---|---|
| A | 2024-12 (31일) | ARS + 판매 | 진짜 날짜 조인 — 상관·래그 실측 | `실측 조인` |
| B | 2023~2024 (2년) | 판매만 | CAI(객장 활동 지수) + 요일 브릿지 | `요일 브릿지` |
| C | 2026 상반기 (181일) | ARS만 | 최신 유입 트렌드 (판매 차트는 안내 처리) | `최신 유입` |

**모든 차트에 구간 배지를 단다.** 어느 근거로 그린 값인지 화면에 항상 드러내는 것이
이 프로젝트의 신뢰 장치다. `components.period_badge()` / `section_header(badge=...)`.

`loaders._load_ars_impl()` 은 `data/raw/ars_*.csv` 를 **glob 으로 concat** 한다.
ARS 파일을 추가로 넣으면 코드 수정 없이 구간 A 가 자동 확장된다.

#### ⚠ 구간 A 는 파일 하나에 매달려 있다

`data/raw/ars_20241201_20241231.csv` 가 **구간 A 의 유일한 근거**다. ARS 전처리본
(`ars_merged.csv`)은 2026-01~06 만 담고 있고, 판매 데이터는 2023~2024 라서
**이 파일이 없으면 ARS 와 판매가 겹치는 날이 0일**이 된다. 그러면 r=0.80 배너,
특산품 D+1 래그, VIP 상관 — 발표의 근거 전부가 사라진다.

`data/raw/` 는 gitignore 되므로 이 파일의 유일한 커밋본은 `data/sample/ars.parquet`
안에 있다. 사라졌으면 이렇게 되살린다:

```powershell
& ".\.venv\Scripts\python.exe" scripts\restore_ars_2024.py
Remove-Item data\processed\*.parquet -ErrorAction SilentlyContinue
```

`make_sample.py` 는 축약본을 덮어쓰기 전에 `guard_ars()` 로 구간 A 포함 여부를
검사한다 — 2024-12 이 빠진 ARS 로 축약본을 덮어쓰면 복원 원천까지 잃기 때문이다.

### 데이터셋 5종

| 키 | 원본 | 기간 | UI |
|---|---|---|---|
| `ars` | `ars_20241201_20241231.csv` + `ars_merged.csv` | 2024-12, 2026 상반기 | 전 탭 |
| `sales` | `casino_fnb / roomservice / local_goods _merged.csv` | 2023-01 ~ 2024-12 | 전 탭 |
| `golf` | `golf_visitors_merged.csv` | 2021-03 ~ 2025-12 | **미노출** |
| `demo` | Open API | 기간 집계 | 탭1 |
| `merchants` | Open API | — | 탭4 |

`golf` 는 로더·스키마·테스트까지만 편입돼 있고 화면에 쓰지 않는다. 판매 3종과
**단위(이용인원 vs 판매수량)와 기간이 달라 채널로 합치면 CAI·CSM 이 오염된다** —
`test_golf_is_separate_from_sales` 가 이 경계를 고정한다. 화면에 올릴 때는 별도
탭이나 보조 지표로 붙이고, `S.CHANNELS` 에 넣지 말 것.

### 6계층 데이터 폴백

로더는 `(DataFrame, source)` 를 반환하고 **예외를 밖으로 던지지 않는다.**

```
L1 data/processed/*.parquet  → PROCESSED
L2 data/raw/*.csv            → RAW      + parquet 캐시 기록
L3 data/sample/*.csv         → SAMPLE
L4 Open API (data.go.kr)     → API      + 디스크 캐시
L5 data/mock/*.csv           → MOCK
L6 src/data/fallback.py      → EMBEDDED  파일·네트워크 접근 0 → 절대 실패 없음
```

`fallback.py` 는 난수를 쓰지 않고 실측 통계(요일지수 토 1.45, 특산품 9월 1.48,
D+1 시차)를 모사한다 — 폴백 상태에서도 차트가 의미 있게 보여야 데모가 성립한다.
`hash()` 대신 md5 기반 `_stable_num()` 을 쓰는 이유는 PYTHONHASHSEED 무작위화다.

`api_client.py` 는 키 부재·인증오류·타임아웃·스키마 변경·JSON 파싱 실패를 전부
흡수하고 디스크 캐시 → None 순으로 폴백한다. 키는 `st.secrets` → 환경변수 순 조회.

### 스코어링 체인

`analysis/scoring.py` 의 `nrm` / `wsum` / `grade` 위에 전부 쌓인다.

- **CII** (`inflow.compute_cii`) — ARS 기반 유입 강도 0~100.
  ⚠ `winners`(총 당첨자)는 쓰지 않는다. 일일 정원 캡(최대 2,999)에 묶여 CV 0.178 로
  거의 상수다. 변동 신호는 `recv_total`(0.372)과 `tickets`(0.302).
  ⚠ **2026년 ARS 전처리본에는 `recv_total` 이 없다.** `has_pressure_signal()` 로
  유효값 3개 이상인지 보고, 없으면 `INFLOW_WEIGHTS_NO_PRESSURE` 로 demand/convert
  2축 폴백한다. 결과에 `pressure_signal` 컬럼이 붙어 어느 경로였는지 드러난다.
  **없는 컬럼을 0 으로 채워 3축을 강행하지 말 것** — "접수자 0명"이라는 거짓이
  지수의 35% 를 차지한다.
- **CAI** (`inflow.compute_cai`) — ARS 없는 구간의 유입 프록시. 카지노 식음 소비량 +
  거래 행 수 + ARS 요일 브릿지. `cai_validity()` 로 대체 타당성을 수치로 노출한다.
- **VRB / VTS** (`vip.py`) — VIP 모수와 타겟 지수. VTS 상위 10일 = 캠페인 집행 캘린더.
  `headroom`(유입 대비 미달분)이 핵심이다 — 유입만 보면 이미 잘 파는 날을 또 공략한다.
  ⚠ **base(VRB) 축은 유입 축의 복사본이라 자동으로 빠진다.** 성별·연령이 기간
  집계라 `vip_ratio` 가 상수 → `VRB = tickets × 상수`(실측 r=0.9988). 두 축을 다
  쓰면 유입에 0.60 을 준 셈이 되어 corr(inflow, vts)=0.981 로 지수가 유입의 재탕이
  된다. `base_is_redundant()` 가 상관으로 판정하고 `drop_base_weight()` 가 그 몫을
  **headroom 으로 이관**한다(균등 재정규화하면 유입 실효 비중이 0.35→0.47 로 되레
  커진다). 결과에 `base_signal` 컬럼이 붙어 어느 경로였는지 드러난다 —
  CII 의 `pressure_signal` 과 같은 규약이다. 일자별 성별·연령이 들어오면 자동 복구.
  `vts_vs_inflow()` 가 유입 대비 차이(상관·겹침·신규일)를 돌려주고 UI 가 그대로
  띄운다. **겹침이 N/N 이면 "유입 지수만 보셔도 된다"고 화면이 말하게 둔다.**
- **CSM** (`crosssell.py`) — 상품별 교차판매 스코어. `lift` = 고유입일 판매 비중 /
  저유입일 판매 비중. 표본 부족 시 `lift=NaN` 으로 두고 1.0 으로 채우지 않는다.
  ⚠ **표본 요건과 스무딩 상수는 절대값이 아니라 창 길이·채널 스케일에 비례한다**
  (`sample_gate`). 판매일수 `max(5, 창×20%)`, 수량 `max(10, 채널 판매량 중앙값)`,
  `alpha = 중앙값 규모 상품 1개분의 비중`. 절대값('5일·10개', 'alpha=1개분')만 두면
  731일 창에서 1,688종 중 1,534종이 통과해 lift 가 24,027 까지 튀고 CSM 상위를
  잡음이 점령한다(실측 → 도입 후 최대 4.5). 수량 요건이 채널별인 이유는 채널 간
  판매량이 10배 이상 다르기 때문 — 공통값을 걸면 작은 채널이 통째로 탈락한다.
  게이트 정보는 결과 프레임의 `attrs["gate"]` 로 화면까지 가서 "몇 종을 왜 뺐는지"
  캡션으로 뜬다. 번들 후보도 같은 이유로 비례 요건 + VIP 티어 제한을 쓴다.
  ⚠ **표본 미달 상품의 탄력성은 `nrm` 기본값 0.5 가 아니라 실측 분포의 중앙값으로
  채운다**(`CSM_ELASTICITY_FILL_QUANTILE`). 0.5 는 중립이 아니다 — 실측 중앙값이
  0.240(731일 창)이라 측정된 상품의 82%보다 높고, "표본 부족"이 점수 보너스가 된다
  (실측: lift 0.71 인 상품 CSM 14.6 vs 미측정 상품 48.1). **축을 빼고 재정규화하는
  안(pressure/base 규약)은 여기선 반대로 뒤집힌다** — 물량·마진이 최상인 미측정
  상품이 100점 1위가 된다(시뮬레이션 +22.5). 축이 둘만 남으면 만점이 쉬워지기
  때문이다. `lift_measured` 컬럼이 붙고 탭2·탭4 표의 `근거` 열로 노출된다.
- **래그** (`lag.py`) — 채널·상품 단위 유입(D) vs 판매(D+k) 상관. 탭3 의 근거.
- **편상관** (`stats.confound_table`) — 요일 더미 6개로 양변을 회귀한 잔차끼리의 상관.
  원 상관 0.801 의 54% 는 주말 효과라서 **통제하면 채널 순서가 뒤집힌다**
  (카지노 0.370 < 룸서비스 0.558). 탭1 배너·통제 절, 탭3 캡션, `prescriptions()` 의
  D+0 집행 순서가 전부 이 표를 근거로 삼는다 — 화면이 원 상관 서사를 말하면 README 와
  정면으로 어긋나므로 새 UI 에서 상관을 인용할 때는 편상관을 같이 띄운다.
  재표집(부트스트랩 CI·순열 p)은 `_rowwise_pearson` 으로 행렬 연산한다. 행마다
  `pearson()` 을 부르면 3채널에 5.5초가 걸려 매 리런마다 치를 수 없다(실측 → 0.07초).

`nrm(robust=True)` 는 5~95 분위 클리핑이다. **매월 1일 카지노 식음 판매량이 1.45배로
튀기 때문에**(2년 평균. 집계 아티팩트 의심) 단순 min-max 를 쓰면 그 하루가 스케일을
독점한다.

### 시각화 규칙 (`src/ui/theme.py`, `charts.py`)

팔레트는 색약 안전성 검증기를 실제로 돌려 통과한 값이다. 지킬 것:

- **이중 축 금지.** 건수와 지수는 `charts.inflow_stack()` 처럼 x축 공유 2단으로.
- **도넛·트리맵·산점도는 앞 3색까지만.** 전체쌍 검증을 통과한 범위가 3색이다.
  트리맵은 색을 카테고리가 아니라 판매량 크기(시퀀셜)로 쓴다.
- **시퀀셜 = `SEQ_BLUE` 단일 청색 램프.** 무지개 금지.
- **발산 팔레트(`DIVERGING_SCALE`)는 래그 히트맵에만.** D+2 상관이 −0.35 라서 부호가
  의미를 갖는 유일한 차트다.
- **색상은 엔티티 고정.** `color_discrete_map` 을 항상 명시 전달한다
  (`CHANNEL_COLOR_BY_LABEL`, `TIER_COLOR_BY_LABEL`, `GENDER_COLOR`). plotly 기본
  순환에 맡기면 필터로 계열이 줄었을 때 생존 계열의 색이 바뀐다.
- 모든 차트에 `components.table_view()` 표 뷰를 붙인다. 텍스트는 잉크색만 쓴다.
- Streamlit 1.60 기준 `width="stretch"` 를 쓴다 (`use_container_width` 아님).

## 실데이터 제약 (계획을 바꾼 것들)

원본 CSV 를 직접 분석해 확인한 사실이다. 코드 곳곳의 방어 로직이 여기서 나왔다.

- **ARS 파일마다 컬럼 구성이 다르다.** 2024-12 원본에는 9개 컬럼이 다 있지만
  2026년 전처리본은 `date/winners_total/ticket_purchases/purchase_rate` 4개뿐이다
  (컬럼명도 한글이 아니라 영문). 2026-05 는 두 소스에 다 있는데 값이 정확히
  일치하므로, 컬럼 소실은 원본 문제가 아니라 **전처리 과정의 산물**이다.
  `normalize_ars()` 는 `S.ARS_OPTIONAL` 을 **0 이 아니라 NaN 으로** 채운다.
- **`add_date_parts()` 는 날짜 결측 행을 파생 컬럼 생성 *전에* 버려야 한다.**
  `assign` 안의 표현식은 전부 먼저 평가되므로 뒤에서 `dropna` 를 해도
  `isocalendar().week.astype("int64")` 가 NA 를 만나 예외를 던진다. 골프 원본의
  빈 패딩 행 47개가 이 경로를 처음 밟았다 (판매 3종에는 결측 날짜가 없다).
- **`hour` 컬럼이 없다** → 원 기획의 "심야 피크타임 분석"은 불가능. 요일 × 월
  히트맵으로 대체했다 (`crosssell.heatmap_dow_month`). `tests` 가 이를 고정한다.
- **영업장 비대칭** — 카지노 식음 5개(크리스탈라운지/써미타스/민트 바/팬지/
  크리스탈라운지2), 룸서비스·특산품은 각 1개. 영업장 필터는 `MULTI_VENUE_CHANNELS`
  에만 적용하고 다른 채널 행을 날리지 않는다 (`FilterState.slice_sales`).
- **단가 컬럼이 없다** → 고마진은 프록시. 상품명 2티어(프리미엄/선물번들) + 판매
  희소도. 단일 정규식이면 특산품 히트율이 31.4% 로 과대해져 티어를 나눴다.
  `is_discount` 는 수량 기준 2.3%/0.2% 로 희소해 보조 신호로만 쓴다.
- **무상 제공 상품이 전체 수량의 6% 안팎** — `(V)` 접두 30,560개 + **`무료` 포함
  414,912개**(`18)헛개차(무료)`, `24)사과(무료)` 등, 2023~2024 합산). `무료` 쪽이 13배 크다.
  VIP 정규식에 단독 `V` 를 넣으면 대량 오탐하므로 접두 형태로만 매칭한다.
  `(V)산삼배양근진` 처럼 프리미엄 패턴에 걸리는 컴프가 실제로 있어 **컴프는 티어를
  일반으로 강등 + 마진 프록시 0 + CSM 0** 으로 눕힌다. 마진만 0 으로 두면 물량·탄력성
  신호가 남아 CSM 75 로 상위를 점령한다(실측).
- **lift 는 가법 스무딩 + 표본 요건이 필수다.** 저유입일 판매가 0인 상품이 72종
  있어서 그냥 나누면 lift 가 2,700만까지 발산한다. 또 판매일수 2일짜리 상품도
  lift 20~60 이 나오는데 그건 탄력성이 아니라 잡음이므로 `LIFT_MIN_DAYS`(5) /
  `LIFT_MIN_QTY`(10) 미달은 `NaN` 으로 둔다 — 1.0 으로 채우면 없는 근거를 만든다.
  번들 추천은 더 보수적으로 `BUNDLE_MIN_DAYS`(8) 를 요구한다.
- **특산품 2024-12-17 하루가 누락** (2023~2024 730일 중 유일). 일별 최소 판매량이
  206 이므로 판매 0인 날이 아니라 데이터 누락이다. **0 으로 채우지 않고 내부 조인으로
  제외**한다 — 채우면 상관 0.380→0.421, 일요일 지수 1.629→1.683 으로 왜곡된다.
  카지노 식음·룸서비스는 2년 730일이 빠짐없이 있다(채널 고유 현상).
- **골프장 영업장 분포가 극단적으로 비대칭** — 그린피 1,735행 / 카트대여 949 /
  용품대여 18 / 드라이빙레인지 1. 영업장별 비교는 사실상 그린피 단독 분석이다.
  `GOLF_VENUE_PRIMARY` 로 기준 영업장을 명시해 뒀다.
- **컬럼명에 오타·불규칙 공백이 있다** (`ARS 담청자`, `당첨자 입장권 구매 건 수`).
  `schema.normalize_key()` 로 공백·괄호를 제거한 키로 조회한다.
- 원본 파일명 규칙이 일관되지 않아 (`(주)강원랜드_` 접두 유무) `ingest.py` 는
  **파일명이 아니라 헤더 시그니처**로 데이터셋을 판정한다.

## 회귀 기준값

`tests/test_stats.py` 는 원본 분석으로 얻은 실측값을 기대값으로 박아둔 회귀 테스트다.
전처리를 리팩터링하다 수치가 흔들리면 여기서 잡힌다. 이 파일의 상수를 바꿀 때는
**왜 바뀌는지 근거를 확인하고 주석에 남긴다.**

- 행 수 179,287 / 69,353 / 36,226 (2023~2024 2년) · **2024년분만 보면
  90,983 / 35,653 / 18,514 로 교체 전과 동일** ← 2023년이 덧붙은 것뿐임을 고정
- ARS 212행 (2024-12 31행 + 2026 상반기 181행) · `recv_total` 유효는 31일뿐
- 컴프 수량 445,472 (그중 `(V)` 접두만 30,560, `무료` 포함 414,912)
- 구간 A 상관: `recv_total ↔ casino_fnb` 0.801/0.859, `↔ roomservice` 0.740/0.827
- VTS: `corr(inflow, vrb)` 0.9988(= base 축을 빼는 근거) · base 제외 후
  `corr(inflow, vts)` 0.889(제외 전 0.981) · 상위 10일 중 유입 상위 밖 2일
- 요일 통제 편상관: casino 0.370(p 0.043) · room 0.558(p 0.002) · local 0.176(p 0.344)
  ← seed 고정(`S.CONFOUND_SEED`)이라 p 도 회귀값이다. 순서 역전이 무너지면 발표 서사가
  깨지므로 `test_roomservice_is_the_robust_channel` 이 이를 고정한다
- 래그 best_lag: `{casino_fnb: 0, roomservice: 0, local_goods: 1}` ← 특산품 D+1 이
  프로젝트의 핵심 발견이다
- 요일 인덱스 특산품 일 1.629 / 목 0.642 (구간 A 기준 — 데이터 교체 후에도 불변)
- 월 인덱스 특산품 9월 **1.43**, 룸서비스 1월 **1.21** ← 2023년이 더해져 2년 평균으로
  희석됐다(1.48/1.23 → 1.43/1.21). **피크 월은 그대로고 두 해 모두에서 재현**되므로
  근거는 오히려 강해졌다 (`test_seasonality_repeats_across_years`)
- 매월 1일 카지노 식음 스파이크 **1.45배** (2년 평균, 기존 1.62)
- `buy_rate == tickets / winners` 전 212행 일치 (원본 정합성)
- CSM 상위 20 위에 컴프 상품이 없을 것 · lift 는 유한하고 표본 미달은 NaN
- CSM 게이트: 31일 창 → 7일 / 카지노 129·특산품 40·룸서비스 16개 · 731일 창 →
  147일 / 983·356·87개 · 두 창 모두 lift 최댓값 < 10 (도입 전 88 / 24,027)
- 탄력성 채움값(실측 중앙값): 31일 창 0.354 · 731일 창 0.240 ← 둘 다 0.5 미만이라는
  것이 "0.5 는 중립이 아니다"의 근거. 미측정 상품이 1위가 되지 않을 것
- 골프 2,703행 (원본 2,750 − 빈 패딩 47) · 판매 채널에 섞이지 않을 것

**구간 A(2024-12) 기반 수치는 데이터 교체 후에도 전부 그대로다.** 흔들린 것은 판매
2년 확장의 영향을 받는 총량·연간 지표뿐이다. 이 구분이 회귀 테스트를 읽는 열쇠다.

실데이터(`data/raw`)가 없으면 해당 테스트는 skip 된다. 현재 76개 전부 통과.

## 환경 주의사항

- **설치된 pandas 는 3.0.5** (2.x 아님). Copy-on-Write 가 기본이고 문자열 dtype 이
  arrow 기반이다. 연쇄 대입 대신 `assign` / `loc` 을 쓰고, bool 컬럼은
  `.to_numpy(dtype=bool)` 로 고정해야 필터링이 예측 가능하다.
- 지도는 **pydeck** 이다. streamlit 이 이미 의존하므로 추가 패키지가 0개다.
  README 의 기술 스택에는 Folium/Streamlit-Folium 이 적혀 있는데 **구현과 불일치**다
  (folium 은 별도 설치가 필요해 Cloud 빌드 리스크를 늘려서 제외했다). README 를
  수정하거나 사용자에게 확인할 것.
- `requirements.txt` 는 현재 `>=` 다. Cloud 빌드 성공을 확인한 뒤 `pip freeze` 로
  `==` 핀 고정하는 순서를 의도한 것이다.
- 로그인·초기 설정 관문을 두지 않는다 (`st.login`/비밀번호 위젯 미사용, 모든 사이드바
  위젯에 유효한 기본값). 접속 즉시 전체 대시보드가 조작 가능해야 한다 — 심사 요구사항.
- `app.py` 는 탭 단위로 예외를 격리한다. 한 탭이 실패해도 나머지는 동작해야 한다.

## 배포 (Streamlit Community Cloud)

- 루트 `requirements.txt` 만 본다. `packages.txt` 는 불필요.
- 키는 파일이 아니라 앱 **Settings → Secrets** 에 넣는다. `secrets.toml` 은 gitignore.
- ⚠ `.gitignore` 가 `data/raw/` 를 제외하므로 **Cloud 에는 원본이 올라가지 않는다.**
  `data/sample/` (커밋 대상)에 축약본을 넣어야 배포판이 실데이터로 뜬다. ARS 2개는
  각 1.7KB 라 전량 커밋한다 — 가설 검증 배너의 r=0.80 이 실측값으로 떠야 한다.
- Linux 는 파일명 대소문자를 구분한다. 저장소에 한글 파일명이 남지 않게 한다
  (`data/incoming/` 은 gitignore).
