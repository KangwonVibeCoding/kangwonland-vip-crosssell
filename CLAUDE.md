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
| B | 2024 전체 | 판매만 | CAI(객장 활동 지수) + 요일 브릿지 | `요일 브릿지` |
| C | 2026-05 (31일) | ARS만 | 최신 유입 트렌드 (판매 차트는 안내 처리) | `최신 유입` |

**모든 차트에 구간 배지를 단다.** 어느 근거로 그린 값인지 화면에 항상 드러내는 것이
이 프로젝트의 신뢰 장치다. `components.period_badge()` / `section_header(badge=...)`.

`loaders._load_ars_impl()` 은 `data/raw/ars_*.csv` 를 **glob 으로 concat** 한다.
ARS 파일을 추가로 넣으면 코드 수정 없이 구간 A 가 자동 확장된다.

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
- **CAI** (`inflow.compute_cai`) — ARS 없는 구간의 유입 프록시. 카지노 식음 소비량 +
  거래 행 수 + ARS 요일 브릿지. `cai_validity()` 로 대체 타당성을 수치로 노출한다.
- **VRB / VTS** (`vip.py`) — VIP 모수와 타겟 지수. VTS 상위 10일 = 캠페인 집행 캘린더.
  `headroom`(유입 대비 미달분)이 핵심이다 — 유입만 보면 이미 잘 파는 날을 또 공략한다.
- **CSM** (`crosssell.py`) — 상품별 교차판매 스코어. `lift` = 고유입일 판매 비중 /
  저유입일 판매 비중. 표본 부족 시 `lift=NaN` 으로 두고 1.0 으로 채우지 않는다.
- **래그** (`lag.py`) — 채널·상품 단위 유입(D) vs 판매(D+k) 상관. 탭3 의 근거.

`nrm(robust=True)` 는 5~95 분위 클리핑이다. **매월 1일 카지노 식음 판매량이 1.62배로
튀기 때문에**(집계 아티팩트 의심) 단순 min-max 를 쓰면 그 하루가 스케일을 독점한다.

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

- **`hour` 컬럼이 없다** → 원 기획의 "심야 피크타임 분석"은 불가능. 요일 × 월
  히트맵으로 대체했다 (`crosssell.heatmap_dow_month`). `tests` 가 이를 고정한다.
- **영업장 비대칭** — 카지노 식음 5개(크리스탈라운지/써미타스/민트 바/팬지/
  크리스탈라운지2), 룸서비스·특산품은 각 1개. 영업장 필터는 `MULTI_VENUE_CHANNELS`
  에만 적용하고 다른 채널 행을 날리지 않는다 (`FilterState.slice_sales`).
- **단가 컬럼이 없다** → 고마진은 프록시. 상품명 2티어(프리미엄/선물번들) + 판매
  희소도. 단일 정규식이면 특산품 히트율이 31.4% 로 과대해져 티어를 나눴다.
  `is_discount` 는 수량 기준 2.3%/0.2% 로 희소해 보조 신호로만 쓴다.
- **무상 제공 상품이 전체 수량의 6.1%** — `(V)` 접두 11종 21,870개 + **`무료` 포함
  38종 210,372개**(`18)헛개차(무료)`, `24)사과(무료)` 등). `무료` 쪽이 10배 크다.
  VIP 정규식에 단독 `V` 를 넣으면 대량 오탐하므로 접두 형태로만 매칭한다.
  `(V)산삼배양근진` 처럼 프리미엄 패턴에 걸리는 컴프가 실제로 있어 **컴프는 티어를
  일반으로 강등 + 마진 프록시 0 + CSM 0** 으로 눕힌다. 마진만 0 으로 두면 물량·탄력성
  신호가 남아 CSM 75 로 상위를 점령한다(실측).
- **lift 는 가법 스무딩 + 표본 요건이 필수다.** 저유입일 판매가 0인 상품이 72종
  있어서 그냥 나누면 lift 가 2,700만까지 발산한다. 또 판매일수 2일짜리 상품도
  lift 20~60 이 나오는데 그건 탄력성이 아니라 잡음이므로 `LIFT_MIN_DAYS`(5) /
  `LIFT_MIN_QTY`(10) 미달은 `NaN` 으로 둔다 — 1.0 으로 채우면 없는 근거를 만든다.
  번들 추천은 더 보수적으로 `BUNDLE_MIN_DAYS`(8) 를 요구한다.
- **특산품 2024-12-17 하루가 누락** (2024년 유일). 일별 최소 판매량이 206 이므로
  판매 0인 날이 아니라 데이터 누락이다. **0 으로 채우지 않고 내부 조인으로 제외**한다
  — 채우면 상관 0.380→0.421, 일요일 지수 1.629→1.683 으로 왜곡된다.
- **컬럼명에 오타·불규칙 공백이 있다** (`ARS 담청자`, `당첨자 입장권 구매 건 수`).
  `schema.normalize_key()` 로 공백·괄호를 제거한 키로 조회한다.
- 원본 파일명 규칙이 일관되지 않아 (`(주)강원랜드_` 접두 유무) `ingest.py` 는
  **파일명이 아니라 헤더 시그니처**로 데이터셋을 판정한다.

## 회귀 기준값

`tests/test_stats.py` 는 원본 분석으로 얻은 실측값을 기대값으로 박아둔 회귀 테스트다.
전처리를 리팩터링하다 수치가 흔들리면 여기서 잡힌다. 이 파일의 상수를 바꿀 때는
**왜 바뀌는지 근거를 확인하고 주석에 남긴다.**

- 행 수 90,983 / 35,653 / 18,514 · ARS 62행
- 컴프 수량 232,242 (그중 `(V)` 접두만 21,870, `무료` 포함 210,372)
- 구간 A 상관: `recv_total ↔ casino_fnb` 0.801/0.859, `↔ roomservice` 0.740/0.827
- 래그 best_lag: `{casino_fnb: 0, roomservice: 0, local_goods: 1}` ← 특산품 D+1 이
  프로젝트의 핵심 발견이다
- 요일 인덱스 특산품 일 1.629 / 목 0.642 · 월 인덱스 특산품 9월 1.48, 룸서비스 1월 1.23
- `buy_rate == tickets / winners` 전 행 일치 (원본 정합성)
- CSM 상위 20 위에 컴프 상품이 없을 것 · lift 는 유한하고 표본 미달은 NaN

실데이터(`data/raw`)가 없으면 해당 테스트는 skip 된다. 현재 36개 전부 통과.

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
