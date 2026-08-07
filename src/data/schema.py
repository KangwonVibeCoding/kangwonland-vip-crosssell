"""스키마 표준화 — 한글 컬럼명 → snake_case, 인코딩 폴백, 파생 컬럼, 검증.

streamlit 을 import 하지 않는다 (pytest 에서 순수 호출 가능).
pandas 3.x 기준으로 작성했다 — Copy-on-Write 가 기본이므로 연쇄 대입 대신
assign / loc 을 쓴다.
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

import pandas as pd

from config import settings as S

# 공공데이터 CSV 는 CP949 가 흔하다. 넓은 것부터 좁은 것 순으로 시도한다.
ENCODINGS = ("utf-8-sig", "cp949", "euc-kr", "utf-8")

_NON_KEY = re.compile(r"[\s()（）\[\]{}·.,/\\\-_'\"]+")


def normalize_key(name: str) -> str:
    """컬럼명을 매핑 조회용 키로 정규화한다.

    '당첨자 입장권 구매 건 수' → '당첨자입장권구매건수'
    '영업장아이디(ID)'        → '영업장아이디id'

    원본의 불규칙 공백과 괄호 표기를 흡수하는 것이 목적이다.
    """
    return _NON_KEY.sub("", str(name)).strip().lower()


def read_csv_kr(path: str | Path, **kwargs) -> pd.DataFrame:
    """한국 공공데이터 CSV 를 인코딩 자동판별로 읽는다.

    바이트를 한 번만 읽고 디코딩을 시도하므로 파일 I/O 는 1회다.
    """
    path = Path(path)
    raw = path.read_bytes()
    last_err: Exception | None = None
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
        try:
            return pd.read_csv(io.StringIO(text), **kwargs)
        except Exception as exc:  # noqa: BLE001  파싱 실패도 다음 인코딩으로
            last_err = exc
            continue
    raise ValueError(f"CSV 를 읽을 수 없습니다: {path} ({last_err})")


def detect_encoding(path: str | Path) -> str | None:
    """파일이 어떤 인코딩으로 디코딩되는지 반환한다 (ingest 리포트용)."""
    raw = Path(path).read_bytes()
    for enc in ENCODINGS:
        try:
            raw.decode(enc)
        except UnicodeDecodeError:
            continue
        return enc
    return None


def rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """COLUMN_MAP 기반으로 컬럼을 표준명으로 바꾼다.

    미매칭 컬럼은 원본명을 그대로 유지하고 목록으로 반환한다 — 앱이 죽지 않고
    계속 동작하게 하기 위함이다.
    """
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    for col in df.columns:
        std = S.COLUMN_MAP.get(normalize_key(col))
        if std:
            mapping[col] = std
        else:
            unmatched.append(str(col))
    return df.rename(columns=mapping), unmatched


def _to_int(s: pd.Series) -> pd.Series:
    """숫자 문자열(콤마 포함) → int. 결측·음수는 0."""
    cleaned = s.astype("string").str.replace(",", "", regex=False).str.strip()
    num = pd.to_numeric(cleaned, errors="coerce").fillna(0)
    return num.clip(lower=0).astype("int64")


def _to_float(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """date 컬럼에서 파생 시간 컬럼을 만든다.

    dow 는 월=0 … 일=6 (pandas 규약). 화면 표기는 DOW_NAMES 로 변환한다.

    ⚠ 파싱 실패 행을 **파생 컬럼을 만들기 전에** 버린다. assign 안의 표현식은
    전부 먼저 평가되므로, 뒤에서 dropna 를 해도 `isocalendar().week` 가 NA 를
    int64 로 변환하려다 예외가 난다 (골프 원본의 빈 패딩 행 47개가 이 경로를
    처음 밟았다).
    """
    d = pd.to_datetime(df["date"], errors="coerce")
    keep = d.notna()
    if not keep.all():
        df = df.loc[keep]
        d = d.loc[keep]
    return df.assign(
        date=d,
        dow=d.dt.dayofweek,
        dow_name=d.dt.dayofweek.map(dict(enumerate(S.DOW_NAMES))),
        month=d.dt.month,
        year=d.dt.year,
        iso_week=d.dt.isocalendar().week.astype("int64"),
        is_month_first=(d.dt.day == 1),
    )


def normalize_ars(df: pd.DataFrame) -> pd.DataFrame:
    """ARS 예약현황 표준화 + 파생 지표.

    실측 특성:
      - winners(총 당첨자)는 일일 정원 캡(최대 2,999)이라 거의 상수 → 유입
        지표로는 recv_total / tickets 를 쓴다.
      - buy_rate 는 % 단위(52.85)로 들어오므로 0~1 로 환산한다.
        (검증: buy_rate == tickets / winners 가 2024-12 실데이터 31행 일치)

    ⚠ 파일마다 컬럼 구성이 다르다. 2024-12 원본에는 총접수자·모바일 계열이
    모두 있지만 2026년 전처리본(ars_merged.csv)에는 winners/tickets/buy_rate
    3개뿐이다. 두 파일을 concat 하면 recv_* 가 부분 결측이 되므로, **없는
    컬럼은 0 이 아니라 결측(NA)으로 만든다.** 0 으로 채우면 '접수자 0명'이라는
    거짓이 생겨 정규화 스케일과 상관계수가 통째로 망가진다.
    """
    df, _ = rename_columns(df)
    int_cols = ("recv_total", "recv_ars", "recv_mobile", "winners",
                "win_ars", "win_mobile", "tickets")
    out = df.assign(**{c: _to_int(df[c]) for c in int_cols if c in df.columns})

    # 없는 선택 컬럼은 float 결측으로 채워 둔다 — 이후 계산이 조건 분기 없이
    # NaN 전파로 자연스럽게 처리된다.
    for col in S.ARS_OPTIONAL:
        if col not in out.columns:
            out = out.assign(**{col: pd.Series(float("nan"), index=out.index,
                                               dtype="float64")})

    if "buy_rate" in out.columns:
        rate = _to_float(out["buy_rate"])
        # 원본이 % 표기(30~53)이므로 1 을 넘으면 100 으로 나눈다.
        rate = rate.where(rate <= 1.0, rate / 100.0)
        out = out.assign(buy_rate=rate)
    elif {"tickets", "winners"} <= set(out.columns):
        out = out.assign(buy_rate=out["tickets"] / out["winners"].replace(0, pd.NA))

    out = add_date_parts(out)

    winners = out["winners"].replace(0, pd.NA)
    recv_total = pd.to_numeric(out["recv_total"], errors="coerce")
    recv_mobile = pd.to_numeric(out["recv_mobile"], errors="coerce")
    return out.assign(
        recv_total=recv_total,
        # 예약 경쟁률 — 정원 대비 수요 과열도. 총접수자가 없는 구간은 NaN.
        compete_ratio=(recv_total / winners).astype("float64"),
        # 디지털 비중. 역수가 고연령 프록시 (실측 ARS 비중 16.5~20.7%)
        mobile_share=(recv_mobile / recv_total.replace(0, pd.NA)).astype("float64"),
        is_capped=out["winners"] >= S.CAPACITY_CAP,
    ).sort_values("date").reset_index(drop=True)


def normalize_golf(df: pd.DataFrame) -> pd.DataFrame:
    """골프장 이용객 현황 표준화 (2021-03 ~ 2025-12).

    원본 특성:
      - 영업일자가 통째로 빈 행이 47개 있다 (파일 끝 패딩) → 날짜 결측은 버린다
      - 영업장이 4종(그린피/카트대여/용품대여/드라이빙레인지)이라 날짜당 다중 행
      - 영업상태 '휴장영업' 은 문을 닫은 날이다. 행을 지우지 않고 플래그로 남겨
        호출자가 판단하게 한다 — 휴장일 분포 자체가 시즌 신호이기 때문이다.
    """
    df, _ = rename_columns(df)
    out = df.assign(visitors=_to_int(df["visitors"]))
    for col in ("season", "day_type", "dow_label", "golf_venue", "open_status"):
        if col in out.columns:
            out = out.assign(
                **{col: out[col].astype("string").fillna("").str.strip()}
            )
        else:
            out = out.assign(**{col: pd.Series([""] * len(out), dtype="string")})

    out = add_date_parts(out)     # date 결측 행(빈 패딩)이 여기서 제거된다
    return out.assign(
        is_closed=out["open_status"].isin(S.GOLF_CLOSED_STATUSES)
        .fillna(False).to_numpy(dtype=bool),
    ).sort_values(["date", "golf_venue"]).reset_index(drop=True)


def _stable_item_id(items: pd.Series, channel: str) -> pd.Series:
    """item_id 가 없는 채널(룸서비스)용 안정적 대체 ID 생성."""
    def h(v: object) -> str:
        digest = hashlib.md5(f"{channel}|{v}".encode()).hexdigest()[:10]
        return f"H{digest.upper()}"

    return items.map(h)


def normalize_sales(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    """판매 3종을 공통 long 스키마로 표준화한다.

    3개 데이터셋이 사실상 같은 스키마(영업일자/영업장명/상품명/판매수량)여서
    단일 테이블로 합칠 수 있다 — 이 프로젝트에서 가장 레버리지가 큰 결정이다.
    """
    df, _ = rename_columns(df)
    out = df.assign(qty=_to_int(df["qty"]))

    for col in ("venue", "item", "item_en", "venue_id"):
        if col in out.columns:
            out = out.assign(
                **{col: out[col].astype("string").fillna("").str.strip()}
            )
        else:
            out = out.assign(**{col: pd.Series([""] * len(out), dtype="string")})

    if "item_id" in df.columns:
        ids = out["item_id"].astype("string").fillna("").str.strip()
        ids = ids.where(ids != "", _stable_item_id(out["item"], channel))
    else:
        ids = _stable_item_id(out["item"], channel)
    out = out.assign(item_id=ids)

    if "is_discount" in df.columns:
        flag = out["is_discount"].astype("string").str.strip().str.upper()
        # pandas 3 에서 arrow 문자열 비교는 nullable bool 을 만든다. 필터링을
        # 예측 가능하게 하려고 numpy bool 로 고정한다.
        out = out.assign(is_discount=flag.eq("Y").fillna(False).to_numpy(dtype=bool))
    else:
        out = out.assign(is_discount=False)

    out = out.assign(channel=channel)
    out = add_date_parts(out)
    out = add_item_tiers(out)
    keep = ["date", "dow", "dow_name", "month", "year", "iso_week",
            "is_month_first", "channel", "venue_id", "venue", "item_id",
            "item", "item_en", "is_discount", "is_comp", "tier", "qty"]
    return out[keep].sort_values(["date", "channel"]).reset_index(drop=True)


def add_item_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """상품명 기반 2티어 분류 + 컴프 플래그.

    프리미엄(명품/한우/와인…)과 선물번들(세트/패키지)을 나누는 이유: 단일
    정규식이면 특산품 히트율이 31.4% 로 과대해져 큐레이션이 무의미해진다(실측).
    """
    item = df["item"].astype("string").fillna("")
    is_premium = item.str.contains(S.TIER_PREMIUM_PATTERN, case=False, regex=True, na=False)
    is_bundle = item.str.contains(S.TIER_BUNDLE_PATTERN, case=False, regex=True, na=False)
    # ⚠ '(V)에비앙' 등은 컴프/바우처 추정. VIP 로 오탐하면 안 된다.
    is_comp = item.str.contains(S.COMP_PATTERN, regex=True, na=False)

    tier = pd.Series(S.TIER_STANDARD, index=df.index, dtype="object")
    tier = tier.mask(is_bundle, S.TIER_BUNDLE)
    tier = tier.mask(is_premium, S.TIER_PREMIUM)   # premium 이 bundle 을 덮는다

    # 컴프/바우처는 무상 제공이므로 마진이 0 이다. '(V)발효홍삼', '(V)산삼배양근진'
    # 처럼 프리미엄 패턴에 걸리는 컴프 상품이 실제로 존재하는데(실측 3종), 이를
    # 프리미엄 매출로 집계하면 고마진 지표가 부풀려진다 → standard 로 강등한다.
    tier = tier.mask(is_comp, S.TIER_STANDARD)

    return df.assign(
        tier=tier.astype("string"),
        is_comp=is_comp.fillna(False).to_numpy(dtype=bool),
    )


def normalize_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """고객성별연령분석현황 표준화. 연령대 표기 변형을 흡수한다."""
    df, _ = rename_columns(df)
    out = df.assign(visitors=_to_int(df["visitors"]))
    # 실 API 는 BSN_DT 로 일자를 준다. 다른 데이터셋의 `date` 가 전부 datetime64 라
    # 문자열로 두면 축약본 생성·parquet dtype 이 데이터셋마다 갈린다.
    # `add_date_parts()` 는 쓰지 않는다 — 이 프레임은 일별 분석에 쓰이지 않고
    # (제공 기간이 판매와 겹치지 않는다) 요일·월 파생이 필요 없다.
    if "date" in out.columns:
        out = out.assign(date=pd.to_datetime(out["date"], errors="coerce"))
    for col in ("gender", "age_band"):
        out = out.assign(
            **{col: out[col].astype("string").fillna("").str.strip()}
        )
    # '60대 이상', '60세이상', '60+' 등을 '60대이상' 으로 통일
    band = out["age_band"].str.replace(r"\s+", "", regex=True)
    band = band.str.replace("세", "대", regex=False)
    band = band.where(~band.str.contains("이상|60\\+", regex=True, na=False), "60대이상")
    # 실 API 는 60대·70대를 나눠 준다. S.AGE_BANDS_ALL 의 최상단이 '60대이상' 이라
    # 접지 않으면 `vip_ratio` 의 isin 필터가 두 밴드를 통째로 버린다 — VIP 비중이
    # 조용히 과소 집계된다. 앱 전체가 60대 이상을 한 덩어리로 다루므로 여기서 접는다.
    band = band.where(~band.isin(["60대", "70대", "80대", "90대"]), "60대이상")
    return out.assign(age_band=band)


def normalize_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """하이원포인트 가맹점 표준화.

    ⚠ 원본 API(`getStoreInfo`)는 **위경도도 업종도 주지 않는다** — 상호명·주소·
    전화번호뿐이다. 둘 다 `scripts/join_merchant_geo.py` 가 소상공인시장진흥공단
    상가(상권)정보와 조인해 채운 뒤 `merchants_geocoded.csv` 로 떨어진다.
    아직 조인하지 않은 원본 응답이 들어오면 예외를 던지는 대신 **빈 프레임**을
    돌려준다 — 로더가 다음 후보 파일로 넘어가야 하고, 예외로 폴백 사슬을 끊으면
    안 되기 때문이다.

    ⚠ **좌표가 없는 행을 버리지 않는다.** 상호·주소 매칭이라 100% 가 되지 않고
    (실측 91.6%), 나머지를 조용히 지우면 "가맹점 1,540곳"이 전수인 것처럼 보인다.
    행은 남기고 `has_coord` 로 표시해 호출자가 건수를 화면에 드러낼 수 있게 한다.
    지도 레이어는 `has_coord` 로 거른다.
    """
    df, _ = rename_columns(df)
    if "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame(columns=["merchant", "lat", "lon", "category",
                                     "has_coord"])
    out = df.assign(lat=_to_float(df["lat"]), lon=_to_float(df["lon"]))
    for col in ("merchant", "category"):
        if col in out.columns:
            out = out.assign(
                **{col: out[col].astype("string").fillna("").str.strip()}
            )
        else:
            out = out.assign(**{col: pd.Series([""] * len(out), dtype="string")})
    valid = out["lat"].notna() & out["lon"].notna() & out["lat"].ne(0) & out["lon"].ne(0)
    # 한반도 대략 범위 밖 좌표는 오염 데이터로 본다 (오매칭 방어)
    valid &= out["lat"].between(33, 39) & out["lon"].between(124, 132)
    # 범위 밖 좌표는 값 자체를 버린다 — 남겨두면 지도에 엉뚱한 점이 찍힌다
    return out.assign(
        has_coord=valid.to_numpy(dtype=bool),
        lat=out["lat"].where(valid),
        lon=out["lon"].where(valid),
    ).reset_index(drop=True)


def validate(df: pd.DataFrame, required: tuple[str, ...], name: str) -> list[str]:
    """필수 컬럼 존재 여부 검사. 누락 목록을 반환한다 (예외를 던지지 않는다)."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"{name}: 필수 컬럼 누락 {missing}"]
    return []
