"""가맹점 × 상가(상권)정보 조인 → 좌표 + 업종 (빌드 타임 1회 실행).

하이원포인트 가맹점 API(`getStoreInfo`)는 **위경도도 업종도 주지 않는다** —
상호명과 주소만 온다. 지도(탭4)와 교차판매 적합도 점수는 좌표가 있어야 성립하므로
소상공인시장진흥공단 상가(상권)정보와 조인해 채운다.

    data/raw/merchants_api.csv  ×  data/incoming/<상가정보>.csv|zip
                                →  data/raw/merchants_geocoded.csv

왜 상가정보인가:
  - **공공데이터라 이용조건에 업종 제한이 없다.** 상용 지도 API 는 약관에서
    사행성 서비스(도박/카지노)를 금지하는 경우가 있어 이 프로젝트에는 못 쓴다.
  - 정적 CSV 라 런타임 네트워크 호출이 0 이다.
  - 업종이 **표준산업분류 기반 공식 체계**(대분류 10 / 중분류 75 / 소분류 247)라,
    교차판매 적합도 가중치를 임의값이 아니라 공개된 분류에 매핑할 수 있다.

⚠ 매칭률은 100% 가 되지 않는다. 하이원포인트 쪽에는 사업자등록번호가 있지만
상가정보에는 없어서(개인정보) 상호명·주소로만 맞출 수 있다. 폐업했거나 상가정보
수집 대상이 아닌 곳은 빠진다. **미매칭은 조용히 버리지 않고 행을 남긴 채
좌표만 결측으로 두고, 리포트와 UI 에 건수를 드러낸다.**

컬럼명은 하드코딩하지 않는다. 후보 집합으로 탐지하고, 못 찾으면 실제 헤더를
그대로 출력한다 — 파일 판(版)이 바뀌어도 원인이 즉시 보이게.

사용법:
    python scripts/join_merchant_geo.py
"""

from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings as S                       # noqa: E402
from src.data import schema                            # noqa: E402

SRC_MERCHANTS = S.RAW_DIR / "merchants_api.csv"
DEST = S.RAW_DIR / "merchants_geocoded.csv"

# 상가정보에서 찾을 컬럼 — normalize_key 결과 기준 후보 집합.
# 판마다 표기가 조금씩 달라서(경도/경도(x), 상권업종대분류명/대분류명) 넓게 잡는다.
WANT = {
    "name":     ("상호명", "사업장명", "상호"),
    "branch":   ("지점명",),
    "lat":      ("위도", "위도y", "y"),
    "lon":      ("경도", "경도x", "x"),
    "category": ("상권업종대분류명", "대분류명", "업종대분류명", "표준산업분류명"),
    "cat_mid":  ("상권업종중분류명", "중분류명"),
    "sigungu":  ("시군구명", "시군구"),
    "jibun":    ("지번주소", "지번주소명"),
    "road":     ("도로명주소", "도로명주소명"),
}

# 가맹점 주소의 시도 표기가 3종 혼재한다 (강원도 824 / 강원특별자치도 801 / 강원 56)
SIDO_PATTERN = re.compile(r"^(강원특별자치도|강원도|강원)\s*")
SIGUNGU_PATTERN = re.compile(r"([가-힣]+(?:시|군))")
EMD_PATTERN = re.compile(r"([가-힣]+[읍면동])")
# 상호명 매칭용 — 공백·괄호·기호를 지우고 비교한다 ('#감동' / '# 감동' / '(주)감동')
NOISE_PATTERN = re.compile(r"[\s()（）\[\]{}·.,/\\\-_'\"&#·]+|^\(?주\)?|주식회사")


def find_sources() -> list[Path]:
    """data/incoming 에서 상가정보 파일을 **전부** 찾는다 (csv 또는 zip).

    시도별·분기별로 나눠 받았을 수도 있으므로 하나만 고르지 않는다. 하나만 읽고
    넘어가면 엉뚱한 지역을 집어와도 알아채지 못한다.

    ⚠ **파일명 역순(최신 우선)으로 돌려준다.** 상가정보는 분기마다 새 판이 나오고
    파일명에 연월이 들어간다. 뒤에서 중복을 `keep="first"` 로 제거하므로, 최신 판이
    앞에 와야 최신 좌표가 살아남는다. 과거 판은 최신 판에 없는 가게(최근 폐업·누락)를
    메우는 보충재로만 쓰인다 — 매칭률은 올리고 좌표 최신성은 지키는 절충이다.
    """
    candidates = sorted(S.INCOMING_DIR.glob("*.csv"), reverse=True) + \
        sorted(S.INCOMING_DIR.glob("*.zip"), reverse=True)
    named = [p for p in candidates
             if any(k in p.name for k in ("상가", "상권", "소상공인", "sangga", "store"))]
    if named:
        return named
    # 이름 규칙이 안 맞을 수 있으니, 하나뿐이면 그걸 쓴다
    return candidates if len(candidates) == 1 else []


def _decode_csv(raw: bytes, label: str) -> pd.DataFrame:
    for enc in schema.ENCODINGS:
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)), dtype="string",
                               low_memory=False)
        except Exception:  # noqa: BLE001
            continue
    raise ValueError(f"디코딩 실패: {label}")


def read_any(path: Path) -> pd.DataFrame:
    """csv, 또는 zip 안의 **모든** csv 를 읽어 합친다.

    상가정보 zip 은 시도별 CSV 를 여러 개 담고 있는 경우가 있다. 첫 파일만 읽으면
    강원이 아닌 지역을 조용히 쓰게 되므로 전부 읽고 뒤에서 시군으로 거른다.
    """
    if path.suffix.lower() != ".zip":
        return schema.read_csv_kr(path, dtype="string", low_memory=False)
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"zip 안에 csv 가 없습니다: {z.namelist()[:5]}")
        for n in names:
            frames.append(_decode_csv(z.read(n), n))
            print(f"     · {n}  ({len(frames[-1]):,}행)")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def resolve_columns(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """후보 집합으로 실제 컬럼명을 찾는다. 못 찾은 역할을 함께 반환한다."""
    by_key = {schema.normalize_key(c): c for c in df.columns}
    found: dict[str, str] = {}
    for role, names in WANT.items():
        for n in names:
            col = by_key.get(schema.normalize_key(n))
            if col is not None:
                found[role] = col
                break
    missing = [r for r in ("name", "lat", "lon") if r not in found]
    return found, missing


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.split(",").str[0].map(
        lambda v: NOISE_PATTERN.sub("", str(v)).lower()
    )


def sigungu(s: pd.Series) -> pd.Series:
    stripped = s.astype("string").fillna("").str.replace(SIDO_PATTERN, "", regex=True)
    return stripped.str.extract(SIGUNGU_PATTERN, expand=False).fillna("")


def emd(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.extract(EMD_PATTERN, expand=False).fillna("")


def norm_addr(s: pd.Series) -> pd.Series:
    """주소 비교용 정규화.

    가맹점 주소는 '삼척시 도계읍 **전두리** 도계로 252' 처럼 법정리와 도로명이
    함께 들어있는 반면, 상가정보 도로명주소는 '삼척시 도계읍 도계로 252' 다.
    법정리 토큰을 떼야 같은 건물이 같은 키가 된다. 도로명은 로/길/대로로 끝나므로
    '…리' 토큰만 지우는 것은 안전하다.
    """
    a = s.astype("string").fillna("")
    a = a.str.replace(SIDO_PATTERN, "", regex=True)
    a = a.str.replace(r"\([^)]*\)", " ", regex=True)          # (동강휴게소)
    a = a.str.replace(r"\s+\d+층.*$", "", regex=True)          # 2층 / 1층 3호
    a = a.str.replace(r"\s+[가-힣]+리(?=\s)", " ", regex=True)  # 전두리
    return a.str.replace(r"\s+", "", regex=True).str.strip()


def main() -> int:
    if not SRC_MERCHANTS.exists():
        print(f"✗ {SRC_MERCHANTS} 가 없습니다. 먼저 실행하세요:")
        print("    python scripts/fetch_api_data.py merchants")
        return 1

    sources = find_sources()
    if not sources:
        print("✗ data/incoming 에서 상가(상권)정보 파일을 찾지 못했습니다.")
        print("  공공데이터포털 → 소상공인시장진흥공단_상가(상권)정보 → 강원 파일을")
        print("  data/incoming/ 에 넣고 다시 실행하세요.")
        print(f"  (현재 폴더 내용: {[p.name for p in S.INCOMING_DIR.iterdir()] or '비어 있음'})")
        return 1

    print("=" * 78)
    print(f"가맹점 × 상가정보 조인   원본 {len(sources)}개")
    print("=" * 78)

    frames: list[pd.DataFrame] = []
    for p in sources:
        print(f"\n▶ {p.name}  ({p.stat().st_size:,} bytes)")
        frames.append(read_any(p))
    shops = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    cols, missing = resolve_columns(shops)
    print(f"\n상가정보: {len(shops):,}행 · 컬럼 {len(shops.columns)}개")
    for role, col in cols.items():
        print(f"   {role:<9} ← {col}")
    if missing:
        print(f"\n✗ 필수 역할을 못 찾았습니다: {missing}")
        print(f"  실제 헤더: {list(shops.columns)}")
        return 1

    # 가맹점 쪽
    raw = schema.read_csv_kr(SRC_MERCHANTS, dtype="string")
    mer, _ = schema.rename_columns(raw)
    print(f"가맹점: {len(mer):,}행")

    mer = mer.assign(
        _name=norm_name(mer["merchant"]),
        _gu=sigungu(mer["address"]),
        _emd=emd(mer["address"]),
        _road=norm_addr(mer["address"]),
    )

    # 상가정보를 가맹점이 실제로 있는 시군으로 좁힌다 (전국 파일이어도 안전)
    shop_addr = shops[cols.get("road", cols.get("jibun", cols["name"]))]
    shops = shops.assign(
        _name=norm_name(shops[cols["name"]]),
        _gu=(shops[cols["sigungu"]].astype("string").fillna("")
             if "sigungu" in cols else sigungu(shop_addr)),
        _emd=emd(shop_addr),
        _lat=pd.to_numeric(shops[cols["lat"]], errors="coerce"),
        _lon=pd.to_numeric(shops[cols["lon"]], errors="coerce"),
        _cat=(shops[cols["category"]].astype("string").fillna("")
              if "category" in cols else ""),
        # 상호명+지점명 결합 키 — 가맹점 쪽 '감탄도계점' 같은 표기를 잡는다
        _name2=(norm_name(shops[cols["name"]].astype("string").fillna("")
                          + shops[cols["branch"]].astype("string").fillna(""))
                if "branch" in cols else norm_name(shops[cols["name"]])),
        _road=(norm_addr(shops[cols["road"]]) if "road" in cols else ""),
        _jibun=(norm_addr(shops[cols["jibun"]]) if "jibun" in cols else ""),
    )
    target_gu = set(mer["_gu"]) - {""}
    if target_gu:
        narrowed = shops.loc[shops["_gu"].isin(target_gu)]
        print(f"   시군 필터({', '.join(sorted(target_gu))}): "
              f"{len(shops):,} → {len(narrowed):,}행")
        shops = narrowed
    # 좌표 없는 행만 여기서 버린다. 중복 제거는 **각 매칭 단계가 자기 키로**
    # 수행한다 — 여기서 (상호,읍면동) 로 미리 접으면 주소 단계에서 살아남았을
    # 다른 건물의 행까지 사라진다. find_sources() 가 최신 파일을 앞에 놓으므로
    # 각 단계의 keep="first" 가 곧 '최신 좌표 채택'이 된다.
    before = len(shops)
    shops = shops.dropna(subset=["_lat", "_lon"])
    print(f"   좌표 결측 제거: {before:,} → {len(shops):,}행")

    # 단계별 매칭 — 좁고 확실한 조건부터. 뒤 단계는 앞에서 못 찾은 행만 본다.
    #
    # (left_key, right_key) 를 따로 두는 이유:
    #   · 상가정보는 상호명과 지점명이 분리돼 있다. 가맹점 쪽 '감탄도계점' 은
    #     '감탄'+'도계점' 을 이어 붙인 `_name2` 와만 맞는다.
    #   · 마지막 주소 단계는 **좌표만** 가져온다. 같은 건물에 여러 가게가 있어
    #     업종까지 가져오면 남의 업종을 뒤집어씌우게 된다. 건물 좌표는 어차피
    #     같으므로 좌표는 안전하고, 업종은 비워 UI 에서 '미상'으로 표시한다.
    stages = (
        ("상호+읍면동", ["_name", "_emd"], ["_name", "_emd"], True),
        ("상호지점+읍면동", ["_name", "_emd"], ["_name2", "_emd"], True),
        ("상호+시군", ["_name", "_gu"], ["_name", "_gu"], True),
        ("상호지점+시군", ["_name", "_gu"], ["_name2", "_gu"], True),
        ("도로명주소", ["_road"], ["_road"], False),
        ("지번주소", ["_road"], ["_jibun"], False),
    )
    out = mer.assign(lat=pd.NA, lon=pd.NA, category="", geo_match="none")
    for label, lkeys, rkeys, take_cat in stages:
        todo = out["geo_match"] == "none"
        if not todo.any():
            break
        if any(k not in shops.columns for k in rkeys):
            continue
        right = shops.dropna(subset=rkeys).loc[shops[rkeys[0]].astype(str).str.len() > 0]
        right = right.drop_duplicates(subset=rkeys, keep="first")[
            rkeys + ["_lat", "_lon", "_cat"]]
        right.columns = lkeys + ["_lat", "_lon", "_cat"]
        merged = out.loc[todo, lkeys].merge(right, on=lkeys, how="left")
        hit = merged["_lat"].notna().to_numpy()
        if not hit.any():
            print(f"   {label:<14} +0건")
            continue
        idx = out.index[todo][hit]
        out.loc[idx, "lat"] = merged.loc[hit, "_lat"].to_numpy()
        out.loc[idx, "lon"] = merged.loc[hit, "_lon"].to_numpy()
        if take_cat:
            out.loc[idx, "category"] = merged.loc[hit, "_cat"].to_numpy()
        out.loc[idx, "geo_match"] = label
        print(f"   {label:<14} +{int(hit.sum()):,}건")

    total = len(out)
    matched = int((out["geo_match"] != "none").sum())
    print(f"\n좌표 확보율: {matched:,}/{total:,} ({matched / total:.1%})")
    print(f"업종 확보율: {(out['category'].astype(str).str.len() > 0).mean():.1%}")

    coords = out.dropna(subset=["lat", "lon"])
    if not coords.empty:
        lat = pd.to_numeric(coords["lat"], errors="coerce")
        lon = pd.to_numeric(coords["lon"], errors="coerce")
        # 영월군 서부(한반도면·남면)가 경도 128.3 대라 하한을 128.4 로 잡으면
        # 정상 좌표가 오매칭으로 잡힌다 (실측 3건). 강원 남부 실제 범위로 넓힌다.
        bad = coords.loc[~(lat.between(37.0, 37.7) & lon.between(128.2, 129.4))]
        print(f"강원 남부 범위 밖 좌표: {len(bad)}건"
              + (" ← 오매칭 의심" if len(bad) else ""))
        for _, r in bad.head(5).iterrows():
            print(f"    {r['merchant']} | {r['address']} → ({r['lat']}, {r['lon']})")
        if "category" in coords:
            top = coords.loc[coords["category"].astype(str).str.len() > 0, "category"]
            if not top.empty:
                print("\n업종 대분류 분포:")
                for k, v in top.value_counts().head(10).items():
                    print(f"    {k:<16} {v:>5}건")

    failed = out.loc[out["geo_match"] == "none"]
    if not failed.empty:
        print(f"\n미매칭 예시 (총 {len(failed):,}건 — 행은 유지, 좌표만 결측):")
        for _, r in failed.head(8).iterrows():
            print(f"    {r['merchant']} | {r['address']}")

    out.drop(columns=[c for c in out.columns if c.startswith("_")]).to_csv(
        DEST, index=False, encoding="utf-8")
    print(f"\n✔ 저장: data/raw/{DEST.name}  ({DEST.stat().st_size:,} bytes)")
    print("\n다음 단계:")
    print("  1) Remove-Item data\\processed\\*.parquet")
    print("  2) python scripts/make_sample.py")
    print("  3) pytest tests/ -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
