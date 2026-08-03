"""가맹점 지도 — pydeck 레이어 + 거리·큐레이션 스코어.

pydeck 을 쓰는 이유: streamlit 이 이미 의존하고 있어 추가 패키지가 0개다.
folium 은 streamlit-folium 별도 설치가 필요해 Cloud 빌드 리스크를 늘린다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk

from config import settings as S
from src.ui import theme

EARTH_R_KM = 6371.0088


def haversine_km(lat: pd.Series | float, lon: pd.Series | float,
                 origin: tuple[float, float] = S.CASINO_LATLON) -> pd.Series | float:
    """기준점으로부터의 대권 거리(km)."""
    lat1, lon1 = np.radians(origin[0]), np.radians(origin[1])
    lat2, lon2 = np.radians(lat), np.radians(lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def category_fit(category: pd.Series) -> tuple[pd.Series, pd.Series]:
    """업종 적합도와 '업종을 알고 있는가' 플래그.

    적합도는 데이터에서 유도한 값이 아니라 **판단**이다(S.CATEGORY_FIT). 그래서
    호출자가 가중치 표를 화면에 노출할 수 있도록 하고, 업종을 모르는 행은
    0 이 아니라 **중립값**으로 둔다 — 0 은 '부적합'이라는 없는 근거를 만든다.

    반환: (fit, fit_known)
    """
    cat = category.astype("string").fillna("").str.strip()
    known = cat.str.len() > 0
    fit = cat.map(lambda v: S.CATEGORY_FIT.get(v, S.CATEGORY_FIT_NEUTRAL))
    return (fit.where(known, S.CATEGORY_FIT_NEUTRAL).astype("float64"),
            known.to_numpy(dtype=bool))


def category_group(category: pd.Series) -> pd.Series:
    """업종을 3그룹 + 기타로 압축한다.

    지도 마커는 모든 색이 동시에 보이므로 전체쌍 색약 검증을 통과한 3색까지만
    쓸 수 있다. 실측 분포상 음식(579)·소매(425)·숙박(115)이 전체의 92% 라
    이 셋에 색을 주고 나머지는 회색으로 눕힌다.
    """
    cat = category.astype("string").fillna("").str.strip()
    return cat.map(lambda v: S.CATEGORY_GROUPS.get(v, S.CATEGORY_GROUP_OTHER))


GROUP_COLOR = {
    "음식": theme.SERIES[0],
    "소매": theme.SERIES[1],
    "숙박": theme.SERIES[2],
    S.CATEGORY_GROUP_OTHER: theme.INK["muted"],
}


def _hex_to_rgb(value: str) -> list[int]:
    v = value.lstrip("#")
    return [int(v[i:i + 2], 16) for i in (0, 2, 4)]


def build_merchants(merchants: pd.DataFrame,
                    radius_km: float = S.DEFAULT_RADIUS_KM) -> pd.DataFrame:
    """거리·업종 적합도로 교차판매 적합도를 산출하고 반경으로 필터링한다.

        fit_score = 100 × (0.6 × proximity + 0.4 × 업종 적합도)

    두 축은 서로 독립이다 — 거리는 위치, 적합도는 업종에서 온다. 상관된 축을
    합치면 가중치를 준 만큼 같은 신호를 두 번 세게 된다.

    ⚠ `PNT_USABLE_AMT`(포인트 사용 한도)는 쓰지 않는다. 1,681건 중 1,578건이
    정확히 4,000,000 이라(CV 0.052) 변별력이 없다 — ARS 의 winners 를 유입
    지표에서 뺀 것과 같은 이유다.

    좌표가 없는 행(주소 매칭 실패)은 지도에 못 그리므로 제외하되, 몇 건을
    뺐는지 `attrs["no_coord"]` 에 남긴다. 호출자가 화면에 표기한다.
    """
    cols = ["merchant", "category", "lat", "lon", "dist_km", "fit", "fit_known",
            "proximity", "fit_score", "group", "color", "radius"]
    if merchants.empty:
        out = pd.DataFrame(columns=cols)
        out.attrs["no_coord"] = 0
        return out

    df = merchants.copy()
    if "has_coord" in df.columns:
        no_coord = int((~df["has_coord"].to_numpy(dtype=bool)).sum())
        df = df.loc[df["has_coord"].to_numpy(dtype=bool)]
    else:
        no_coord = int(df["lat"].isna().sum() + 0)
        df = df.dropna(subset=["lat", "lon"])

    if df.empty:
        out = pd.DataFrame(columns=cols)
        out.attrs["no_coord"] = no_coord
        return out

    fit, fit_known = category_fit(df["category"])
    df = df.assign(
        dist_km=haversine_km(df["lat"], df["lon"]),
        fit=fit,
        fit_known=fit_known,
        group=category_group(df["category"]),
    )
    proximity = np.exp(-df["dist_km"] / S.PROXIMITY_DECAY_KM)
    df = df.assign(
        proximity=proximity,
        fit_score=100.0 * (
            S.CURATION_WEIGHTS["proximity"] * proximity
            + S.CURATION_WEIGHTS["fit"] * df["fit"]
        ),
    )
    df = df.loc[df["dist_km"] <= radius_km].copy()
    if df.empty:
        out = pd.DataFrame(columns=cols)
        out.attrs["no_coord"] = no_coord
        return out

    df = df.assign(
        color=df["group"].map(lambda g: _hex_to_rgb(GROUP_COLOR.get(g, theme.INK["muted"]))),
        # 마커 반경(m): 적합도에 비례
        radius=(200 + df["fit_score"] * 12).astype("float64"),
    )
    out = df.sort_values("fit_score", ascending=False).reset_index(drop=True)
    out.attrs["no_coord"] = no_coord
    return out


def _circle_polygon(center: tuple[float, float], radius_km: float,
                    points: int = 72) -> list[list[float]]:
    """반경 원을 다각형 좌표로. 위도 보정을 적용한다."""
    lat, lon = center
    angles = np.linspace(0, 2 * np.pi, points)
    dlat = (radius_km / 111.32) * np.cos(angles)
    dlon = (radius_km / (111.32 * np.cos(np.radians(lat)))) * np.sin(angles)
    return [[float(lon + b), float(lat + a)] for a, b in zip(dlat, dlon)]


def merchant_deck(df: pd.DataFrame, radius_km: float = S.DEFAULT_RADIUS_KM) -> pdk.Deck:
    """가맹점 지도 Deck. 카지노 기준점 + 반경 원 + 가맹점 마커."""
    casino = pd.DataFrame([{
        "merchant": "강원랜드 / 하이원리조트",
        "lat": S.CASINO_LATLON[0], "lon": S.CASINO_LATLON[1],
    }])
    layers = [
        pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": _circle_polygon(S.CASINO_LATLON, radius_km)}],
            get_polygon="polygon",
            get_fill_color=[42, 120, 214, 18],
            get_line_color=[42, 120, 214, 110],
            line_width_min_pixels=1, stroked=True, filled=True, pickable=False,
        ),
    ]
    if not df.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=df[["merchant", "category", "group", "lat", "lon",
                     "dist_km", "fit_score", "color", "radius"]],
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=5, radius_max_pixels=26,
            stroked=True, get_line_color=[252, 252, 251], line_width_min_pixels=2,
            pickable=True, opacity=0.85,
        ))
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=casino,
        get_position=["lon", "lat"],
        get_fill_color=[208, 59, 59, 230],
        get_radius=520,
        radius_min_pixels=9, radius_max_pixels=22,
        stroked=True, get_line_color=[252, 252, 251], line_width_min_pixels=3,
        pickable=True,
    ))

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=S.CASINO_LATLON[0], longitude=S.CASINO_LATLON[1],
            zoom=9.6 if radius_km > 20 else 10.6, pitch=0, bearing=0,
        ),
        map_style=None,     # 외부 타일 요청 없이 렌더 (Cloud 안전)
        tooltip={
            "html": "<b>{merchant}</b><br/>업종 {category}"
                    "<br/>거리 {dist_km} km<br/>교차판매 적합도 {fit_score}",
            "style": {"backgroundColor": theme.INK["surface"],
                      "color": theme.INK["primary"], "fontSize": "12px"},
        },
    )


def curation_packages(merchants: pd.DataFrame, local_items: pd.DataFrame,
                      top_n: int = 3) -> list[dict]:
    """상위 적합도 가맹점 × 상위 CSM 특산품 → VIP 로컬 패키지."""
    if merchants.empty or local_items.empty:
        return []
    out: list[dict] = []
    for i in range(min(top_n, len(merchants), len(local_items))):
        m, item = merchants.iloc[i], local_items.iloc[i]
        out.append({
            "rank": i + 1,
            "merchant": str(m["merchant"]),
            "category": str(m["category"]),
            "dist_km": float(m["dist_km"]),
            "fit_score": float(m["fit_score"]),
            "fit_known": bool(m.get("fit_known", False)),
            "item": str(item["item"]),
            "item_csm": float(item["csm"]),
            "tier": S.TIER_LABEL.get(str(item["tier"]), str(item["tier"])),
        })
    return out
