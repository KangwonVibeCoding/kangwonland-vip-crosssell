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


def category_premium(category: pd.Series) -> pd.Series:
    """업종 프리미엄 가중 — 부분 문자열 매칭 (원본 업종 표기가 다양하다)."""
    cat = category.astype("string").fillna("")

    def score(value: str) -> float:
        for key, weight in S.CATEGORY_PREMIUM.items():
            if key in value:
                return weight
        return S.CATEGORY_PREMIUM_DEFAULT

    return cat.map(score).astype("float64")


def category_group(category: pd.Series) -> pd.Series:
    """업종을 3그룹으로 압축한다.

    지도 마커는 모든 색이 동시에 보이므로 전체쌍 색약 검증을 통과한 3색까지만
    쓸 수 있다. 원본 업종이 수십 종이어도 3그룹으로 접는다.
    """
    cat = category.astype("string").fillna("")

    def group(value: str) -> str:
        if any(k in value for k in ("숙박", "호텔", "콘도", "펜션")):
            return "숙박·리조트"
        if any(k in value for k in ("골프", "레저", "스키", "관광", "체험")):
            return "레저·골프"
        if any(k in value for k in ("특산", "농산", "쇼핑", "판매", "음식", "식당", "카페")):
            return "식음·특산품"
        return "기타"

    return cat.map(group)


GROUP_COLOR = {
    "숙박·리조트": theme.SERIES[0],
    "레저·골프": theme.SERIES[1],
    "식음·특산품": theme.SERIES[2],
    "기타": theme.INK["muted"],
}


def _hex_to_rgb(value: str) -> list[int]:
    v = value.lstrip("#")
    return [int(v[i:i + 2], 16) for i in (0, 2, 4)]


def build_merchants(merchants: pd.DataFrame,
                    radius_km: float = S.DEFAULT_RADIUS_KM) -> pd.DataFrame:
    """거리·큐레이션 스코어·색상을 붙이고 반경으로 필터링한다."""
    cols = ["merchant", "category", "lat", "lon", "dist_km", "premium",
            "curation", "group", "color", "radius"]
    if merchants.empty:
        return pd.DataFrame(columns=cols)

    df = merchants.copy()
    df = df.assign(
        dist_km=haversine_km(df["lat"], df["lon"]),
        premium=category_premium(df["category"]),
        group=category_group(df["category"]),
    )
    proximity = np.exp(-df["dist_km"] / S.PROXIMITY_DECAY_KM)
    df = df.assign(
        proximity=proximity,
        curation=100.0 * (
            S.CURATION_WEIGHTS["proximity"] * proximity
            + S.CURATION_WEIGHTS["premium"] * df["premium"]
        ),
    )
    df = df.loc[df["dist_km"] <= radius_km].copy()
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.assign(
        color=df["group"].map(lambda g: _hex_to_rgb(GROUP_COLOR.get(g, theme.INK["muted"]))),
        # 마커 반경(m): 큐레이션 스코어에 비례
        radius=(200 + df["curation"] * 12).astype("float64"),
    )
    return df.sort_values("curation", ascending=False).reset_index(drop=True)


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
                     "dist_km", "curation", "color", "radius"]],
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
                    "<br/>거리 {dist_km} km<br/>큐레이션 {curation}",
            "style": {"backgroundColor": theme.INK["surface"],
                      "color": theme.INK["primary"], "fontSize": "12px"},
        },
    )


def curation_packages(merchants: pd.DataFrame, local_items: pd.DataFrame,
                      top_n: int = 3) -> list[dict]:
    """상위 큐레이션 가맹점 × 상위 CSM 특산품 → VIP 로컬 패키지."""
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
            "curation": float(m["curation"]),
            "item": str(item["item"]),
            "item_csm": float(item["csm"]),
            "tier": S.TIER_LABEL.get(str(item["tier"]), str(item["tier"])),
        })
    return out
