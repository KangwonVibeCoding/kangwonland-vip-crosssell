"""내보내기 계약 — 화면에서 본 것과 파일로 받은 것이 같아야 한다.

집행 캘린더·공략 상품·패키지는 이 대시보드의 **최종 산출물**이다. 화면 밖으로
나가지 못하면 마케터가 날짜를 받아 적어야 하고, 나가더라도 열 이름이 다르거나
한글이 깨지면 받는 쪽에서 다시 만들어야 한다. 둘 다 "제안했지만 집행되지 않는"
같은 결과로 끝나므로 여기서 고정한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from config import settings as S
from src.analysis import inflow as inflow_mod, vip
from src.data import loaders
from src.ui import components as C
from src.views import overview


@pytest.fixture(scope="module")
def period_a():
    data, _ = loaders.load_all_impl()
    start, end = pd.Timestamp("2024-12-01"), pd.Timestamp("2024-12-31")
    ars = data["ars"].loc[data["ars"]["date"].between(start, end)]
    sales = data["sales"].loc[data["sales"]["date"].between(start, end)]
    if ars.empty or sales.empty:
        pytest.skip("구간 A 데이터 없음")
    return data, ars, sales


@pytest.fixture(scope="module")
def calendar_display(period_a):
    """탭1 이 화면에 그리는 것과 **같은 프레임**을 만든다."""
    data, ars, sales = period_a
    infl = inflow_mod.build_inflow(ars, sales)
    vrb = vip.compute_vrb(infl, data["demo"])
    vts = vip.compute_vts(infl, vrb, sales)
    calendar = vip.campaign_calendar(vts)
    return calendar.assign(
        날짜=calendar["date"].dt.strftime("%Y-%m-%d"),
        요일=calendar.get("dow_name", ""),
        유입지수=calendar["inflow"].round(1),
        유입순위=calendar["inflow_rank"],
        VIP소비=calendar["vip_qty"].fillna(0).round(0),
        미달기회=calendar["headroom"].round(3),
        VTS=calendar["vts"].round(1),
        등급=calendar["vts_grade"],
        유입밖=calendar["inflow_only_miss"],
        산출=calendar["source"],
    )[["날짜", "요일", "유입지수", "유입순위", "VIP소비", "미달기회",
       "VTS", "등급", "유입밖", "산출"]]


def test_csv_bytes_carry_bom():
    """한국어 Windows 의 Excel 이 CP949 로 열어 컬럼명을 깨뜨리지 않게 한다."""
    raw = C.to_csv_bytes(pd.DataFrame({"날짜": ["2024-12-07"], "등급": ["피크"]}))
    assert raw.startswith(b"\xef\xbb\xbf"), "BOM 이 없으면 Excel 에서 한글이 깨진다"
    assert raw.decode("utf-8-sig").splitlines()[0] == "날짜,등급"


def test_calendar_export_column_names_match_the_screen(calendar_display):
    """버튼이 '화면에 보이는 그대로'라고 말하므로 열 이름이 같아야 한다.

    화면 표의 열 이름은 st.dataframe 의 column_config 가 붙인 것이라 프레임에는
    압축형만 들어 있다. 그 사본을 그대로 내보내면 문구가 거짓이 된다.
    """
    export = overview.calendar_export(calendar_display)
    # app.py 탭1 의 column_config 가 화면에 세우는 이름들
    on_screen = {"날짜", "요일", "유입 지수", "유입 순위", "VIP 소비", "미달 기회",
                 "VTS", "등급", "유입만 봤다면 놓침", "산출"}
    assert set(export.columns) == on_screen
    assert list(export.columns).index("날짜") == 0, "날짜가 첫 열이어야 일정에 옮긴다"


def test_calendar_export_has_no_raw_booleans(calendar_display):
    """체크박스 열은 CSV 에서 TRUE/FALSE 로 떨어진다 — 사람이 읽는 값으로 바꾼다."""
    export = overview.calendar_export(calendar_display)
    col = export["유입만 봤다면 놓침"]
    assert col.dtype != bool
    assert set(col.dropna().unique()) <= {"예", ""}
    # 수량은 소수점 없이 나가야 한다 (화면도 %d 포맷이다)
    assert export["VIP 소비"].dtype.kind in "iu"


def test_calendar_export_keeps_every_row(calendar_display):
    """상위 N일이 전부 나가야 한다 — 조용히 잘라내면 캘린더가 아니다."""
    export = overview.calendar_export(calendar_display)
    assert len(export) == len(calendar_display) == S.TOP_N_VTS_DAYS


def test_calendar_export_round_trips_as_utf8_sig(calendar_display):
    """파일로 나갔다 돌아와도 한글 컬럼·값이 그대로인지."""
    export = overview.calendar_export(calendar_display)
    import io

    back = pd.read_csv(io.BytesIO(C.to_csv_bytes(export)), encoding="utf-8-sig")
    assert list(back.columns) == list(export.columns)
    assert len(back) == len(export)
