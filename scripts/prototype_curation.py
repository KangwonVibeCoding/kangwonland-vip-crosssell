"""AI VIP 큐레이션 추천 JSON — CLI.

로직은 전부 `src/analysis/curation.py` 에 있다. 이 파일은 **출력만** 담당한다:
로드 → curate() → stdout 진단 + JSON 파일. 같은 로직을 앱의 '큐레이션' 탭
(`src/views/curation.py`)이 쓴다 — 화면과 CLI 가 다른 답을 내지 않게 하려는 것이다.

사용법:
    $env:PYTHONIOENCODING="utf-8"; & ".\\.venv\\Scripts\\python.exe" scripts\\prototype_curation.py `
        --persona SEG-F50-ROOM --date 2024-12-21 --out out\\curation.json

    --persona 는 SEG-{F|M}{20|30|40|50|60|70}대-ROOM (12종). 목록은 --help.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                     # noqa: E402

from config import settings as S                        # noqa: E402
from src.analysis import curation                       # noqa: E402
from src.analysis.curation import (                     # noqa: E402
    CROSS_MARGIN_QUANTILE, PERSONAS, SEGMENT_ASSUMED_WINDOW,
)
from src.data import loaders                            # noqa: E402


# ── 출력 ──────────────────────────────────────────────────────────────

def print_segment(pref: pd.DataFrame, seg: pd.DataFrame, persona: dict) -> None:
    """세그먼트 축이 실제로 무엇을 들고 왔는지 — 게이트 탈락분까지 드러낸다."""
    group = persona.get("segment_group", "?")
    if pref.empty:
        print(f"[세그먼트 선호] 표 없음 → 이 축은 비활성 (배수 전부 1.00)")
        return
    lo, hi = SEGMENT_ASSUMED_WINDOW
    print(f"[세그먼트 선호 · {group}]  출처 data/raw/segment_preference.csv "
          f"— [파생] 회원 누적 구성비 기반 추정")
    print(f"  ⚠ [가정] {lo:%Y-%m}~{hi:%Y-%m} 재계산본으로 간주 "
          f"(원파일은 24개월 산출) · 커버리지도 같은 창에서 측정")
    if seg.empty:
        print("  해당 그룹 매칭 0종 → 축 비활성")
        print()
        return
    for r in seg.sort_values("seg_r", ascending=False).itertuples():
        mark = "적용" if r.seg_used else "미적용(커버리지)"
        print(f"  {str(r.seg_item)[:30]:<32} r {r.seg_r:+.2f}"
              f" · 판매월 {int(r.seg_months):>2}  → {mark}")
    used = int(seg["seg_used"].sum())
    print(f"  적용 {used}종 / 표 매칭 {len(seg)}종")
    print()


def print_report(funnel, scored, gate, ch_sig, persona, as_of, day) -> None:
    line = "─" * 78
    print(line)
    print(f"AI VIP 큐레이션 프로토타입 — {persona['label']} · 기준일 {as_of:%Y-%m-%d}")
    print(line)
    print(f"창: {gate.get('window_days')}일 · lift 표본요건 {gate.get('min_days')}일 "
          f"· 실측 {gate.get('measured')}종 / 탈락 {gate.get('dropped')}종 "
          f"· 탄력성 대체값 {gate.get('elasticity_fill', float('nan')):.3f}")
    tickets = day.get("tickets")
    tickets_txt = f" · 입장권 {int(tickets):,}건" if pd.notna(tickets) else ""
    print(f"당일 유입: 지수 {float(day['inflow']):.1f} · 등급 '{day['grade']}'"
          f" · 근거 {day['source']}{tickets_txt}")
    print()
    print("[후보 필터 흐름]")
    for i, (label, n) in enumerate(funnel):
        arrow = "  " if i == 0 else "→ "
        print(f"  {arrow}{label:<44} {n:>6,}종")
    print()
    print("[채널 신호 — 요일통제 편상관 × 래그계수 × 등급계수 × 채널 등급반응]")
    for ch in S.CHANNELS:
        s = ch_sig[ch]
        print(f"  {S.CHANNEL_LABEL[ch]:<8} partial {s['partial']:+.3f}"
              f" · best_lag D+{s['best_lag']} (r {s['lag_pearson']:.3f})"
              f" · 계수 {s['penalty']:.1f}×{s['grade_factor']:.1f}"
              f"×{s['grade_response']:.3f}"
              f" → 신호 {s['signal']:.3f}")
    print()

    def table(title: str, frame: pd.DataFrame) -> None:
        print(title)
        print(f"  {'#':<3}{'상품':<24}{'채널':<10}{'티어':<8}"
              f"{'cur':>7}{'intent':>8}{'계절':>6}"
              f"{'base':>7}{'세그':>6}{'CSM':>7}  {'lift':<10}밴드")
        for i, r in enumerate(frame.itertuples(), 1):
            lift = (f"{r.lift:.2f}" if bool(r.lift_measured) and pd.notna(r.lift)
                    else "NaN(미달)")
            print(f"  {i:<3}{str(r.item)[:22]:<24}{S.CHANNEL_LABEL[r.channel]:<10}"
                  f"{S.TIER_LABEL.get(r.tier, r.tier):<8}"
                  f"{r.cur_score:>7.1f}{r.intent:>8.3f}"
                  f"{r.season:>6.2f}"
                  f"{r.base:>7.3f}{r.seg_mult:>6.2f}{r.csm:>7.1f}  "
                  f"{lift:<10}{'O' if r.in_band else '-'}")
        print()

    # 전체 상위 8 — 3순위(티어 밴드 면제)가 여기서 나온다
    table("[cur_score 상위 8종 · 전체]  (밴드 O = 1·2순위 자격)", scored.head(8))
    # 밴드 통과 상위 8 — 1·2순위는 이 안에서만 고른다
    table("[cur_score 상위 8종 · 티어 밴드 통과분]  (1·2순위 후보 풀)",
          scored.loc[scored["in_band"]].head(8))


# ── 메인 ──────────────────────────────────────────────────────────────


# ── 메인 ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="AI VIP 큐레이션 추천 JSON")
    ap.add_argument("--persona", default="SEG-F50-ROOM", choices=sorted(PERSONAS))
    ap.add_argument("--date", required=True, help="기준일 YYYY-MM-DD")
    ap.add_argument("--out", default="out/curation.json", help="JSON 출력 경로")
    ap.add_argument("--quiet", action="store_true", help="stdout 진단 출력을 생략한다")
    args = ap.parse_args()

    as_of = pd.Timestamp(args.date).normalize()
    data, sources = loaders.load_all_impl()
    try:
        res = curation.curate(data["ars"], data["sales"], data["demo"],
                              args.persona, as_of)
    except ValueError as exc:
        raise SystemExit(str(exc))

    for msg in res["warnings"]:
        print(f"⚠ {msg}")
    if not args.quiet:
        print(f"데이터 출처: {sources}")
        print_segment(res["seg_pref"], res["seg"], res["persona"])
        print_report(res["funnel"], res["scored"], res["gate"], res["ch_sig"],
                     res["persona"], as_of, res["day"])

    text = json.dumps(res["payload"], ensure_ascii=False, indent=2)
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\n✔ 저장: {out}  ({len(text.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
