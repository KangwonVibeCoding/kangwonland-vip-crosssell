"""UI 크롬 텍스트의 WCAG 명암비를 측정해 표로 출력한다.

    & ".\\.venv\\Scripts\\python.exe" scripts\\check_contrast.py

색을 바꿨을 때 **바꾼 사람이 바로 확인할 수 있는 창구**다. 같은 값을
`tests/test_contrast.py` 가 회귀로 잡지만, 테스트는 통과/실패만 말하고
"몇 대 몇인지"는 알려주지 않는다. 색을 조정하는 중에는 이쪽이 필요하다.

미달 항목이 있으면 종료코드 1 을 돌려준다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui import contrast  # noqa: E402


def main() -> int:
    print("WCAG 2.1 AA · 일반 텍스트 기준 4.5:1")
    print("배지·칩·캡션은 0.68~0.86rem 이라 'large text' 완화(3:1) 대상이 아니다.\n")
    print(contrast.report())
    failed = [p for p in contrast.ui_chrome_pairs() if not p.passes]
    if failed:
        print("\n미달 조합을 고치려면 같은 색상(H)·채도(S)에서 명도(L)만 낮춘다 —")
        print("색상환 위치가 바뀌면 배지가 어느 구간을 뜻하는지 알아보는 방식이 깨진다.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
