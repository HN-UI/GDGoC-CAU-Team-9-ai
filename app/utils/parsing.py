# app/utils/parsing.py

import json
import re
from typing import Any, Dict, List, Optional


def strip_code_fence(text: str) -> str:
    """
    ```json ... ``` 또는 ``` ... ``` 코드블록 제거
    """
    if not text:
        return ""
    t = text.strip()

    # 시작 ```json / ``` 제거
    t = re.sub(r"^\s*```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    # 끝 ``` 제거
    t = re.sub(r"\s*```\s*$", "", t)

    return t.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    텍스트에서 첫 번째 JSON 객체({ ... })를 찾아 dict로 파싱.
    - 코드블록이 있으면 제거 후 시도
    - 탐욕적 매칭 문제를 줄이기 위해 '첫 { 부터' 스캔하며 파싱 성공하는 지점에서 종료
    """
    t = strip_code_fence(text)
    if not t:
        return None

    start = t.find("{")
    if start == -1:
        return None

    # 첫 '{' 이후부터 끝까지, 점진적으로 잘라보며 json 파싱 시도
    # (중간에 설명 텍스트가 섞여도 가장 먼저 성공하는 dict를 찾는 방식)
    for end in range(len(t), start, -1):
        chunk = t[start:end].strip()
        if not chunk.endswith("}"):
            continue
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def normalize_list(xs: Any, limit: int = 200) -> List[str]:
    """
    문자열 리스트 정리:
    - 문자열만 남기기
    - 공백 정리
    - 중복 제거
    - 길이 제한
    """
    out: List[str] = []
    seen = set()

    if not isinstance(xs, list):
        return out

    for x in xs:
        if not isinstance(x, str):
            continue
        s = re.sub(r"\s+", " ", x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= limit:
            break

    return out


def clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return max(lo, min(hi, x))


def clamp_float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return max(lo, min(hi, x))
