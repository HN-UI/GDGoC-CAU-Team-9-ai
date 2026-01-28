import json
import re
from typing import Any, Dict, List
import unicodedata

from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list, clamp_int, clamp_float


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    # 악센트 제거(ã, í 같은 거)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s


class MenuRanker:
    def __init__(self, gemma: GemmaClient, uncertainty_penalty: int = 40):
        self.gemma = gemma
        self.uncertainty_penalty = uncertainty_penalty  # 30~50 추천

    def rank(self, items: List[str], avoid: List[str]) -> Dict[str, Any]:
        items = normalize_list(items)
        avoid = normalize_list(avoid, limit=100)

        if not items:
            return {"items": [], "best": {"menu": None, "score": 0, "reason_ko": "메뉴가 비어 있어요."}}

        prompt = f"""
You are a food-risk assessor for menu items.

INPUT
- Menu items (use ONLY these, do not add/remove): {items}
- User avoid-ingredients list: {avoid}

TASK
For each menu item:
1) Predict likely key ingredients (suspected_ingredients).
2) Decide which avoid ingredients might be present (matched_avoid).
3) Output a risk score and confidence:
    - risk: integer 0~100 (higher = more likely to violate avoid list)
    - confidence: float 0.0~1.0 (higher = more confident about your assessment)
    - suspected_ingredients: up to 3 items
    - matched_avoid: only from avoid list
    - why_ko: ONE short sentence (<= 25 Korean characters if possible)
IMPORTANT
- If you are unsure, still make a best guess BUT set confidence low.
- Output reasons in Korean.

For "menu", copy the menu name EXACTLY from the input list (character-by-character). Do not translate or modify.

OUTPUT: Return ONLY valid JSON (no markdown) with this schema:
{{
  "items": [
    {{
      "menu": "string",
      "risk": 0,
      "confidence": 0.0,
      "suspected_ingredients": ["string"],
      "matched_avoid": ["string"],
      "why_ko": "string"
    }}
  ]
}}
""".strip()

        text = self.gemma.generate_text([prompt], max_output_tokens=2000)
        data = extract_first_json_object(text)
        if not data or not isinstance(data.get("items"), list):
            raise ValueError(f"Step2 parse failed. RAW: {text[:300]}")

        allowed = {norm(m): m for m in items}  # normalized -> original
        scored = []

        for it in data["items"]:
            if not isinstance(it, dict):
                continue

            menu_raw = it.get("menu", "")
            if not isinstance(menu_raw, str):
                continue

            k = norm(menu_raw)
            if k not in allowed:
                continue
            menu = allowed[k]

            risk = clamp_int(it.get("risk", 50), 0, 100, 50)
            conf = clamp_float(it.get("confidence", 0.5), 0.0, 1.0, 0.5)

            suspected = it.get("suspected_ingredients", [])
            if not isinstance(suspected, list):
                suspected = []
            suspected = [s for s in suspected if isinstance(s, str)][:3]

            matched = it.get("matched_avoid", [])
            if not isinstance(matched, list):
                matched = []
            matched = [m for m in matched if isinstance(m, str) and m in avoid][:20]

            why = it.get("why_ko", "")
            if not isinstance(why, str):
                why = ""

            base_score = 100 - risk
            # ✅ confidence 낮으면 추가 감점 (보수적으로)
            final_score = base_score - (1.0 - conf) * self.uncertainty_penalty
            final_score = int(max(0, min(100, round(final_score))))

            scored.append({
                "menu": menu,
                "score": final_score,
                "risk": risk,
                "confidence": conf,
                "matched_avoid": matched,
                "suspected_ingredients": suspected,
                "reason_ko": why,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0] if scored else {"menu": None, "score": 0, "reason_ko": "결과가 없어요."}

        return {"items": scored, "best": best}
