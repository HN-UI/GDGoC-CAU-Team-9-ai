import json
import re
from typing import List
from google.genai import types
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


class MenuExtractor:
    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def extract(self, image_part: types.Part) -> List[str]:
        prompt = """
Extract ONLY standalone menu item names from the image.

Rules:
- Do NOT include prices.
- Do NOT include options/sizes/add-ons/toppings (e.g., items under "Make your own", "Extra", "+$").
- Output ONLY valid JSON in this format:
{ "items": ["Menu 1", "Menu 2", "..."] }
""".strip()

        text = self.gemma.generate_text([image_part, prompt], max_output_tokens=900)
        data = extract_first_json_object(text) or {}
        return normalize_list(data.get("items", []))