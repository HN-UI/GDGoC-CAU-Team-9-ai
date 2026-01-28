import mimetypes
import time
from typing import List, Optional, Union

from google import genai
from google.genai import types


Content = Union[str, types.Part]

class GemmaClient:
    """
    - Gemma(google-genai) 호출만 담당하는 얇은 래퍼
    - 비즈니스 로직(메뉴 추출/랭킹)은 services로 분리
    """

    def __init__(self, api_key: str, model: str = "gemma-3-4b-it"):
        if not api_key:
            raise ValueError("api_key is required")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def image_part_from_file(self, path: str, mime_type: Optional[str] = None) -> types.Part:
        # mime_type 자동 추정 (png/jpg 등)
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(path)
            mime_type = mime_type or "image/jpeg"

        with open(path, "rb") as f:
            data = f.read()

        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def image_part_from_bytes(self, data: bytes, mime_type: str) -> types.Part:
        return types.Part.from_bytes(data=data, mime_type=mime_type)

    def generate_text(self, contents: List[Content], max_output_tokens: int = 900) -> str:
        if not contents:
            raise ValueError("contents must be a non-empty list")

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=0.2,  # JSON/규칙 지키게 낮게
            ),
        )
        return resp.text or ""
