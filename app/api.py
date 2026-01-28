import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.clients.gemma_client import GemmaClient
from app.services.step1_extract import MenuExtractor
from app.services.step2_rank import MenuRanker
from app.utils.image_io import load_image_from_url


app = FastAPI(title="Menu AI API", version="0.1")

# --- singletons ---
gemma = GemmaClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("MODEL_ID", "gemma-3-4b-it"),
)
extractor = MenuExtractor(gemma)
ranker = MenuRanker(gemma, uncertainty_penalty=40)


class RankRequest(BaseModel):
    image_url: str = Field(..., description="메뉴판 이미지 URL")
    avoid: List[str] = Field(default_factory=list, description="기피 재료 리스트")


class RankResponse(BaseModel):
    items_extracted: List[str]
    items: list
    best: dict


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    # 1) URL에서 이미지 다운로드
    data, mime = load_image_from_url(req.image_url)

    # 2) Gemma 입력 Part 만들기
    img_part = gemma.image_part_from_bytes(data, mime)

    # 3) Step1: 메뉴 추출
    items = extractor.extract(img_part)

    # 4) Step2: 랭킹
    ranked = ranker.rank(items, req.avoid)

    # 5) 백엔드에 넘길 최종 응답
    return {
        "items_extracted": items,
        "items": ranked["items"],  # score 내림차순 정렬된 리스트
        "best": ranked["best"],
    }
