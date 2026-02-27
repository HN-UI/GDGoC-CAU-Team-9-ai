from typing import List

from google.genai import types

from app.agents.extract_agent import MenuExtractAgent
from app.clients.gemma_client import GemmaClient


class MenuExtractor:
    """
    하위 호환용 래퍼.
    내부 구현은 Agent(MenuExtractAgent)로 위임한다.
    """

    def __init__(self, gemma: GemmaClient):
        self.agent = MenuExtractAgent(gemma)

    def extract(self, image_part: types.Part) -> List[str]:
        return self.agent.run(image_part).items
