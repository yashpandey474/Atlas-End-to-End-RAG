from pydantic import BaseModel
from abc import ABC, abstractmethod

from code.llm.config import LLMGenerationConfig

class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: LLMGenerationConfig
    ) -> str:
        pass