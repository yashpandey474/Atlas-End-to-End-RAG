from pydantic import BaseModel
from abc import ABC, abstractmethod

from code.llm.config import GenerationConfig

class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig
    ) -> str:
        pass