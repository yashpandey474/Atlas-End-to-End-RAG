from pydantic import BaseModel
from abc import ABC, abstractmethod

class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512
    ) -> str:
        pass