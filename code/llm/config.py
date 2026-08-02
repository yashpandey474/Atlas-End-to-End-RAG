from dataclasses import dataclass
from code.model.enum.llm_provider import LLMProvider

@dataclass(slots=True, frozen=True)
class LLMGenerationConfig:
    temperature: float = 0.0
    max_new_tokens: int = 512

@dataclass(slots=True, frozen=True)
class LLMConfig:
    provider: LLMProvider
    model: str
    generation_config: LLMGenerationConfig