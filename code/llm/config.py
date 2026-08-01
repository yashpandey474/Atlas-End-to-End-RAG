from dataclasses import dataclass

@dataclass(slots=True frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.0
    max_new_tokens: int = 512