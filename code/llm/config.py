from dataclasses import dataclass

@dataclass(slots=True frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_new_tokens: int = 512

@dataclass(slots=True frozen=True)
class LLMConfig:
    provider: str
    model: str
    generation_config: GenerationConfig