

from code.llm.config import LLMConfig
from code.llm.huggingface_llm import HuggingFaceLLM
from code.llm.llm import LLM


class LLMFactory:

    @staticmethod
    def create(
        config: LLMConfig,
    ) -> LLM:
        if config.provider == "huggingface":
            return HuggingFaceLLM(config.model)

        raise ValueError(
            f"Unknown provider {config.provider}"
        )