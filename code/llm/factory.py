

from code.llm.config import LLMConfig
from code.llm.huggingface_llm import HuggingFaceLLM
from code.llm.llm import LLM
from code.model.enum.llm_provider import LLMProvider

class LLMFactory:

    @staticmethod
    def create(
        config: LLMConfig,
    ) -> LLM:
        if config.provider == LLMProvider.HUGGING_FACE:
            return HuggingFaceLLM(config.model)

        raise ValueError(
            f"Unknown provider {config.provider}"
        )