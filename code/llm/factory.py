

from llm.config import LLMConfig
from llm.huggingface_llm import HuggingFaceLLM
from llm.llm import LLM
from model.enum.llm_provider import LLMProvider

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