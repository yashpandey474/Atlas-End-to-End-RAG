from pydantic import BaseModel

from code.model.search import SearchResult
from code.utils.constants import DEFAULT_PROMPT_TEMPLATE
import logging

logger = logging.getLogger(__name__)

class PromptBuilder(BaseModel):
    prompt_template: str

    def __init__(
        self,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ):
        self.prompt_template = prompt_template

    def search_results_context(
        search_results: list[SearchResult]
    ) -> str:

        if not search_results:
            return ""

        context = []
        for idx, result in enumerate(search_results):
            context.append(f"""
            Document: {idx}
            Source: {result.chunk.source}
            Page: {result.chunk.page}

            {result.chunk.text}
            """)

        return "\n\n".join(context)

    def build(
        self,
        query: str,
        search_results: list[SearchResult]
    ) -> str:
        if not search_results:
            logger.info(f"No search results for building prompt")
            return ""

        if not query:
            logger.info(f"No query provided for promtp building")
            return ""

        return self.prompt_template.format(
            query=query,
            search_results=self.search_results_context(search_results)
        )