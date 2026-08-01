from pydantic import BaseModel

from code.model.search import SearchResult
from code.utils.constants import DEFAULT_PROMPT_TEMPLATE

class PromptBuilder(BaseModel):
    prompt_template: str

    def __init__(
        self,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ):
        self.prompt_template = prompt_template

    def build(
        self,
        query: str,
        search_results: list[SearchResult]
    ) -> str:
        return self.prompt_template.format(
            query=query,
            search_results=search_results
        )