from dataclasses import dataclass

from code.model.search import SearchResult

@dataclass(slots=True, frozen=True)
class Answer:
    question: str
    answer: str
    search_results: list[SearchResult]