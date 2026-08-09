from dataclasses import dataclass
from model.document import Chunk

@dataclass(slots=True, frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int