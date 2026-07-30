from dataclasses import dataclass

import numpy as np

@dataclass
class Document:
    text: str
    page: int
    source: str

@dataclass(slots=True, frozen=True)
class Chunk:
    id: str
    chunk_number: int # in the page
    start_index: int # in the page
    end_index: int
    text: str
    page: int
    source: str

@dataclass(slots=True, frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarry