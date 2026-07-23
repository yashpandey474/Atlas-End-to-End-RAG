from dataclasses import dataclass

@dataclass
class Document:
    text: str
    page: int
    source: str

@dataclass
class Chunk:
    id: int
    chunk_number: int # in the page
    start_index: int # in the page
    end_index: int
    text: str
    page: int
    source: str