from dataclasses import dataclass

@dataclass
class Document:
    text: str
    page: int
    source: str


@dataclass
class Chunk:
    id: int
    text: str
    page: int
    source: str