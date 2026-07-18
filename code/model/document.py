from dataclasses import dataclass

@dataclass
class Document:
    text: str
    page: int
    source: str