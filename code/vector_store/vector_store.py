import numpy as np
from code.model.document import Chunk

class VectorStore:
    # Owns:
    ## FAISS Index
    ## FAISS Index -> Chunk ID

    # store vector -> ID
    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk]
    ) -> None:
        pass

    def search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[Chunk]:
        pass