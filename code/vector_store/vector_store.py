import numpy as np
from code.model.document import Chunk, EmbeddedChunk

class VectorStore:
    # Owns:
    ## FAISS Index
    ## FAISS Index -> Chunk ID

    # store vector -> ID
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk]
    ) -> None:
        embeddings = [embedded_chunk.embedding for embedded_chunk in embedded_chunks]
        chunks = [embedded_chunk.chunk for embedded_chunk in embedded_chunks]

        ## Logic to add embeddings corresponding to chunk into vector store 
    def search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[Chunk]:
        pass