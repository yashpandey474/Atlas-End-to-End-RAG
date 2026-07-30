import numpy as np
from code.model.document import Chunk, EmbeddedChunk
from code.model.search import SearchResult
from faiss import IndexFlatL2
import logging

logger = logging.getLogger(__name__)

# Abstract class that other store classes will inherit
class VectorStore:
    # store vector -> ID
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk]
    ) -> None:
        pass
        ## Logic to add embeddings corresponding to chunk into vector store 
    def search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[Chunk]:
        pass

class FAISSVector(VectorStore):
    embedding_dimension: int
    index: IndexFlatL2

    # FAISS index -> Chunk
    index_mapping: dict[int, Chunk]

    def __init__(self, embedding_dimension: int):
        self.embedding_dimension = embedding_dimension
        self.index = IndexFlatL2(self.embedding_dimension)

    def update_index(
        self,
        chunk_ids: list[int]
    ):
        for i in range(len(chunk_ids)):
            self.index_mapping[len]
        
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk]
    ) -> bool:
        current_size = self.index.ntotal
        embeddings = np.vstack([embedded_chunk.embedding for embedded_chunk in embedded_chunks])
        chunks = [embedded_chunk.chunk for embedded_chunk in embedded_chunks]
        chunk_ids = [chunk.id for chunk in chunks]

        # Add to vector store
        try:
            self.index.add(embeddings)
        except Exception as e:
            logger.error(f"Failed to add chunks: {chunk_ids}")
            return False

        # Add to index mapping
        for i in range(len(chunks)):
            self.index_mapping[current_size + i] = chunks[i]

        return True

    def search(
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[SearchResult]:
        """
        Return list of search result
        """
        distances, indices = self.index.search(query_embedding, k)

