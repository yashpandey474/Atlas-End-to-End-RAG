from abc import abstractmethod

import numpy as np
from code.model.document import Chunk, EmbeddedChunk
from code.model.search import SearchResult
import faiss
import logging
from code.utils.file_utils import check_file_exists

logger = logging.getLogger(__name__)

# Abstract class that other store classes will inherit
class VectorStore:
    # store vector -> ID
    @abstractmethod
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk]
    ) -> None:
        pass
        ## Logic to add embeddings corresponding to chunk into vector store 
    
    @abstractmethod
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

    def __init__(self, embedding_dimension: int, index_file: str):
        self.embedding_dimension = embedding_dimension

        # reload data if exists
        if check_file_exists(index_file):
            try:
                self.index = faiss.read_index(index_file)
            except Exception as e:
                logger.error(f"Error while reading index from file: {index_file}")
                raise
            
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index_mapping = {}
        
    def add(
        self,
        embedded_chunks: list[EmbeddedChunk]
    ):
        if not embedded_chunks:
            logger.warning(f"No chunks provided to add. Aborting add to store")
            return
        
        current_size = self.index.ntotal

        #(num_embeddings, embedding_dimension)
        embeddings = np.vstack([embedded_chunk.embedding for embedded_chunk in embedded_chunks])

        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Emeddings provided to add to vectore store are not of correct dimension: {self.embedding_dimension}")

        chunks = [embedded_chunk.chunk for embedded_chunk in embedded_chunks]
        chunk_ids = [chunk.id for chunk in chunks]

        # Add to vector store
        try:
            self.index.add(embeddings)
        except Exception as e:
            logger.error(f"Failed to add chunks: {chunk_ids}")
            raise

        # Add to index mapping
        for i in range(len(chunks)):
            self.index_mapping[current_size + i] = chunks[i]

        logger.info(f"Successfully added {len(chunks)} to vector store")


    def search( 
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[SearchResult]:
        """
        Return list of search result
        """
        if not query_embedding or not k:
            logger.warning(f"No query embedding or no k: {query_embedding} : {k}")
            return []
        
        distances, indices = self.index.search(query_embedding, k)
        chunks = []
        for dist, idx in zip(distances[0], indices[0]):
            logger.info(f"Index: {idx} - Distance: {dist}")

            if idx not in self.index_mapping:
                logger.error(f"")
            chunks.append(self.index_mapping[idx])

        return chunks
