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
    index: faiss.IndexFlatL2
    index_file: str

    # FAISS index -> Embedded Chunk to store original embedding as well
    index_mapping: dict[int, EmbeddedChunk]

    def __init__(self, embedding_dimension: int, index_file: str):
        self.embedding_dimension = embedding_dimension
        self.index_file = index_file
        self.load(index_file)
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.index_mapping = {}

    def save(self, index_file: str):
        try:
            faiss.write_index(self.index, index_file)
            logger.info(f"Successfully wrote index to file: {index_file}")
        except Exception as e:
            logger.exception(f"Failed to write index to file: {index_file}: {e}")

    def load(self, index_file: str):
        # reload data if exists
        if check_file_exists(index_file):
            try:
                self.index = faiss.read_index(self.index_file)
            except Exception as e:
                logger.exception(f"Error while reading index from file: {index_file}: {e}")
                raise
        else:
            logger.info(f"Index file : {index_file} does not exist, cannot load index")

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

        # Add to vector store
        try:
            self.index.add(embeddings)
        except Exception as e:
            logger.exception(f"Failed to add {len(chunks)} chunks")
            raise

        # Add to index mapping
        for i in range(len(embedded_chunks)):
            self.index_mapping[current_size + i] = embedded_chunks[i]

        logger.info(f"Successfully added {len(chunks)} to vector store")


    def search( 
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[SearchResult]:
        """
        Return list of search result
        """
        if query_embedding is None or k <= 0:
            logger.warning(f"No query embedding or no k: {query_embedding} : {k}")
            return []
        
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            logger.info(f"Index: {idx} - Distance: {dist} - Rank: {i + 1}")

            if idx not in self.index_mapping:
                logger.error(f"Index: {idx} from search is not in mapping")
                continue

            results.append(SearchResult(
                chunk=self.index_mapping[idx].chunk,
                score=1-dist,
                rank=i + 1
            ))

        return results
