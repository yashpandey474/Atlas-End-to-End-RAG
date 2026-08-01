from abc import abstractmethod

import numpy as np
from code.model.document import Chunk, EmbeddedChunk
from code.model.search import SearchResult
import faiss
import logging
from code.utils.file_utils import check_file_exists, read_from_json, write_to_json
from abc import ABC
logger = logging.getLogger(__name__)

# Abstract class that other store classes will inherit
class VectorStore(ABC):
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

    @abstractmethod
    def save(
        self,
        index_file: str
    )

class FAISSVector(VectorStore):
    embedding_dimension: int
    index: faiss.IndexFlatL2
    index_file: str

    # FAISS index -> Embedded Chunk to store original embedding as well
    index_mapping: dict[int, EmbeddedChunk]

    def __init__(self, embedding_dimension: int, index_file: str, metadata_file: str):
        self.embedding_dimension = embedding_dimension
        self.index_file = index_file

        if check_file_exists(index_file):
            self.load(index_file)
        else:
            logger.info(f"Index file : {index_file} does not exist, cannot load index")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)

        # later, load this from a file too
        if check_file_exists(metadata_file):
            self.load_mapping(metadata_file)
            logger.info(f"Successfully loaded emtadata for {len(self.index_mapping)} chunks from {metadata_file}")
        else:
            logger.info(f"Metadata file: {metadata_file} does not exist, starting fresh")
            self.index_mapping = {}

    def save(self, index_file: str):
        try:
            faiss.write_index(self.index, index_file)
            logger.info(f"Successfully wrote index to file: {index_file}")
        except Exception as e:
            logger.exception(f"Failed to write index to file: {index_file}: {e}")

    def load_mapping(self, metadata_file: str):
        try:
            self.index_mapping = read_from_json(metadata_file)
        except Exception as e;
            logger.exception(f"Failed to load metadata from {metadata_file}: {e}")
            raise

    def save_mapping(self, metadata_file: str):
        try:
            write_to_json(metadata_file, self.index_mapping)
        except Exception as e:
            logger.exception(f"Failed to write metadata to file: {metadata_file}: {e}")
            raise
        
    def load(self, index_file: str):
        # reload data if exists
        try:
            self.index = faiss.read_index(self.index_file)
        except Exception as e:
            logger.exception(f"Error while reading index from file: {index_file}: {e}")
            raise

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
            logger.exception(f"Failed to add {len(chunks)} chunks: {e}")
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
                score=dist,
                rank=i + 1
            ))

        return results
