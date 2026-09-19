from abc import abstractmethod

import numpy as np
from model.document import Chunk, EmbeddedChunk
from model.search import SearchResult
import faiss
import logging
from utils.file_utils import check_file_exists, read_from_json, write_to_json
from abc import ABC
from dataclasses import asdict

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
    ) -> None:
        pass

class FAISSVectorStore(VectorStore):
    embedding_dimension: int
    index: faiss.IndexFlatL2
    index_file: str

    # FAISS index -> Chunk to store metadata, embedding is not needed as index stores vector -> index
    index_mapping: dict[int, Chunk]

    def __init__(self, embedding_dimension: int, index_file: str, metadata_file: str):
        self.embedding_dimension = embedding_dimension
        self.index_file = index_file
        self.metadata_file = metadata_file

        if check_file_exists(index_file):
            self.load()
            logger.info(f"Loaded from index file")
        else:
            logger.info(f"Index file : {index_file} does not exist, cannot load index")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)

        # later, load this from a file too
        if check_file_exists(metadata_file):
            self.load_mapping(metadata_file)
        else:
            logger.info(f"Metadata file: {metadata_file} does not exist, starting fresh")
            self.index_mapping = {}

        logger.info(
            "FAISS invariant: ntotal=%d, metadata=%d",
            self.index.ntotal,
            len(self.index_mapping)
        )
        if self.index.ntotal != len(self.index_mapping):
            raise RuntimeError(
                f"FAISS/metadata mismatch: "
                f"index has {self.index.ntotal} vectors but "
                f"metadata has {len(self.index_mapping)} entries"
            )
    
    def save(self):
        self.save_mapping(self.metadata_file)
        logger.info(f"Successfully wrote metadata to file: {self.metadata_file}")

        self.save_index(self.index_file)
        logger.info(f"Successfully wrote index to file: {self.index_file}")


    def load_mapping(self, metadata_file: str):
        try:
            raw_mapping = read_from_json(metadata_file) 

            self.index_mapping = {
                int(index): Chunk(**chunk_data)
                for index, chunk_data in raw_mapping.items()
            }

            logger.info(f"Successfully loaded emtadata for {len(self.index_mapping)} chunks from {metadata_file}")
        except Exception as e:
            logger.exception(f"Failed to load metadata from {metadata_file}: {e}")
            self.index_mapping = {}

    def save_index(self, index_file: str):
        try:
            faiss.write_index(self.index, index_file)
            logger.info(f"Successfully saved FAISS index to {index_file}")
        except Exception as e:
            logger.exception(f"Failed to write index to file: {index_file}: {e}")
            raise
        
    def save_mapping(self, metadata_file: str):
        try:
            serializable_mapping = {
                str(index): asdict(chunk) for index, chunk in self.index_mapping.items()
            }

            write_to_json(metadata_file, serializable_mapping)

            logger.info(f"Successfully wrote metadata for {len(serializable_mapping)} chunks to file {metadata_file}")
        except Exception as e:
            logger.exception(f"Failed to write metadata to file: {metadata_file}: {e}")
            raise

    def load_index(self, index_file: str):
        try:
            self.index = faiss.read_index(index_file)
        except Exception as e:
            logger.exception(f"Failed to load index from file: {index_file}: {e}")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)
            
    def load(self):
        # reload data if exists
        if check_file_exists(self.index_file):
            self.load_index(self.index_file)
            logger.info(f"Successfully loaded index from file: {self.index_file}")
        else:
            logger.info(f"Index file: {self.index_file} does not exist, cannot load index")
            self.index = faiss.IndexFlatL2(self.embedding_dimension)

        if check_file_exists(self.metadata_file):
            self.load_mapping(self.metadata_file)
            logger.info(f"Successfully loaded emtadata for {len(self.index_mapping)} chunks from {self.metadata_file}")
        else:
            logger.info(f"Metadata file: {self.metadata_file} does not exist, starting fresh")
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

        # Add to vector store
        try:
            self.index.add(embeddings)
        except Exception as e:
            logger.exception(f"Failed to add {len(chunks)} chunks: {e}")
            raise

        # Add to index mapping
        for i in range(len(embedded_chunks)):
            self.index_mapping[current_size + i] = embedded_chunks[i].chunk

        logger.info(f"Successfully added {len(chunks)} to vector store")

    def search( 
        self,
        query_embedding: np.ndarray,
        k: int
    ) -> list[SearchResult]:
        """
        Return list of search result
        """

        if self.index.ntotal == 0:
            logger.warning(f"FAISS index is empty")
            return []

        if query_embedding is None or k <= 0:
            logger.warning(f"No query embedding or no k: {query_embedding} : {k}")
            return []

        logger.info(f"Before re-aliging query embedding")

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.ndim != 2:
            raise ValueError(f"Expected 2D query embedding but got {query_embedding.shape}")

        if query_embedding.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected embedding dimention {self.embedding_dimension} but got {query_embedding.shape[1]}"
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains NaN or Inf")

        k = min(k, self.index.ntotal)

        logger.info(
            "query dtype=%s shape=%s contiguous=%s",
            query_embedding.dtype,
            query_embedding.shape,
            query_embedding.flags["C_CONTIGUOUS"]
        )

        logger.info(
            "query min=%f max=%f norm=%f",
            query_embedding.min(),
            query_embedding.max(),
            np.linalg.norm(query_embedding)
        )

        logger.info(f"Searching FAISS: shape={query_embedding.shape}, k={k}, total={self.index.ntotal}")

        distances, indices = self.index.search(
            np.ascontiguousarray(query_embedding),
            k
        )

        logger.info(f"After finding in index")
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            logger.info(f"Index: {idx} - Distance: {dist} - Rank: {i + 1}")

            if idx not in self.index_mapping:
                logger.error(f"Index: {idx} from search is not in mapping")
                continue

            results.append(SearchResult(
                chunk=self.index_mapping[idx],
                score=dist,
                rank=i + 1
            ))

        return results
