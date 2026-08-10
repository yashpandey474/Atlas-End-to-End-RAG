from abc import ABC, abstractmethod
from pathlib import Path

from model.document import Chunk
from embeddings.embedder import Embedder
from model.search import SearchResult
from vector_store.vector_store import VectorStore
import json

import logging
logger = logging.getLogger(__name__)

class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[SearchResult]:
        pass
    
class DenseRetriever:
    embedder: Embedder
    vector_store: VectorStore

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def add_chunks(
        self,
        chunked_file_path: Path
    ):
        with open(chunked_file_path, mode="r", encoding="utf-8") as file:
            content = file.read()
            content_json = json.loads(content)
            chunks = []
            for chunk_json in content_json:
                try:
                    chunk = Chunk(**chunk_json)
                    chunks.append(chunk)
                except TypeError as e:
                    logger.error(f"Error creating Chunk from JSON: {e}")
                    continue

            embedded_chunks = self.embedder.embed_chunk(chunks)
            self.vector_store.add(embedded_chunks)
            logger.info(f"Added {len(embedded_chunks)} embedded chunks to the vector store from file: {chunked_file_path.name}")
                

    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[SearchResult]:
        # embed the query
        query_embedding = self.embedder.embed(query)

        # retrieve most relevant results
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k
        )

        return search_results