## File to index the chunked documents and create the FAISS vector store
from pathlib import Path
from utils.processing_utils import call_in_batches
from model.document import Chunk
from embeddings.embedder import Embedder
from vector_store.vector_store import FAISSVectorStore
import json
import logging

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(
        self, 
        embedder: Embedder,
        vector_store: FAISSVectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def load_chunks_from_file(self, file_path: str) -> list[Chunk]:
        content = file_path.read_text()
        file_chunks = []
        try:
            content_json = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from file {file_path.name}: {e}")
            return []

        for chunk_json in content_json:
            try:
                file_chunks.append(Chunk(**chunk_json))
            except TypeError as e:
                logger.error(f"Error creating Chunk from JSON: {e}")

        return file_chunks

    def embed_and_add_chunks(self, chunks: list[Chunk], embedding_batch_size: int = 16):
        if not chunks:
            logger.warning("No chunks provided for embedding and adding to vector store.")
            return

        logger.info(f"Embedding and adding {len(chunks)} chunks to the vector store.")
        total_chunks = len(chunks)
        for start in range(0, len(chunks), embedding_batch_size):
            end = min(start + embedding_batch_size, total_chunks)
            batch = chunks[start:end]
            logger.info("Embedding chunks: %d to %d", start, end)
            embedded_chunks = self.embedder.embed_batch_chunk(
                batch,
                batch_size=embedding_batch_size)
            logger.info("Adding embedded chunks to vector store: %d to %d", start, end)
            self.vector_store.add(embedded_chunks)


    def index_chunked_documents(
        self,
        chunked_folder_path: str,
        embedding_batch_size: int = 16,
    ):
        dir_path = Path(chunked_folder_path)
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                logger.info(f"Indexing chunks from file: {file_path.name}")
                file_chunks = self.load_chunks_from_file(file_path)

                logger.info(f"Loaded {len(file_chunks)} chunks from file: {file_path.name}")
                total_chunks = len(file_chunks)

                self.embed_and_add_chunks(file_chunks, embedding_batch_size)
                logger.info(f"Completed indexing chunks from file: {file_path.name}")

        self.vector_store.save()
        logger.info(f"Completed indexing all chunks. Total chunks indexed: {total_chunks}")