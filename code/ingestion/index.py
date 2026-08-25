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

    def index_chunked_documents(
        self,
        chunked_folder_path: str,
        embedding_batch_size: int = 16,
        indexing_batch_size: int = 200
    ):
        dir_path = Path(chunked_folder_path)
        chunks = []
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                logger.info(f"Indexing chunks from file: {file_path.name}")
                content = file_path.read_text()
                try:
                    content_json = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from file {file_path.name}: {e}")
                    continue

                file_chunks = []
                for chunk_json in content_json:
                    try:
                        chunk = Chunk(**chunk_json)
                        file_chunks.append(chunk)
                    except TypeError as e:
                        logger.error(f"Error creating Chunk from JSON: {e}")
                        continue
                logger.info(f"Loaded {len(file_chunks)} chunks from file: {file_path.name}")
                chunks.extend(file_chunks)

        logger.info(f"Total chunks to embed and add to vector store: {len(chunks)}")
        total_chunks = len(chunks)

        all_embedded_chunks = []

        for start in range(0, total_chunks, embedding_batch_size):
            end = min(start + embedding_batch_size, total_chunks)
            batch = chunks[start:end]
            logger.info("Embedding chunks: %d to %d", start, end)
            embedded_chunks = self.embedder.embed_batch_chunk(batch, batch_size=embedding_batch_size)
            all_embedded_chunks.extend(embedded_chunks)

        for start in range(0, len(all_embedded_chunks), indexing_batch_size):
            end = min(start + indexing_batch_size, len(all_embedded_chunks))
            to_index_batch = all_embedded_chunks[start:end]
            logger.info(f"Adding {len(to_index_batch)} embedded chunks to the vector store")
            self.vector_store.add(to_index_batch)
            logger.info(f"Added {len(to_index_batch)} embedded chunks to the vector store")

        self.vector_store.save()
        logger.info(f"Completed indexing all chunks. Total chunks indexed: {total_chunks}")