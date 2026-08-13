## File to index the chunked documents and create the FAISS vector store
from pathlib import Path
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
        batch_size: int = 32
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

                for chunk_json in content_json:
                    try:
                        chunk = Chunk(**chunk_json)
                        chunks.append(chunk)
                    except TypeError as e:
                        logger.error(f"Error creating Chunk from JSON: {e}")
                        continue
    
        embedded_chunks = self.embedder.embed_batch_chunk(chunks, batch_size=batch_size)
        self.vector_store.add(embedded_chunks)
        logger.info(f"Added {len(embedded_chunks)} embedded chunks to the vector store from folder: {dir_path.name}")
                    