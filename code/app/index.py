import logging

from ingestion.index import Indexer
from embeddings.embedder import Embedder
from vector_store.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

def main(
    embedding_model: str = 'BAAI/bge-base-en-v1.5',
    embedding_index_file: str = "faiss_vector_store_index.faiss",
    embedding_metadata_file: str = "faiss_vector_store_metadata.json",
    embedding_batch_size: int = 16,
    indexing_batch_size: int = 200,
    chunked_data_folder_path = "../Data/chunked",
    device: str = "cpu",
):
    logger.info("Starting the indexing pipeline...")

    embedder = Embedder(
        model_name=embedding_model,
        device=device
    )

    vector_store = FAISSVectorStore(
        embedding_dimension=embedder.embedding_dimension,
        index_file=embedding_index_file,
        metadata_file=embedding_metadata_file
    )

    indexer: Indexer = Indexer(
        embedder=embedder,
        vector_store=vector_store
    )

    indexer.index_chunked_documents(
        chunked_folder_path=chunked_data_folder_path,
        embedding_batch_size=embedding_batch_size,
        indexing_batch_size=indexing_batch_size
    )


        