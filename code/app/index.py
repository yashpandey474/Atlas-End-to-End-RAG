import logging

from ingestion.index import Indexer
from embeddings.embedder import Embedder
from vector_store.vector_store import FAISSVectorStore
from utils.inference_utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


CHUNKED_DATA_FOLDER_PATH = "../Data/chunked"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5' 

INDEX_FILE = "faiss_vector_store_index.faiss"
METADATA_FILE = "faiss_vector_store_metadata.json"

DEVICE = get_device()

INDEXING_BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 16

def main(
    embedding_model: str,
    embedding_index_file: str,
    embedding_metadata_file: str,
    embedding_batch_size: int,
    indexing_batch_size: int,
    chunked_data_folder_path: str,
    device: str,
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
        embedding_batch_size=embedding_batch_size
    )


if __name__ == "__main__":
    main(
        embedding_model=EMBEDDING_MODEL,
        embedding_index_file=INDEX_FILE,
        embedding_metadata_file=METADATA_FILE,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        indexing_batch_size=INDEXING_BATCH_SIZE,
        chunked_data_folder_path=CHUNKED_DATA_FOLDER_PATH,
        device=DEVICE
    )