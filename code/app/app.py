from pathlib import Path

from app.factory import create_pipeline
from ingestion.index import Indexer
from embeddings.embedder import Embedder
from llm.prompt_builder import PromptBuilder
from model.enum.llm_provider import LLMProvider
from pipeline.pipeline import RAGPipeline
from pipeline.retriever import DenseRetriever
from utils.constants import DEFAULT_PROMPT_TEMPLATE
from vector_store.vector_store import FAISSVectorStore
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

def main(
    embedding_model: str = 'BAAI/bge-base-en-v1.5',
    llm_model: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",
    chunked_folder_path: str = "../Data/chunked",
    indexing_batch_size: int = 200,
    embedding_batch_size: int = 16
):
    rag_objects = create_pipeline(
        llm_model=llm_model,
        embedding_model=embedding_model,
        device=device
    )
    pipeline, embedder, vector_store = rag_objects["pipeline"], rag_objects["embedder"], rag_objects["vector_store"]

    # Ingest the documents
    indexer: Indexer = Indexer(
        embedder=embedder,
        vector_store=vector_store
    )
    indexer.index_chunked_documents(
        chunked_folder_path=chunked_folder_path,
        embedding_batch_size=embedding_batch_size,
        indexing_batch_size=indexing_batch_size
    )

    # Ask the questions
    pipeline.ask(
        "hello world",
        generation_config=llm_generation_config,
    )

if __name__ == "__main__":
    main()