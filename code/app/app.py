from pathlib import Path

from app.factory import create_pipeline
from code.llm.config import LLMGenerationConfig
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
    llm_generation_temperature: float = 0,
    llm_generation_max_new_tokens: int = 512,
):
    llm_generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=llm_generation_temperature,
        max_new_tokens=llm_generation_max_new_tokens
    ),

    pipeline: RAGPipeline = create_pipeline(
        llm_model=llm_model,
        embedding_model=embedding_model,
        device=device
    )

    pipeline.ask(
        "hello world",
        generation_config=llm_generation_config,
    )

if __name__ == "__main__":
    main()