from pathlib import Path

from ingestion.index import Indexer
from embeddings.embedder import Embedder
from llm.config import LLMConfig, LLMGenerationConfig
from llm.factory import LLMFactory
from llm.llm import LLM
from llm.prompt_builder import PromptBuilder
from model.enum.llm_provider import LLMProvider
from pipeline.pipeline import RAGPipeline
from pipeline.retriever import DenseRetriever
from utils.constants import DEFAULT_PROMPT_TEMPLATE
from vector_store.vector_store import FAISSVectorStore
import logging


logger = logging.getLogger(__name__)

def main():
    embedding_model: str = 'BAAI/bge-large-en-v1.5'
    llm_model: str = "Qwen/Qwen2.5-0.5B"
    device: str = "cpu"
    chunked_folder_path = "../Data/chunked"

    llm_provider: LLMProvider = LLMProvider.HUGGING_FACE

    llm_generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=0,
        max_new_tokens=512
    )

    prompt_builder: PromptBuilder = PromptBuilder(
        prompt_template=DEFAULT_PROMPT_TEMPLATE
    )

    llm_config = LLMConfig(
        model=llm_model,
        provider=llm_provider,
        generation_config=llm_generation_config
    )

    logger.info(f"Creating Embedder with model: {embedding_model} on device: {device}")

    embedder = Embedder(
        model_name = embedding_model,
        device = device
    )

    logger.info(f"Creating FAISSVectorStore with embedding dimension: {embedder.embedding_dimension}")

    vector_store = FAISSVectorStore(
        embedder.embedding_dimension,
        index_file = "faiss_vector_store_index.faiss",
        metadata_file = "faiss_vector_store_metadata.json"
    )

    logger.info(f"Creating DenseRetriever with embedder and vector store")

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store
    )

    logger.info(f"Creating LLM with config: {llm_config}")

    llm: LLM = LLMFactory.create(
        config=llm_config
    )


    pipeline: RAGPipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder
    )

    # Ingest the documents
    indexer: Indexer = Indexer(
        embedder=embedder,
        vector_store=vector_store
    )
    indexer.index_chunked_documents(
        chunked_folder_path=chunked_folder_path,
        batch_size=32
    )

    # Ask the questions
    pipeline.ask(
        "hello world",
        generation_config=llm_generation_config,
    )

if __name__ == "__main__":
    main()