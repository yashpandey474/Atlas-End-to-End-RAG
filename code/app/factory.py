'''
Responsible for constructing the RAG Application by initializing the necessary components such as the embedder, vector store, and retriever. 
It provides a factory method to create an instance of the RAG application with the specified configuration.
'''

import logging

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
logger = logging.getLogger(__name__)

def create_pipeline(
    llm_provider: LLMProvider,
    llm_generation_config: LLMGenerationConfig,
    llm_model: str,
    embedding_model: str,
    embedder_index_file: str,
    embedder_metadata_file: str,
    device: str
) -> RAGPipeline:
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
        index_file=embedder_index_file,
        metadata_file=embedder_metadata_file
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

    logger.info(f"LLM loaded with model: {llm_model} from provider: {llm_provider}")


    pipeline: RAGPipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder
    )

    return pipeline