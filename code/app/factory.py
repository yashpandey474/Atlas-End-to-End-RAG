'''
Responsible for constructing the RAG Application by initializing the necessary components such as the embedder, vector store, and retriever. 
It provides a factory method to create an instance of the RAG application with the specified configuration.
'''

import logging

from code.embeddings.embedder import Embedder
from code.llm.config import LLMConfig, LLMGenerationConfig
from code.llm.factory import LLMFactory
from code.llm.llm import LLM
from code.llm.prompt_builder import PromptBuilder
from code.model.enum.llm_provider import LLMProvider
from code.pipeline.pipeline import RAGPipeline
from code.pipeline.retriever import DenseRetriever
from code.utils.constants import DEFAULT_PROMPT_TEMPLATE
from code.vector_store.vector_store import FAISSVectorStore
logger = logging.getLogger(__name__)

def create_pipeline(
    llm_provider: LLMProvider = LLMProvider.HUGGING_FACE,
    llm_generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=0,
        max_new_tokens=512
    ),
    llm_model: str = "Qwen/Qwen2.5-0.5B",
    embedding_model: str = 'BAAI/bge-base-en-v1.5',
    embedder_index_file: str =  "faiss_vector_store_index.faiss",
    embedder_metadata_file: str = "faiss_vector_store_metadata.json",
    device: str = "cpu"
):
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

    return {
        "pipeline": pipeline,
        "embedder": embedder,
        "vector_store": vector_store
    }