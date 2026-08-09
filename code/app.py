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


def main():
    pass

if __name__ == "__main__":
    embedding_model: str = 'BAAI/bge-large-en-v1.5'
    llm_model: str = "Qwen/Qwen2.5-3B"
    device: str = "cpu"
    llm_provider: LLMProvider = LLMProvider.HUGGING_FACE

    llm_generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=0,
        max_new_tokens=512
    )

    embedder = Embedder(
        model_name = embedding_model,
        device = device
    )

    vector_store = FAISSVectorStore(
        embedder.embedding_dimension,
        index_file = "faiss_vector_store_index.faiss",
        metadata_file = "faiss_vector_store_metadata.json"
    )

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store
    )

    llm_config = LLMConfig(
        model=llm_model,
        provider=llm_provider,
        generation_config=llm_generation_config
    )

    llm: LLM = LLMFactory.create(
        config=llm_config
    )

    prompt_builder: PromptBuilder = PromptBuilder(
        prompt_template=DEFAULT_PROMPT_TEMPLATE
    )
    pipeline: RAGPipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder
    )

    pipeline.ask(
        "hello world"
    )





