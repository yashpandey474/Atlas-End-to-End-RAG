"""
Interactive CLI for the RAG pipeline.
"""

import logging
from app.factory import create_pipeline
from code.app.config import DEVICE, EMBEDDING_MODEL, INDEX_FILE, INDEX_METADATA_FILE, LLM_MODEL, LLM_MAX_NEW_TOKENS, LLM_PROVIDER, LLM_TEMPERATURE
from code.model.enum.llm_provider import LLMProvider
from llm.config import LLMGenerationConfig
from pipeline.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    llm_provider: LLMProvider,
    llm_model: str,
    llm_generation_temperature: float,
    llm_generation_max_new_tokens: int,
    embedding_model: str,
    embedder_index_file: str,
    embedder_metadata_file: str,
    device: str
):
    generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=llm_generation_temperature,
        max_new_tokens=llm_generation_max_new_tokens
    )

    logger.info(f"Creating lLM pipeline with model: {llm_model} and generation config: {generation_config}")

    pipeline: RAGPipeline = create_pipeline(
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_generation_config=generation_config,
        embedding_model=embedding_model,
        embedder_index_file=embedder_index_file,
        embedder_metadata_file=embedder_metadata_file,
        device=device
    )

    print()
    print("=" * 60)
    print("Retrieval-Augmented Generation Pipeline")
    print("=" * 60)
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        try:
            question = input("Question: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        logger.info(f"Received question: {question}")
        try:
            answer = pipeline.ask(
                question,
                generation_config=generation_config,
                k = 5
            )

            print()
            print(f"Answer: {answer.answer}")
            print("-" * 60)


            for result in answer.search_results:
                print(f"Rank: {result.rank} | Distance: {result.score:.4f} | Chunk ID: {result.chunk.id}")
                print(f"Chunk Text: {result.chunk.text}")
                print("-" * 60)

        except Exception as e:
            logger.exception(f"Error while processing the question: {e}")
            print(f"An error occurred: {e}")
            print("-" * 60)

if __name__ == "__main__":
    main(
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        llm_generation_temperature=LLM_TEMPERATURE,
        llm_generation_max_new_tokens=LLM_MAX_NEW_TOKENS,
        embedding_model=EMBEDDING_MODEL,
        embedder_index_file=INDEX_FILE,
        embedder_metadata_file=INDEX_METADATA_FILE,
        device=DEVICE
    )