"""
Interactive CLI for the RAG pipeline.
"""

import logging
from app.factory import create_pipeline
from llm.config import LLMGenerationConfig
from pipeline.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def main(
    llm_model: str = "Qwen/Qwen2.5-0.5B",
    llm_generation_temperature: float = 0,
    llm_generation_max_new_tokens: int = 512
):
    generation_config: LLMGenerationConfig = LLMGenerationConfig(
        temperature=llm_generation_temperature,
        max_new_tokens=llm_generation_max_new_tokens
    )

    logger.info(f"Creating lLM pipeline with model: {llm_model} and generation config: {generation_config}")

    pipeline: RAGPipeline = create_pipeline(
        llm_model=llm_model,
        llm_generation_config=generation_config,
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
    main()