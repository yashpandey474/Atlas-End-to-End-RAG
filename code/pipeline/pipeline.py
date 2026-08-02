## This file will tie in the different componnets
from code.llm.llm import LLM
from code.llm.prompt_builder import PromptBuilder
from code.model.answer import Answer
from code.pipeline.dense_retriever import DenseRetriever
from code.llm.config import GenerationConfig

class RAGPipeline:
    retriever: DenseRetriever
    llm: LLM
    prompt_builder: PromptBuilder

    def __init__(
        self,
        retriever: DenseRetriever,
        llm: LLM,
        prompt_builder: PromptBuilder
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder

    def ask(
        self,
        question: str,
        generation_config: GenerationConfig,
        k: int = 5,
    ) -> Answer:
        # fetch search results
        search_results = self.retriever(question, k)

        # build prompt
        built_prompt = self.prompt_builder.build(
            question,
            search_results
        )

        # generate answer using LLM
        answer: str = self.llm.generate(
            built_prompt,
            generation_config
        )

        return Answer(
            question=question,
            answer=answer,
            search_results=search_results
        )
