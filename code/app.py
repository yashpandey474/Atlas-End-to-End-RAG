from code.embeddings.embedder import Embedder
from code.llm.config import LLMConfig
from code.pipeline.retriever import DenseRetriever
from code.vector_store.vector_store import FAISSVectorStore


def main():
    pass

if __name__ == "__main__":
    embedding_model: str = 'BAAI/bge-large-en-v1.5'
    llm_model: str = ""
    device = "cpu"

    embedder = Embedder(
        model_name = embedding_model,
        device = device
    )

    vector_store = FAISSVectorStore(
        embedder.embedding_dimension,
        index_file = "faiss_vector_store_index.faiss",
        metadata_file = "faiss_vector_store_metadata.json"
    )

    dense_retriever = DenseRetriever(
        embedder=embedder,
        vector_store=vector_store
    )

    config = LLMConfig(
        
    )






