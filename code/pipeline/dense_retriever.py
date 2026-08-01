from code.embeddings.embedder import Embedder
from code.model.search import SearchResult
from code.vector_store.vector_store import VectorStore

class DenseRetriever:
    embedder: Embedder
    vector_store: VectorStore

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        k: int
    ) -> list[SearchResult]:
        # embed the query
        query_embedding = self.embedder.embed(query)

        # retrieve most relevant results
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=k
        )

        return search_results