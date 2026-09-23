from code.embeddings.embedder import Embedder
import pytest
import logging
import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

def test_embedding():
    logger.info("Creating embedder")

    embedder = Embedder(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
    )

    logger.info("Embedding")

    embedding = embedder.embed("tell me about apple")

    logger.info("Embedding returned")
    logger.info("shape:", embedding.shape)
    logger.info("dtype:", embedding.dtype)
    logger.info("first values:", embedding[:5])


def test_embed_and_search():
    logger.info("Creating embedder")

    embedder = Embedder(
        model_name="BAAI/bge-base-en-v1.5",
        device="cpu",
    )

    logger.info("Before creating embedding")

    embedding = embedder.embed("tell me about apple")

    logger.info("Creating vector store")
    index = faiss.IndexFlatL2(embedding.shape[0])

    logger.info("Embedding returned")
    logger.info("shape:", embedding.shape)
    logger.info("dtype:", embedding.dtype)
    logger.info("first values:", embedding[:5])


    logger.info("Before add to vector store")

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(1, -1)

    index.add(embedding)

    logger.info("After add")

    distances, indices = index.search(embedding, 1)

    logger.info("After search")
    logger.info(distances)
    logger.info(indices)