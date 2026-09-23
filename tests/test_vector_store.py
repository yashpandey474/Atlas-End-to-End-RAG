import faiss
import numpy as np
import pytest
from code.utils.file_utils import check_file_exists
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)
dimension = 768

def test_basic_faiss():
    index = faiss.IndexFlatL2(dimension)

    embedding = np.random.rand(
        1,
        dimension
    ).astype(np.float32)

    logger.info("Before add")

    index.add(embedding)

    logger.info("After add")

    query = np.random.rand(
        1,
        dimension
    ).astype(np.float32)

    distances, indices = index.search(query, 5)

    logger.info("After search")
    logger.info(distances)
    logger.info(indices)

def test_faiss_multivector_add_and_search():
    logger.info("FAISS:", faiss.__version__)

    d = 768

    index = faiss.IndexFlatL2(d)

    vectors = np.random.rand(100, d).astype(np.float32)
    query = np.random.rand(1, d).astype(np.float32)

    logger.info("Adding...")
    index.add(vectors)

    logger.info("ntotal:", index.ntotal)
    logger.info("Searching...")

    distances, indices = index.search(query, 5)

    logger.info(distances)
    logger.info(indices)

def test_load_index():
    """ Read the index and then search"""
    index_file = "faiss_vector_store_index.faiss"
    if not check_file_exists(index_file):
        logger)
    index = faiss.read_index()

    logger.info("dimension:", index.d)
    logger.info("ntotal:", index.ntotal)

    query = np.random.rand(1, index.d).astype(np.float32)
    query /= np.linalg.norm(query)

    logger.info("Searching...")
    distances, indices = index.search(query, 5)

    logger.info(distances)
    logger.info(indices)