from FlagEmbedding import FlagAutoModel
import numpy as np
import logging
from model.document import EmbeddedChunk, Chunk
from model.metrics import EmbeddingMetrics
import math
from time import perf_counter

logger = logging.getLogger(__name__)

class Embedder:
    model: FlagAutoModel
    _embedding_dimension: int
    device: str

    def __init__(
            self,
            model_name: str = 'BAAI/bge-large-en-v1.5',
            device: str = 'cpu'
        ):
        try:

            if not device or not device.lower() in ['cuda', 'cpu']:
                raise ValueError(f"Device value is invalid: {device}")

            self.device = device
            self.model: FlagAutoModel = FlagAutoModel.from_finetuned(model_name, use_fp16=True, device=device)
            logger.info(f"Loaded model: {model_name} for generating embeddings")
            self._embedding_dimension = self.model.model.config.hidden_size
            logger.info(f"For model: {model_name}, embedding dimension: {self.embedding_dimension}")


        except Exception as e:
            logger.exception(f"Could not load embedding model due to error: {e}, cannot embed documents.")
            raise # construct valid objects or don't construct them at all

    def embed(self, text: str) -> np.ndarray:
        logger.info(f"Generating embedding for text of length: {len(text)}")
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate normalized embeddings for a batch of texts.

        Args:
            texts: Input texts.

        Returns:
            A NumPy array of shape (len(texts), embedding_dimension).
        """
        if not texts:
            logger.error(f"Empty array of texts provided to embedder, nothing to embed")
            return np.empty((0, 0), dtype=np.float32)


        # divide into batches to be able to track progress
        embeddings = []
        num_batches = math.ceil(len(texts)/batch_size)
        logger.info(f"Generating embedding for {len(texts)} texts using batch size of {batch_size} - {num_batches} batches")
        batch_idx = 1
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))

            batch = texts[start: end]

            start_time = perf_counter()
            batch_embeddings = self.model.encode(batch)

            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings,
                axis=1,
                keepdims=True
            )

            elapsed = perf_counter() - start_time

            logger.info(f"Generated embeddings for batch {batch_idx}/{num_batches} ({len(batch)} texts) in {elapsed} seconds")
            batch_idx+=1
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)


    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        return self.embed_batch_chunk([chunk])[0]

    def embed_batch_chunk(self, chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
        if not chunks:
            logger.info(f"No chunks provided to embedder, nothing to embed")
            return []

        logger.info(f"Chunks received for embedding: {chunks}")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_batch(texts=texts, batch_size=batch_size)

        return [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding
            ) for chunk, embedding in zip(chunks, embeddings)
        ]
    
    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension