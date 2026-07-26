from FlagEmbedding import FlagAutoModel
import numpy as np
import logging

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
            
            self.model: FlagAutoModel = FlagAutoModel.from_finetuned(model_name, use_fp16=device == 'cuda')
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

        logger.info(f"Generating embedding for {len(texts)} texts using batch size of {batch_size}")
        return self.model.encode(texts, normalize_embeddings=True, batch_size=batch_size)
    
    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension