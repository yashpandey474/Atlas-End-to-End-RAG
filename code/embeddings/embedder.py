from FlagEmbedding import FlagAutoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Embedder:
    model: FlagAutoModel

    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        try:
            self.model = FlagAutoModel.from_finetuned(model_name, use_fp16=True)
            logger.info(f"Loaded model: {model_name} for generating embeddings")
        except Exception as e:
            logger.exception(f"Could not load embedding model due to error: {e}, cannot embed documents.")
            raise # construct valid objects or don't construct them at all

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)