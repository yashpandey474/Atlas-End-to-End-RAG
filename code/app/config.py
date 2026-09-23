from code.model.enum.llm_provider import LLMProvider
from code.utils.inference_utils import get_device

LLM_MODEL = "Qwen/Qwen2.5-0.5B"
EMBEDDING_MODEL = 'BAAI/bge-base-en-v1.5' 

LLM_PROVIDER = LLMProvider.HUGGING_FACE
LLM_TEMPERATURE = 0
LLM_MAX_NEW_TOKENS = 512
INDEXING_BATCH_SIZE = 200
EMBEDDING_BATCH_SIZE = 16
DEVICE = get_device()

INDEX_FILE =  "faiss_vector_store_index.faiss",
INDEX_METADATA_FILE = "faiss_vector_store_metadata.json",
CHUNKED_DATA_FOLDER_PATH = "../data/chunked"

INDEX_FILE = "faiss_vector_store_index.faiss"
METADATA_FILE = "faiss_vector_store_metadata.json"

