from code.embeddings.embedder import Embedder
import faiss
import numpy as np


print("Creating embedder")

embedder = Embedder(
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu",
)

print("Before creating embedding")

embedding = embedder.embed("tell me about apple")

print("Creating vector store")
index = faiss.IndexFlatL2(embedding.shape[0])

print("Embedding returned")
print("shape:", embedding.shape)
print("dtype:", embedding.dtype)
print("first values:", embedding[:5])


print("Before add to vector store")

embedding = np.asarray(
    embedding,
    dtype=np.float32
).reshape(1, -1)

index.add(embedding)

print("After add")

distances, indices = index.search(embedding, 1)

print("After search")
print(distances)
print(indices)