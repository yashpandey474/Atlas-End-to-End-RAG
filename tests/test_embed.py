from code.embeddings.embedder import Embedder

print("Creating embedder")

embedder = Embedder(
    model_name="BAAI/bge-base-en-v1.5",
    device="cpu",
)

print("Embedding")

embedding = embedder.embed("tell me about apple")

print("Embedding returned")
print("shape:", embedding.shape)
print("dtype:", embedding.dtype)
print("first values:", embedding[:5])