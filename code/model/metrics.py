from dataclasses import dataclass

@dataclass(slots=True)
class EmbeddingMetrics:
    total_time: float
    throughput: float
    num_chunks: int