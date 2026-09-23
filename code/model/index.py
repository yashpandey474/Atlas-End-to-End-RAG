from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

@dataclass
class IndexedDocument:
    sha256: str
    num_chunks: int

@dataclass
class IndexManifest:
    version: int = 1
    embedding_model: str = ""
    embedding_dimension: int = 0
    documents: dict[str, IndexedDocument] = field(default_factory=dict)

    @staticmethod
    def hash_file(path: Path) -> str:
        hasher = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(1024*1024):
                hasher.update(chunk)

        return hasher.hexdigest()

    def save(self, path) -> None:
        try:
            data = {
                "version": self.version,
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
                "documents": {
                    source: {
                        "sha256": document.sha256,
                        "num_chunks": document.num_chunks,
                    }
                    for source, document in self.documents.items()
                },
            }

            path.parent.mkdir(parents=True, exist_oks=True)

            with path.open("w") as file:
                json.dump(data, file, indent=2)

            print(f"Successfully saved index manifest to: {path}")

        except Exception as e:
            print(f"Failed to save manifest to: {path} with error: {e}")
