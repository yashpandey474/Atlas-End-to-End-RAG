
from pathlib import Path
import json
from model.document import Chunk, Document

def build_chunk_id(document: Document, chunk_number: int) -> str:
    return f"{document.source}_page{document.page}_chunk{chunk_number}"

def chunk_document(document: Document, overlap: int, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    curr_chunk_num = 0

    if not document.text:
        print(f"Received empty document {document} to chunk")
        return chunks
    
    if overlap >= chunk_size:
        print(f"Received invalid overlap {overlap} and chunk size {chunk_size}")
        raise ValueError(f"Received invalid overlap {overlap} and chunk size {chunk_size}")
    i = 0
    while i < len(document.text):
        chunk = Chunk(
            id=build_chunk_id(document=document, chunk_number=curr_chunk_num),
            text=document.text[i: i + chunk_size],
            source=document.source,
            page=document.page
        )
        chunks.append(chunk)

        curr_chunk_num += 1
        i += chunk_size - overlap

    print(f"For document with source {document.source} and page {document.page} - created {len(chunks)} chunks")
    return chunks

def chunk(data_processed_filepath: str, overlap: int, chunk_size: int) -> list[Chunk]:
    # chunk the documents into chunk size and overlap
    # overlap is used so that the sentence's semantic meaning is preserved
    
    data_processed_folder = Path(data_processed_filepath)
    chunks = []
    for f in data_processed_folder.iterdir():
        if f.is_file():
            processed_data_filepath = data_processed_filepath + "/" + f.name
            with open(processed_data_filepath, "r") as f:
                file_content = json.load(f)
                file_documents = [Document(**doc) for doc in file_content]

                for doc in file_documents:
                    chunks.extend(
                        chunk_document(doc, overlap=overlap, chunk_size=chunk_size)
                    )
    
    print(f"For {data_processed_filepath} folder - created {len(chunks)} chunks")
    return chunks

