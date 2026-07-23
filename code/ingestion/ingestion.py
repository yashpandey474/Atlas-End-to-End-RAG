
from pathlib import Path
import json
from code.model.document import Chunk, Document
from dataclasses import asdict

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
            chunk_number=i,
            start_index=i,
            end_index=min(i+chunk_size, len(document.text)),
            text=document.text[i: i + chunk_size],
            source=document.source,
            page=document.page,
        )   
        chunks.append(chunk)

        curr_chunk_num += 1
        i += chunk_size - overlap

    print(f"For document with source {document.source} and page {document.page} - created {len(chunks)} chunks")
    return chunks

def chunk_file(processed_data_filepath: str, overlap: int, chunk_size: int) -> list[Chunk]:
    chunks = []

    with open(processed_data_filepath, "r") as f:
        file_content = json.load(f)

    file_documents = [Document(
        text=doc['text'],
        page=doc['page'],
        source=doc['source']
    ) for doc in file_content]

    for doc in file_documents:
        chunks.extend(
            chunk_document(doc, overlap=overlap, chunk_size=chunk_size)
        )
    
    return chunks

def chunk(
        data_processed_folderpath: str,
        data_chunked_folderpath: str,
        overlap: int,
        chunk_size: int
) -> list[Chunk]:
    # chunk the documents into chunk size and overlap
    # overlap is used so that the sentence's semantic meaning is preserved
    
    data_processed_folder = Path(data_processed_folderpath)
    chunks = []
    for f in data_processed_folder.iterdir():
        if f.is_file():
            processed_data_filepath = data_processed_folder / f.name
            chunks.extend(
                chunk_file(processed_data_filepath=processed_data_filepath, overlap=overlap, chunk_size=chunk_size)
            )
            chunked_data_folder = Path(data_chunked_folderpath)
            chunked_data_folder.mkdir(parents=True, exist_ok=True)
            chunk_data_filepath = chunked_data_folder / (f.name[:-5] + "_chunked.json")
            with open(chunk_data_filepath, 'w') as f:
                json.dump([asdict(chunk) for chunk in chunks], f)

    print(f"For {data_processed_folder} folder - created {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    chunk("Data/processed", "Data/chunked", 50, 300)

