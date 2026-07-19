import fitz
from pathlib import Path
from model.document import Document
import json

def parse_pdf(pdf_filepath: str, pdf_filename: str) -> list[Document]:
    # open the pdf
    current_documents = []
    doc = fitz.open(pdf_filepath)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        document = Document(
            source=pdf_filename,
            text=text,
            page=page_num + 1
        )
        current_documents.append(document)
    return current_documents

def parse_folder_pdfs(
    raw_data_folder_pathname: str = "../../Data/raw"
):
    documents = []
    raw_data_folder = Path(raw_data_folder_pathname)
    for f in raw_data_folder.iterdir():
        if f.is_file():
            documents.extend(parse_pdf(raw_data_folder_pathname + "/" + f.name, f.name))

    return documents

def save_parsed_pdfs(
    data_folderpath = "../../Data/",
    raw_data_folderame = "raw",
    processed_data_foldername = "processed"
):
    # for each raw pdf
    raw_data_folderpath = data_folderpath + raw_data_folderame
    raw_data_folder = Path(raw_data_folderpath)

    processed_data_folderpath = data_folderpath + processed_data_foldername
    for f in raw_data_folder.iterdir():
        if f.is_file():
            raw_data_filepath = raw_data_folderpath + "/" + f.name

            # parse the pdf
            pdf_documents = parse_pdf(raw_data_filepath)

            if not pdf_documents:
                print(f"Could not parse any documents for file: {raw_data_filepath}")
                continue

            # save into json
            processed_data_filepath = processed_data_folderpath + "/" + pdf_documents[0].source

            with open(processed_data_filepath) as f:
                json.dump(pdf_documents, f)

            print(f"Saved {len(pdf_documents)} to {processed_data_filepath}")
            
