import fitz
from pathlib import Path
from model.document import Document

def parse_pdf(pdf_file_path):
    # open the pdf
    current_documents = []
    doc = fitz.open(pdf_file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        document = Document(
            source=f.name,
            text=text,
            page=page_num + 1
        )
        current_documents.append(document)
    return current_documents

def parse_folder_pdfs(
    raw_data_folder_pathname = "../../Data/raw"
):
    documents = []
    raw_data_folder = Path(raw_data_folder_pathname)
    for f in raw_data_folder.iterdir():
        if f.is_file():
            documents.extend(parse_pdf(raw_data_folder_pathname + "/" + f.name))

    return documents
