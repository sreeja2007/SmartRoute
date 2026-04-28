from PyPDF2 import PdfReader
from docx import Document

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return extract_addresses(text)


def parse_docx(file_path):
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return extract_addresses(text)


def extract_addresses(text):
    lines = text.split("\n")

    addresses = []
    for line in lines:
        if len(line.strip()) > 5:  # simple filter
            addresses.append(line.strip())

    return addresses