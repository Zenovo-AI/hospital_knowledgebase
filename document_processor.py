
from PyPDF2 import PdfReader
import streamlit as st
from langchain.docstore.document import Document
import trafilatura
from utils import clean_text
import logging
import openai
import pdfplumber
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

openai.api_key = st.secrets["OPENAI_API_KEY"]

logging.basicConfig(level=logging.INFO)


class DocumentProcessor:
    # Helper function to read text from a TXT file
    def extract_txt_content(self, file_path):
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {e}")
    
    

    def extract_text_and_tables_from_pdf(self, file):
        text = ""
        table_texts = []

        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n[Page {page_num}]\n{page_text}"
                
                # Extract tables
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables):
                    table_str = f"\n\n[Page {page_num} - Table {table_idx + 1}]\n"
                    for row in table:
                        cleaned_row = [cell if cell is not None else "" for cell in row]  # Replace None with ""
                        table_str += " | ".join(cleaned_row) + "\n"
                    table_texts.append(table_str)

        # Combine extracted text and tables
        full_text = text + "\n\n".join(table_texts)
        return full_text

    
    def preprocess_document(self, file):
        """
        Preprocess the document by extracting all text.
        """
        pdf_text = self.extract_text_and_tables_from_pdf(file)
        # Return the entire content as a single Document object
        documents = [Document(page_content=pdf_text)]
        return documents

    def process_webpage(self, url):
        """
        Download and extract text content from a webpage using trafilatura.
        """
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            web_page = trafilatura.extract(downloaded)
            return clean_text(web_page) if web_page else None
        else:
            logging.error(f"Failed to fetch webpage: {url}")
            return None



def extract_text_from_pdf(file_path: Path):
    """Extracts text from a PDF file and splits it into chunks of max 512 characters with 200 overlap."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    return [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in chunks]

def extract_text_from_url(url: str):
    """Extracts text from a URL and splits it into chunks of max 512 characters with 200 overlap."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    
    return [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]


def handle_file_upload(uploaded_files, web_links, documents_dir: Path):
    """Handles processing for both file uploads and web links."""

    placeholder = st.empty()
    
    try:
        processed_documents = []

        # Process uploaded files
        for uploaded_file in uploaded_files:
            file_path = documents_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            documents = []  # List to store chunked documents

            # Process PDF files
            if uploaded_file.type == 'application/pdf':
                documents = extract_text_from_pdf(file_path)  # Returns a list of Document objects

            # Process Text files
            elif uploaded_file.type == 'text/plain':
                text = uploaded_file.getvalue().decode("utf-8")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(text)
                documents = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]

            else:
                placeholder.error(f"Unsupported file type: {uploaded_file.type}")
                continue  # Skip this file
            
            # Iterate over chunked documents
            for doc in documents:
                processed_documents.append(doc)  # Append each chunk separately

        # Process web links
        for url in web_links:
            url = url.strip()
            if not url:
                continue  # Skip empty lines
            try:
                documents = extract_text_from_url(url)  # Returns a list of Document objects

                for doc in documents:  
                    processed_documents.append(doc)  # Append each document separately

            except Exception as e:
                placeholder.error(f"Failed to process {url}: {str(e)}")

        time.sleep(5)
        placeholder.empty()
        
        return processed_documents  # Return all processed documents
        
    except Exception as e:
        placeholder.error(f"An error occurred: {str(e)}")
        return None


        
def create_vector_index(docs, embeddings):
    return FAISS.from_documents(docs, embeddings)


