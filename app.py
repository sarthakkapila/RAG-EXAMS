import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

# Set up directories
DATA_DIR = Path(__file__).resolve().parent.joinpath('data')

# Streamlit setup
st.set_page_config(page_title="Document Search")
st.title("Document Search")

# Load documents
@st.cache_data
def load_documents():
    loader = PyPDFLoader(DATA_DIR.as_posix())
    documents = loader.load_and_split(chunk_size=1000, chunk_overlap=0)
    return documents

# Split and embed documents
@st.cache_data
def embed_documents(documents, gemini_api_key):
    embeddings = GoogleGenerativeAIEmbeddings(gemini_api_key)
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=DATA_DIR.as_posix())
    vectordb.persist()
    return vectordb.as_retriever()

# Input fields
gemini_api_key = st.text_input("Gemini API key", type="password")
source_docs = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True)

# Process documents
if gemini_api_key and source_docs:
    with st.spinner("Loading and embedding documents..."):
        documents = load_documents()
        retriever = embed_documents(documents, gemini_api_key)

    query = st.text_input("Search documents")
    if query:
        results = retriever.get_relevant_documents(query)
        for result in results:
            st.write(result.page_content)

# if __name__ == "__main__":
#     main()