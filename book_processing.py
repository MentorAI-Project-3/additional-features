from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import glob
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embeddings for RAG
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_books(book_path="books"):
    documents = []
    print(f"📁 Checking for books at: {book_path}")
    
    # Check if book_path is a file or directory
    if os.path.isfile(book_path):
        files = [book_path]
        print(f"📄 Found single file: {book_path}")
    elif os.path.isdir(book_path):
        files = glob.glob(os.path.join(book_path, "*.txt")) + glob.glob(os.path.join(book_path, "*.pdf"))
        print(f"📂 Found {len(files)} files in directory: {files}")
    else:
        print(f"❌ Path '{book_path}' does not exist.")
        return documents

    for file in files:
        try:
            if file.endswith(".txt"):
                print(f"📜 Loading text file: {file}")
                loader = TextLoader(file)
                documents.extend(loader.load())
            elif file.endswith(".pdf"):
                print(f"📕 Loading PDF file: {file}")
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Error loading {file}: {str(e)}")
    
    if not documents:
        print("⚠️ No documents loaded successfully.")
        return documents
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"📑 Processed {len(chunks)} document chunks.")
    return chunks


def create_vector_store(book_path):
    chunks = load_books(book_path)
    print("🛠️ Creating new FAISS index.")
    if not chunks:
        print("⚠️ No document chunks provided. Returning None.")
        return None
    
    # Remove existing FAISS index if it exists
    if os.path.exists("faiss_index"):
        print("🗑️ Removing existing FAISS index.")
        shutil.rmtree("faiss_index")
    
    # Create new FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("💾 FAISS index saved.")
    return vector_store
create_vector_store("C:\\Users\\Windows 11\\Desktop\\UN\\Project 3\\code\\project-3/Diagram Generation/Books/")