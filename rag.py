import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
PDF_PATH = "traffic_manual.pdf"
DB_DIR = "./VECTOR_DB"

# FORCE OFFLINE MODE (Enable after first successful download)
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"

def setup_rag():
    loader = PyMuPDFLoader(PDF_PATH)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1,        # Minimal size to keep lines separate
        chunk_overlap=0,     # No overlapping between rules
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'local_files_only': False} 
    )
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vector_db

def retrieve_rule(sign_name):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'local_files_only': True}
    )
    
    # Load the existing database
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    
    docs = vector_db.similarity_search(sign_name, k=5)
    
    if docs:
        for doc in docs:
            line = doc.page_content.strip()
            
        
            clean_name = sign_name.replace("_", " ")
            
            if clean_name.lower() in line.lower():
                
                if ":" in line:
                    return line.split(":", 1)[1].strip()
                return line
        
       
        return docs[0].page_content
        
    return "No specific rule found."

if __name__ == "__main__":
    
    if not os.path.exists(DB_DIR):
        print("🚀 Building line-by-line database...")
        setup_rag()
        print("✅ Database ready!")
    else:
        print("ℹ️ Database exists.")