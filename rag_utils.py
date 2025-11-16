from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.text import TextLoader

OLLAMA_BASE_URL = "http://10.50.1.101:11434"
print('start')

def load_docs():
    print('done')
    customer = TextLoader("C:/Users/diloyanomon/Documents/customer issue.txt")
    customer = customer.load()
    fibre = TextLoader("C:/Users/diloyanomon/Documents/fibre issues.txt")
    fibre = fibre.load()
    documents=customer + fibre
    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embedding = OllamaEmbeddings(model="nomic-embed-text")  
    vectordb = Chroma.from_documents(chunks, embedding,collection_name="chroma", persist_directory="rag_db")
    return vectordb.as_retriever(k=3)

