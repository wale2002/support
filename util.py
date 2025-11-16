# # # # from langchain_ollama import OllamaEmbeddings
# # # # from langchain_chroma import Chroma
# # # # from langchain_text_splitters import CharacterTextSplitter
# # # # from langchain_community.document_loaders import TextLoader
# # # # from langchain_core.documents import Document
# # # # import os


# # # # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust if your Ollama is remote

# # # # def get_retriever(customer_file, fibre_file, persist_dir="rag_db"):
# # # #     """
# # # #     Loads or creates the vector DB retriever.
# # # #     Run once to build; subsequent calls load from disk.
# # # #     """
# # # #     print("Loading/creating vector DB...")
    
# # # #     # Load documents
# # # #   customer_loader = TextLoader(customer_file, encoding="utf-8")
# # # # customer_docs = customer_loader.load()
# # # # fibre_loader = TextLoader(fibre_file, encoding="utf-8")
# # # # fibre_docs = fibre_loader.load()
    
# # # #     # Add metadata to distinguish categories
# # # #     for doc in customer_docs:
# # # #         doc.metadata["category"] = "customer"
# # # #     for doc in fibre_docs:
# # # #         doc.metadata["category"] = "fibre"
    
# # # #     all_documents = customer_docs + fibre_docs
    
# # # #     # Split into chunks
# # # #     splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
# # # #     chunks = splitter.split_documents(all_documents)
    
# # # #     # Embed and store (or load if exists)
# # # #     embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    
# # # #     if os.path.exists(persist_dir):
# # # #         vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
# # # #     else:
# # # #         vectordb = Chroma.from_documents(chunks, embedding, collection_name="complaints", persist_directory=persist_dir)
# # # #         vectordb.persist()
    
# # # #     return vectordb.as_retriever(search_kwargs={"k": 3})  # Top 3 similar chunks

# # # from langchain_ollama import OllamaEmbeddings
# # # from langchain_chroma import Chroma
# # # from langchain_text_splitters import CharacterTextSplitter
# # # from langchain_community.document_loaders import TextLoader
# # # from langchain_core.documents import Document
# # # import os

# # # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust if your Ollama is remote

# # # def get_retriever(customer_file, fibre_file, persist_dir="rag_db"):
# # #     """
# # #     Loads or creates the vector DB retriever.
# # #     Run once to build; subsequent calls load from disk.
# # #     """
# # #     print("Loading/creating vector DB...")
    
# # #     # Load documents
# # #     customer_loader = TextLoader(customer_file, encoding="utf-8")
# # #     customer_docs = customer_loader.load()
# # #     fibre_loader = TextLoader(fibre_file, encoding="utf-8")
# # #     fibre_docs = fibre_loader.load()
    
# # #     # Add metadata to distinguish categories
# # #     for doc in customer_docs:
# # #         doc.metadata["category"] = "customer"
# # #     for doc in fibre_docs:
# # #         doc.metadata["category"] = "fibre"
    
# # #     all_documents = customer_docs + fibre_docs
    
# # #     # Split into chunks
# # #     splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
# # #     chunks = splitter.split_documents(all_documents)
    
# # #     # Embed and store (or load if exists)
# # #     embedding = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    
# # #     if os.path.exists(persist_dir):
# # #         vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
# # #     else:
# # #         vectordb = Chroma.from_documents(chunks, embedding, collection_name="complaints", persist_directory=persist_dir)
# # #         vectordb.persist()
    
# # #     return vectordb.as_retriever(search_kwargs={"k": 3})  # Top 3 similar chunks

# # # # If running this file directly, it won't do anything useful—use it via main.py
# # # if __name__ == "__main__":
# # #     print("Util module loaded. Run main.py to use the retriever.")


# # from langchain_community.vectorstores import FAISS
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# # from langchain_community.document_loaders import TextLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # import os

# # def get_retriever(customer_file: str, fibre_file: str):
# #     """
# #     Load and index complaint files into a FAISS vector store.
# #     Returns a retriever for similarity search.
# #     """
# #     if not os.path.exists(customer_file) or not os.path.exists(fibre_file):
# #         raise FileNotFoundError(f"Data files missing: {customer_file}, {fibre_file}")

# #     # Embeddings model (use HuggingFace for local/offline)
# #     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# #     # Load and split customer docs
# #     customer_loader = TextLoader(customer_file)
# #     customer_docs = customer_loader.load()
# #     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# #     customer_chunks = splitter.split_documents(customer_docs)
# #     for doc in customer_chunks:
# #         doc.metadata["category"] = "customer"

# #     # Load and split fibre docs
# #     fibre_loader = TextLoader(fibre_file)
# #     fibre_docs = fibre_loader.load()
# #     fibre_chunks = splitter.split_documents(fibre_docs)
# #     for doc in fibre_chunks:
# #         doc.metadata["category"] = "fibre"

# #     # Combine all docs
# #     all_docs = customer_chunks + fibre_chunks

# #     # Create vector store (loads from disk if exists, else builds)
# #     vector_store_path = "faiss_index"
# #     if os.path.exists(vector_store_path):
# #         vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
# #         vector_store.add_documents(all_docs)  # Update if new docs
# #     else:
# #         vector_store = FAISS.from_documents(all_docs, embeddings)
# #         vector_store.save_local(vector_store_path)

# #     return vector_store.as_retriever(search_kwargs={"k": 4})  # Top 4 similar examples


# from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for LangChain 0.2+
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os

# def get_retriever(customer_file: str, fibre_file: str):
#     """
#     Load and index complaint files into a FAISS vector store.
#     Returns a retriever for similarity search.
#     """
#     if not os.path.exists(customer_file) or not os.path.exists(fibre_file):
#         raise FileNotFoundError(f"Data files missing: {customer_file}, {fibre_file}")

#     # Embeddings model (use HuggingFace for local/offline)
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # Load and split customer docs (with UTF-8 encoding to fix UnicodeDecodeError)
#     customer_loader = TextLoader(customer_file, encoding='utf-8')
#     customer_docs = customer_loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     customer_chunks = splitter.split_documents(customer_docs)
#     for doc in customer_chunks:
#         doc.metadata["category"] = "customer"

#     # Load and split fibre docs (with UTF-8 encoding)
#     fibre_loader = TextLoader(fibre_file, encoding='utf-8')
#     fibre_docs = fibre_loader.load()
#     fibre_chunks = splitter.split_documents(fibre_docs)
#     for doc in fibre_chunks:
#         doc.metadata["category"] = "fibre"

#     # Combine all docs
#     all_docs = customer_chunks + fibre_chunks

#     # Create vector store (loads from disk if exists, else builds)
#     vector_store_path = "faiss_index"
#     if os.path.exists(vector_store_path):
#         vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
#         # Optional: To avoid duplicates on update, you could merge/rebuild, but for simplicity, just add
#         # If duplicates become an issue, consider rebuilding: vector_store = FAISS.from_documents(all_docs, embeddings)
#         vector_store.add_documents(all_docs)  # Update if new docs
#     else:
#         vector_store = FAISS.from_documents(all_docs, embeddings)
#         vector_store.save_local(vector_store_path)

#     return vector_store.as_retriever(search_kwargs={"k": 4})  # Top 4 similar examples


from langchain_huggingface import HuggingFaceEmbeddings  # Updated import for LangChain 0.2+
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path

def get_retriever(customer_file: str, fibre_file: str):
    """
    Load and index complaint files into a FAISS vector store.
    Returns a retriever for similarity search.
    """
    if not os.path.exists(customer_file) or not os.path.exists(fibre_file):
        raise FileNotFoundError(f"Data files missing: {customer_file}, {fibre_file}")

    # Embeddings model (use HuggingFace for local/offline)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load and split customer docs (with UTF-8 encoding to fix UnicodeDecodeError)
    customer_loader = TextLoader(customer_file, encoding='utf-8')
    customer_docs = customer_loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    customer_chunks = splitter.split_documents(customer_docs)
    for doc in customer_chunks:
        doc.metadata["category"] = "customer"

    # Load and split fibre docs (with UTF-8 encoding)
    fibre_loader = TextLoader(fibre_file, encoding='utf-8')
    fibre_docs = fibre_loader.load()
    fibre_chunks = splitter.split_documents(fibre_docs)
    for doc in fibre_chunks:
        doc.metadata["category"] = "fibre"

    # Combine all docs
    all_docs = customer_chunks + fibre_chunks

    # Create vector store (loads from disk if exists and valid, else builds)
    vector_store_path = Path("faiss_index")
    vector_store_path.mkdir(exist_ok=True)  # Ensure dir exists
    
    index_file = vector_store_path / "index.faiss"  # FAISS default index name
    
    # Check for valid existing index (non-empty file)
    if index_file.exists() and index_file.stat().st_size > 0:
        try:
            vector_store = FAISS.load_local(
                str(vector_store_path), embeddings, allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index successfully.")
            # Add new docs to update (avoids full rebuild for incremental changes)
            if all_docs:  # Only add if there are docs
                vector_store.add_documents(all_docs)
                print(f"Added {len(all_docs)} new document chunks to the index.")
        except Exception as e:  # Catch load errors (e.g., corruption)
            if "read_index" in str(e) or "faiss" in str(e).lower():
                print(f"Failed to load FAISS index (likely corrupted): {e}. Deleting and rebuilding...")
                index_file.unlink()  # Remove bad file
                # Fall through to full rebuild
            else:
                raise  # Re-raise non-FAISS errors
    else:
        if index_file.exists():
            print("Detected empty or invalid FAISS index file. Deleting and rebuilding...")
            index_file.unlink()
        print("No valid FAISS index found. Building new one...")

    # Build or rebuild if needed
    if not index_file.exists() or index_file.stat().st_size == 0:
        vector_store = FAISS.from_documents(all_docs, embeddings)
        vector_store.save_local(str(vector_store_path))
        print(f"New FAISS index created and saved. Size: {index_file.stat().st_size} bytes")

    return vector_store.as_retriever(search_kwargs={"k": 4})  # Top 4 similar examples