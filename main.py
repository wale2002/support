# # # # # from langchain_ollama import ChatOllama  # Optional, if switching back to Ollama LLM
# # # # # from langchain_core.prompts import ChatPromptTemplate
# # # # # from langchain_openai import ChatOpenAI
# # # # # from util import get_retriever
# # # # # from dotenv import load_dotenv
# # # # # import os

# # # # # # Load environment variables
# # # # # load_dotenv()

# # # # # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust as needed

# # # # # # Paths to your data files (adjust to your setup)
# # # # # CUSTOMER_FILE = "data/customer_file.txt"
# # # # # FIBRE_FILE = "data/fibre_file.txt"

# # # # # # Load retrievers (this will build/load the DB)
# # # # # retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # # # # # LLM setup (using OpenAI; uncomment below for Ollama)
# # # # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # # # # model = ChatOpenAI(
# # # # #     model="gpt-4o",
# # # # #     temperature=0,
# # # # #     max_tokens=None,
# # # # #     timeout=None,
# # # # #     max_retries=2,
# # # # #     api_key=openai_api_key
# # # # # )
# # # # # # model = ChatOllama(model="qwen2.5:7b", base_url=OLLAMA_BASE_URL)  # If using Ollama LLM

# # # # # template = """
# # # # # You are a support classification agent.

# # # # # You will be given examples of either customer complaints or fibre complaints.
# # # # # Classify the user's complaint as one of the following: "customer" or "fibre".

# # # # # Examples:
# # # # # {examples}

# # # # # User Complaint:
# # # # # {complaint}

# # # # # Respond with only one word: "customer" or "fibre".
# # # # # """

# # # # # prompt = ChatPromptTemplate.from_template(template)
# # # # # chain = prompt | model

# # # # # while True:
# # # # #     print("\n-------------------------------")
# # # # #     complaint = input("Enter complaint (q to quit): ")
# # # # #     if complaint.lower() == "q":
# # # # #         break

# # # # #     # Retrieve similar examples
# # # # #     examples = retriever.invoke(complaint)
# # # # #     all_examples = ""
# # # # #     for doc in examples:
# # # # #         category = doc.metadata.get("category", "unknown")
# # # # #         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

# # # # #     result = chain.invoke({"examples": all_examples, "complaint": complaint})
# # # # #     print(f"\nClassification: {result.content.strip()}")

# # # # from langchain_ollama import ChatOllama  # Optional fallback
# # # # from langchain_core.prompts import ChatPromptTemplate
# # # # # from langchain_openai import ChatOpenAI  # Commented out
# # # # from util import get_retriever
# # # # from dotenv import load_dotenv
# # # # import os
# # # # import requests  # For OkeyMeta API calls
# # # # from langchain_core.language_models import LLM  # For custom OkeyMeta wrapper
# # # # from langchain_core.messages import AIMessage  # If needed for chat compat
# # # # from langchain_core.outputs import Generation, LLMResult  # For LLM output

# # # # # Load environment variables
# # # # load_dotenv()

# # # # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust as needed

# # # # # Paths to your data files (adjust if needed)
# # # # CUSTOMER_FILE = "data/customer_file.txt"
# # # # FIBRE_FILE = "data/fibre_file.txt"

# # # # # Load retrievers (this will build/load the DB)
# # # # retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # # # # Custom OkeyMeta LLM Wrapper
# # # # class OkeyMetaLLM(LLM):
# # # #     def __init__(self, token: str):
# # # #         super().__init__()
# # # #         self.token = token
# # # #         self.base_url = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

# # # #     def _call(
# # # #         self,
# # # #         prompt: str,
# # # #         stop: list[str] | None = None,
# # # #         run_manager=None,
# # # #         **kwargs,
# # # #     ) -> str:
# # # #         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
# # # #         headers = {"Authorization": f"Bearer {self.token}"}
        
# # # #         try:
# # # #             response = requests.get(full_url, headers=headers, timeout=30)
# # # #             response.raise_for_status()
            
# # # #             data = response.json()
# # # #             # Adjust key based on actual response (e.g., 'output', 'choices[0].text', or 'response')
# # # #             output = data.get('output') or data.get('response', '')
# # # #             if not output and 'choices' in data:
# # # #                 output = data['choices'][0].get('text', '') if data['choices'] else ''
# # # #             return output.strip()
            
# # # #         except requests.exceptions.RequestException as e:
# # # #             raise ValueError(f"OkeyMeta API error: {e}")

# # # #     @property
# # # #     def _llm_type(self) -> str:
# # # #         return "okeymeta"

# # # # # LLM Setup: Use OkeyMeta (fallback to Ollama if needed)
# # # # okey_token = os.getenv("OKEYMETA_TOKEN")
# # # # if okey_token:
# # # #     model = OkeyMetaLLM(token=okey_token)
# # # # else:
# # # #     print("No OKEYMETA_TOKEN found—falling back to Ollama.")
# # # #     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # # # # Original commented OpenAI (for reference)
# # # # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # # # # model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

# # # # template = """
# # # # You are a support classification agent.

# # # # You will be given examples of either customer complaints or fibre complaints.
# # # # Classify the user's complaint as one of the following: "customer" or "fibre".

# # # # Examples:
# # # # {examples}

# # # # User Complaint:
# # # # {complaint}

# # # # Respond with only one word: "customer" or "fibre".
# # # # """

# # # # prompt = ChatPromptTemplate.from_template(template)
# # # # chain = prompt | model

# # # # while True:
# # # #     print("\n-------------------------------")
# # # #     complaint = input("Enter complaint (q to quit): ")
# # # #     if complaint.lower() == "q":
# # # #         break

# # # #     # Retrieve similar examples
# # # #     examples = retriever.invoke(complaint)
# # # #     all_examples = ""
# # # #     for doc in examples:
# # # #         category = doc.metadata.get("category", "unknown")
# # # #         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

# # # #     result = chain.invoke({"examples": all_examples, "complaint": complaint})
# # # #     print(f"\nClassification: {result.content if hasattr(result, 'content') else result.strip()}")


# # # from langchain_ollama import ChatOllama  # Optional fallback
# # # from langchain_core.prompts import ChatPromptTemplate
# # # # from langchain_openai import ChatOpenAI  # Commented out
# # # from util import get_retriever
# # # from dotenv import load_dotenv
# # # import os
# # # import requests  # For OkeyMeta API calls
# # # from langchain_core.language_models import LLM  # For custom OkeyMeta wrapper
# # # from langchain_core.outputs import Generation, LLMResult  # For LLM output

# # # # Load environment variables
# # # load_dotenv()

# # # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust as needed

# # # # Paths to your data files (adjust if needed)
# # # CUSTOMER_FILE = "data/customer_file.txt"
# # # FIBRE_FILE = "data/fibre_file.txt"

# # # # Load retrievers (this will build/load the DB)
# # # retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # # # Custom OkeyMeta LLM Wrapper (Fixed: Added token as Pydantic field)
# # # class OkeyMetaLLM(LLM):
# # #     token: str  # Declare as class field for Pydantic validation

# # #     def __init__(self, token: str):
# # #         super().__init__(token=token)  # Pass to base for validation
# # #         self.base_url = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

# # #     def _call(
# # #         self,
# # #         prompt: str,
# # #         stop: list[str] | None = None,
# # #         run_manager=None,
# # #         **kwargs,
# # #     ) -> str:
# # #         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
# # #         headers = {"Authorization": f"Bearer {self.token}"}
        
# # #         try:
# # #             response = requests.get(full_url, headers=headers, timeout=30)
# # #             response.raise_for_status()
            
# # #             data = response.json()
# # #             # Adjust key based on actual response (e.g., 'output', 'choices[0].text', or 'response')
# # #             output = data.get('output') or data.get('response', '')
# # #             if not output and 'choices' in data:
# # #                 output = data['choices'][0].get('text', '') if data['choices'] else ''
# # #             return output.strip()
            
# # #         except requests.exceptions.RequestException as e:
# # #             raise ValueError(f"OkeyMeta API error: {e}")

# # #     @property
# # #     def _llm_type(self) -> str:
# # #         return "okeymeta"

# # #     # Required for LangChain compatibility (returns mock generations)
# # #     def _generate(
# # #         self, prompts, stop=None, run_manager=None, **kwargs
# # #     ) -> LLMResult:
# # #         generations = []
# # #         for prompt in prompts:
# # #             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
# # #             generations.append([Generation(text=text)])
# # #         return LLMResult(generations=generations)

# # # # LLM Setup: Use OkeyMeta (fallback to Ollama if needed)
# # # okey_token = os.getenv("OKEYMETA_TOKEN")
# # # if okey_token:
# # #     model = OkeyMetaLLM(token=okey_token)
# # # else:
# # #     print("No OKEYMETA_TOKEN found—falling back to Ollama.")
# # #     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # # # Original commented OpenAI (for reference)
# # # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # # # model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

# # # template = """
# # # You are a support classification agent.

# # # You will be given examples of either customer complaints or fibre complaints.
# # # Classify the user's complaint as one of the following: "customer" or "fibre".

# # # Examples:
# # # {examples}

# # # User Complaint:
# # # {complaint}

# # # Respond with only one word: "customer" or "fibre".
# # # """

# # # prompt = ChatPromptTemplate.from_template(template)
# # # chain = prompt | model

# # # while True:
# # #     print("\n-------------------------------")
# # #     complaint = input("Enter complaint (q to quit): ")
# # #     if complaint.lower() == "q":
# # #         break

# # #     # Retrieve similar examples
# # #     examples = retriever.invoke(complaint)
# # #     all_examples = ""
# # #     for doc in examples:
# # #         category = doc.metadata.get("category", "unknown")
# # #         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

# # #     result = chain.invoke({"examples": all_examples, "complaint": complaint})
# # #     print(f"\nClassification: {result.content if hasattr(result, 'content') else result.strip()}")


# # from langchain_ollama import ChatOllama  # Optional fallback
# # from langchain_core.prompts import ChatPromptTemplate
# # # from langchain_openai import ChatOpenAI  # Commented out
# # from util import get_retriever
# # from dotenv import load_dotenv
# # import os
# # import requests  # For OkeyMeta API calls
# # from langchain_core.language_models import LLM  # For custom OkeyMeta wrapper
# # from langchain_core.outputs import Generation, LLMResult  # For LLM output

# # # Load environment variables
# # load_dotenv()

# # OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust as needed

# # # Paths to your data files (adjust if needed)
# # CUSTOMER_FILE = "data/customer_file.txt"
# # FIBRE_FILE = "data/fibre_file.txt"

# # # Load retrievers (this will build/load the DB)
# # retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # # Custom OkeyMeta LLM Wrapper (Fixed: Declared all fields with types/defaults)
# # class OkeyMetaLLM(LLM):
# #     token: str  # Required field
# #     base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"  # Default field

# #     def __init__(self, token: str):
# #         super().__init__(token=token)  # Validates fields via Pydantic

# #     def _call(
# #         self,
# #         prompt: str,
# #         stop: list[str] | None = None,
# #         run_manager=None,
# #         **kwargs,
# #     ) -> str:
# #         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
# #         headers = {"Authorization": f"Bearer {self.token}"}
        
# #         try:
# #             response = requests.get(full_url, headers=headers, timeout=30)
# #             response.raise_for_status()
            
# #             data = response.json()
# #             # Adjust key based on actual response (e.g., 'output', 'choices[0].text', or 'response')
# #             output = data.get('output') or data.get('response', '')
# #             if not output and 'choices' in data:
# #                 output = data['choices'][0].get('text', '') if data['choices'] else ''
# #             return output.strip()
            
# #         except requests.exceptions.RequestException as e:
# #             raise ValueError(f"OkeyMeta API error: {e}")

# #     @property
# #     def _llm_type(self) -> str:
# #         return "okeymeta"

# #     # Required for LangChain compatibility (returns mock generations)
# #     def _generate(
# #         self, prompts, stop=None, run_manager=None, **kwargs
# #     ) -> LLMResult:
# #         generations = []
# #         for prompt in prompts:
# #             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
# #             generations.append([Generation(text=text)])
# #         return LLMResult(generations=generations)

# # # LLM Setup: Use OkeyMeta (fallback to Ollama if needed)
# # okey_token = os.getenv("OKEYMETA_TOKEN")
# # if okey_token:
# #     model = OkeyMetaLLM(token=okey_token)
# # else:
# #     print("No OKEYMETA_TOKEN found—falling back to Ollama.")
# #     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # # Original commented OpenAI (for reference)
# # # openai_api_key = os.getenv("OPENAI_API_KEY")
# # # model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)

# # template = """
# # You are a support classification agent.

# # You will be given examples of either customer complaints or fibre complaints.
# # Classify the user's complaint as one of the following: "customer" or "fibre".

# # Examples:
# # {examples}

# # User Complaint:
# # {complaint}

# # Respond with only one word: "customer" or "fibre".
# # """

# # prompt = ChatPromptTemplate.from_template(template)
# # chain = prompt | model

# # while True:
# #     print("\n-------------------------------")
# #     complaint = input("Enter complaint (q to quit): ")
# #     if complaint.lower() == "q":
# #         break

# #     # Retrieve similar examples
# #     examples = retriever.invoke(complaint)
# #     all_examples = ""
# #     for doc in examples:
# #         category = doc.metadata.get("category", "unknown")
# #         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

# #     result = chain.invoke({"examples": all_examples, "complaint": complaint})
# #     print(f"\nClassification: {result.content if hasattr(result, 'content') else result.strip()}")


# from langchain_ollama import ChatOllama  # Optional fallback
# from langchain_core.prompts import ChatPromptTemplate
# # from langchain_openai import ChatOpenAI  # Commented out
# from util import get_retriever
# from dotenv import load_dotenv
# import os
# import requests  # For OkeyMeta API calls
# from langchain_core.language_models import LLM  # For custom OkeyMeta wrapper
# from langchain_core.outputs import Generation, LLMResult  # For LLM output
# import re  # For parsing classification response

# # Load environment variables
# load_dotenv()

# OLLAMA_BASE_URL = "http://localhost:11434"  # Adjust as needed

# # Paths to your data files (adjust if needed)
# CUSTOMER_FILE = "data/customer_file.txt"
# FIBRE_FILE = "data/fibre_file.txt"

# # Load retrievers (this will build/load the DB)
# retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # Custom OkeyMeta LLM Wrapper (Fixed: Declared all fields with types/defaults)
# class OkeyMetaLLM(LLM):
#     token: str  # Required field
#     base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"  # Default field

#     def __init__(self, token: str):
#         super().__init__(token=token)  # Validates fields via Pydantic

#     def _call(
#         self,
#         prompt: str,
#         stop: list[str] | None = None,
#         run_manager=None,
#         **kwargs,
#     ) -> str:
#         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
#         headers = {"Authorization": f"Bearer {self.token}"}
        
#         try:
#             response = requests.get(full_url, headers=headers, timeout=30)
#             response.raise_for_status()
            
#             data = response.json()
#             # Adjust key based on actual response (e.g., 'output', 'choices[0].text', or 'response')
#             output = data.get('output') or data.get('response', '')
#             if not output and 'choices' in data:
#                 output = data['choices'][0].get('text', '') if data['choices'] else ''
#             return output.strip()
            
#         except requests.exceptions.RequestException as e:
#             raise ValueError(f"OkeyMeta API error: {e}")

#     @property
#     def _llm_type(self) -> str:
#         return "okeymeta"

#     # Required for LangChain compatibility (returns mock generations)
#     def _generate(
#         self, prompts, stop=None, run_manager=None, **kwargs
#     ) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
#             generations.append([Generation(text=text)])
#         return LLMResult(generations=generations)

# # LLM Setup: Use OkeyMeta (fallback to Ollama if needed)
# okey_token = os.getenv("OKEYMETA_TOKEN")
# if okey_token:
#     model = OkeyMetaLLM(token=okey_token)
# else:
#     print("No OKEYMETA_TOKEN found—falling back to Ollama.")
#     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # Function to extract classification from potentially verbose response
# def extract_classification(response_text: str) -> str:
#     # Look for "customer" or "fibre" in the response (case-insensitive, last match)
#     match = re.search(r'\b(customer|fibre)\b', response_text.lower())
#     return match.group(1) if match else "unknown"

# # Classification prompt (strengthened to enforce one-word response)
# classification_template = """
# You are a support classification agent. Classify the complaint STRICTLY as "customer" or "fibre" ONLY. Do NOT add any other text, explanations, or acknowledgments.

# Examples:
# {examples}

# User Complaint: {complaint}

# Classification:"""

# classification_prompt = ChatPromptTemplate.from_template(classification_template)
# classification_chain = classification_prompt | model

# # Solution generation prompt
# solution_template = """
# You are a helpful support agent. Based on the classified complaint type and the user's issue, provide a concise, empathetic solution or next steps. Keep it under 150 words.

# Classification: {classification}
# User Complaint: {complaint}

# Response:"""

# solution_prompt = ChatPromptTemplate.from_template(solution_template)
# solution_chain = solution_prompt | model

# while True:
#     print("\n-------------------------------")
#     complaint = input("Enter complaint (q to quit): ")
#     if complaint.lower() == "q":
#         break

#     # Retrieve similar examples for classification
#     examples = retriever.invoke(complaint)
#     all_examples = ""
#     for doc in examples:
#         category = doc.metadata.get("category", "unknown")
#         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

#     # Get classification
#     class_result = classification_chain.invoke({"examples": all_examples, "complaint": complaint})
#     class_text = class_result.content if hasattr(class_result, 'content') else class_result
#     classification = extract_classification(class_text)
    
#     print(f"\nClassification: {classification}")
    
#     if classification in ["customer", "fibre"]:
#         # Generate solution
#         sol_result = solution_chain.invoke({"classification": classification, "complaint": complaint})
#         sol_text = sol_result.content if hasattr(sol_result, 'content') else sol_result
#         print(f"\nSuggested Solution:\n{sol_text.strip()}")
#     else:
#         print("\nUnable to classify. Please try again.")



# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from util import get_retriever
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List
from datetime import datetime
import os
import requests
import logging
import re
from functools import lru_cache
import torch

# ------------------- MEMORY OPTIMIZATIONS -------------------
torch.set_num_threads(1)  # Render free has only ~0.5 vCPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Stops huggingface warnings + memory bloat

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CUSTOMER_FILE = "data/customer_file.txt"
FIBRE_FILE = "data/fibre_file.txt"

# MongoDB (optional – works without one too)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = mongo_client.get_database("complaints_db")
    complaints_collection = db["complaints"]
    logger.info("MongoDB connected")
except Exception as e:
    logger.warning(f"MongoDB not available: {e}")
    complaints_collection = None

# Hardcoded token (remove in real prod – use .env + Render env vars)
os.environ["OKEYMETA_TOKEN"] = "okeyai_65b110107b482d2b58ba9192a05457565d278862a785c21208910defe653020e"

# ------------------- CUSTOM LLM (unchanged) -------------------
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult

class OkeyMetaLLM(LLM):
    token: str
    base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

    def __init__(self, token: str):
        super().__init__(token=token)

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(full_url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            output = data.get('output') or data.get('response', '') or data.get('text', '')
            return output.strip()
        except Exception as e:
            raise ValueError(f"OkeyMeta API error: {e}")

    @property
    def _llm_type(self) -> str:
        return "okeymeta"

    def _generate(self, prompts, **kwargs) -> LLMResult:
        generations = []
        for p in prompts:
            text = self._call(p, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

# LLM instance
model = OkeyMetaLLM(token=os.getenv("OKEYMETA_TOKEN"))

# ------------------- LAZY RETRIEVER (this is the magic) -------------------
@lru_cache(maxsize=1)
def get_cached_retriever():
    logger.info("Loading embeddings + FAISS index for the first time...")
    return get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# ------------------- Chains -------------------
classification_prompt = ChatPromptTemplate.from_template("""
You are a support classification agent. Classify STRICTLY as "customer" or "fibre" ONLY.

Examples:
{examples}

Complaint: {complaint}

Classification:""")

solution_prompt = ChatPromptTemplate.from_template("""
You are a helpful agent. Give a short empathetic solution (<120 words).

Type: {classification}
Complaint: {complaint}

Response:""")

classification_chain = classification_prompt | model
solution_chain = solution_prompt | model

def extract_classification(text: str) -> str:
    match = re.search(r'\b(customer|fibre)\b', text.lower())
    return match.group(1) if match else "unknown"

# ------------------- FastAPI App -------------------
app = FastAPI(title="AI Complaint Classifier")

class ComplaintRequest(BaseModel):
    complaint: str

class ClassificationResponse(BaseModel):
    classification: str
    solution: str | None = None

class ComplaintItem(BaseModel):
    id: str
    complaint: str
    classification: str
    solution: str
    timestamp: str

class ComplaintsResponse(BaseModel):
    complaints: List[ComplaintItem]

@app.post("/classify", response_model=ClassificationResponse)
async def classify_complaint(request: ComplaintRequest):
    if not request.complaint.strip():
        raise HTTPException(status_code=400, detail="Empty complaint")

    retriever = get_cached_retriever()  # ← Loads only on first request

    examples = retriever.invoke(request.complaint)
    example_text = "\n".join(
        f"{doc.metadata.get('category','?').capitalize()}: {doc.page_content}"
        for doc in examples
    )

    # Classify
    result = classification_chain.invoke({"examples": example_text, "complaint": request.complaint})
    text = result.content if hasattr(result, "content") else str(result)
    classification = extract_classification(text)

    if classification not in ["customer", "fibre"]:
        raise HTTPException(status_code=400, detail="Could not classify")

    # Solution
    sol_result = solution_chain.invoke({"classification": classification, "complaint": request.complaint})
    solution = (sol_result.content if hasattr(sol_result, "content") else str(sol_result)).strip()

    # Optional: store in MongoDB
    if complaints_collection:
        complaints_collection.insert_one({
            "complaint": request.complaint,
            "classification": classification,
            "solution": solution,
            "timestamp": datetime.now().isoformat()
        })

    return ClassificationResponse(classification=classification, solution=solution)

@app.get("/getComplaints", response_model=ComplaintsResponse)
async def get_complaints(limit: int = 20, skip: int = 0):
    if not complaints_collection:
        return ComplaintsResponse(complaints=[])

    cursor = complaints_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
    items = [
        ComplaintItem(
            id=str(doc["_id"]),
            complaint=doc["complaint"],
            classification=doc["classification"],
            solution=doc["solution"],
            timestamp=doc["timestamp"]
        )
        for doc in cursor
    ]
    return ComplaintsResponse(complaints=items)

@app.get("/health")
def health():
    return {"status": "healthy", "memory_optimized": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))