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


# main.py — FINAL VERSION FOR RENDER FREE TIER (lazy + tiny memory)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from util import get_retriever
from dotenv import load_dotenv
from datetime import datetime
import os
import torch
from functools import lru_cache
import logging

# ------------------- MEMORY FIXES -------------------
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CUSTOMER_FILE = "data/customer_file.txt"
FIBRE_FILE = "data/fibre_file.txt"

# ------------------- LAZY RETRIEVER (this saves your life) -------------------
@lru_cache(maxsize=1)
def get_cached_retriever():
    logger.info("First request → loading embeddings + FAISS (15–20s cold start)...")
    return get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# ------------------- LLM (your OkeyMeta) -------------------
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
import requests

class OkeyMetaLLM(LLM):
    token: str = os.getenv("OKEYMETA_TOKEN", "okeyai_65b110107b482d2b58ba9192a05457565d278862a785c21208910defe653020e")
    base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

    def _call(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}?input={requests.utils.quote proportionality(prompt)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        r = requests.get(url, headers=headers, timeout=60)
        data = r.json()
        return (data.get("output") or data.get("response") or "").strip()

    @property
    def _llm_type(self) -> str: return "okeymeta"
    def _generate(self, prompts, **kwargs) -> LLMResult:
        return LLMResult(generations=[[Generation(text=self._call(p))] for p in prompts])

model = OkeyMetaLLM()

# ------------------- Chains -------------------
classify_tpl = ChatPromptTemplate.from_template("Classify as customer or fibre ONLY.\nExamples:\n{examples}\nComplaint: {complaint}\nAnswer:")
solution_tpl = ChatPromptTemplate.from_template("Short empathetic solution (<100 words).\nType: {classification}\nComplaint: {complaint}\nResponse:")

classify_chain = classify_tpl | model
solution_chain = solution_tpl | model

import re
def extract(text): 
    m = re.search(r"\b(customer|fibre)\b", text.lower())
    return m.group(1) if m else "unknown"

# ------------------- FastAPI -------------------
app = FastAPI()

class Req(BaseModel):
    complaint: str

class Resp(BaseModel):
    classification: str
    solution: str

@app.post("/classify", response_model=Resp)
async def classify(req: Req):
    if not req.complaint.strip():
        raise HTTPException(400, "Empty complaint")

    retriever = get_cached_retriever()  # ← lazy load here
    docs = retriever.invoke(req.complaint)
    examples = "\n".join(f"{d.metadata.get('category','?').capitalize()}: {d.page_content}" for d in docs)

    cls = classify_chain.invoke({"examples": examples, "complaint": req.complaint})
    classification = extract(cls.content if hasattr(cls, "content") else str(cls))

    if classification not in ["customer", "fibre"]:
        raise HTTPException(400, "Cannot classify")

    sol = solution_chain.invoke({"classification": classification, "complaint": req.complaint})
    solution = (sol.content if hasattr(sol, "content") else str(sol)).strip()

    return Resp(classification=classification, solution=solution)

@app.get("/health")
def health(): return {"status": "healthy", "memory_optimized": True}