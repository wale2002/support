# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from langchain_ollama import ChatOllama
# # from langchain_core.prompts import ChatPromptTemplate
# # from util import get_retriever  # Your vector DB util
# # from dotenv import load_dotenv
# # import os
# # import requests
# # from langchain_core.language_models import LLM
# # from langchain_core.outputs import Generation, LLMResult
# # import re
# # import logging  # For logging

# # # Load env
# # load_dotenv()

# # # Setup logging
# # logging.basicConfig(level=logging.INFO, filename='api.log', format='%(asctime)s - %(levelname)s - %(message)s')
# # logger = logging.getLogger(__name__)

# # OLLAMA_BASE_URL = "http://localhost:11434"
# # CUSTOMER_FILE = "data/customer_file.txt"
# # FIBRE_FILE = "data/fibre_file.txt"
# # retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # # Your OkeyMeta LLM (unchanged)
# # class OkeyMetaLLM(LLM):
# #     token: str
# #     base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

# #     def __init__(self, token: str):
# #         super().__init__(token=token)

# #     def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
# #         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
# #         headers = {"Authorization": f"Bearer {self.token}"}
# #         try:
# #             response = requests.get(full_url, headers=headers, timeout=30)
# #             response.raise_for_status()
# #             data = response.json()
# #             output = data.get('output') or data.get('response', '')
# #             if not output and 'choices' in data:
# #                 output = data['choices'][0].get('text', '') if data['choices'] else ''
# #             return output.strip()
# #         except requests.exceptions.RequestException as e:
# #             raise ValueError(f"OkeyMeta API error: {e}")

# #     @property
# #     def _llm_type(self) -> str:
# #         return "okeymeta"

# #     def _generate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
# #         generations = []
# #         for prompt in prompts:
# #             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
# #             generations.append([Generation(text=text)])
# #         return LLMResult(generations=generations)

# # # LLM Setup (unchanged)
# # okey_token = os.getenv("OKEYMETA_TOKEN")
# # if okey_token:
# #     model = OkeyMetaLLM(token=okey_token)
# #     logger.info("Using OkeyMeta LLM")
# # else:
# #     print("No OKEYMETA_TOKEN—using Ollama.")
# #     logger.info("Falling back to Ollama LLM")
# #     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # # Extract classification (unchanged)
# # def extract_classification(response_text: str) -> str:
# #     match = re.search(r'\b(customer|fibre)\b', response_text.lower())
# #     return match.group(1) if match else "unknown"

# # # Chains (slightly adapted for async if needed, but sync works fine)
# # classification_template = """
# # You are a support classification agent. Classify the complaint STRICTLY as "customer" or "fibre" ONLY. Do NOT add any other text.

# # Examples:
# # {examples}

# # User Complaint: {complaint}

# # Classification:"""
# # classification_prompt = ChatPromptTemplate.from_template(classification_template)
# # classification_chain = classification_prompt | model

# # solution_template = """
# # You are a helpful support agent. Based on the classified complaint type and the user's issue, provide a concise, empathetic solution or next steps. Keep it under 150 words.

# # Classification: {classification}
# # User Complaint: {complaint}

# # Response:"""
# # solution_prompt = ChatPromptTemplate.from_template(solution_template)
# # solution_chain = solution_prompt | model

# # # FastAPI App
# # app = FastAPI(title="AI Complaint Classifier", version="1.0")

# # # Pydantic models for input/output
# # class ComplaintRequest(BaseModel):
# #     complaint: str

# # class ClassificationResponse(BaseModel):
# #     classification: str
# #     solution: str | None = None
# #     confidence: str = "high"  # Optional: Add later

# # @app.post("/classify", response_model=ClassificationResponse)
# # async def classify_complaint(request: ComplaintRequest):
# #     if not request.complaint.strip():
# #         raise HTTPException(status_code=400, detail="Complaint cannot be empty")
    
# #     logger.info(f"Processing complaint: {request.complaint}")

# #     # Retrieve examples
# #     examples = retriever.invoke(request.complaint)
# #     all_examples = ""
# #     for doc in examples:
# #         category = doc.metadata.get("category", "unknown")
# #         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

# #     # Classify
# #     class_result = classification_chain.invoke({"examples": all_examples, "complaint": request.complaint})
# #     class_text = class_result.content if hasattr(class_result, 'content') else class_result
# #     classification = extract_classification(class_text)
    
# #     logger.info(f"Classified as: {classification}")

# #     if classification not in ["customer", "fibre"]:
# #         logger.warning("Unable to classify complaint")
# #         raise HTTPException(status_code=400, detail="Unable to classify complaint")

# #     # Solution
# #     sol_result = solution_chain.invoke({"classification": classification, "complaint": request.complaint})
# #     sol_text = sol_result.content if hasattr(sol_result, 'content') else sol_result

# #     logger.info(f"Generated solution for {classification}")
# #     return ClassificationResponse(classification=classification, solution=sol_text.strip())

# # # Health check
# # @app.get("/health")
# # def health():
# #     return {"status": "healthy"}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from util import get_retriever  # Your vector DB util
# from dotenv import load_dotenv
# import os
# import requests
# from langchain_core.language_models import LLM
# from langchain_core.outputs import Generation, LLMResult
# import re
# import logging  # For logging
# from pymongo import MongoClient  # For MongoDB integration
# from typing import List

# # Load env
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO, filename='api.log', format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# OLLAMA_BASE_URL = "http://localhost:11434"
# CUSTOMER_FILE = "data/customer_file.txt"
# FIBRE_FILE = "data/fibre_file.txt"
# retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# # MongoDB Setup - Yes, you'll need a database to store and retrieve complaints.
# # Using MongoDB for persistence. Install with: pip install pymongo
# # Add your MongoDB URI to .env as MONGO_URI=mongodb://localhost:27017/complaints_db
# # Or for cloud: mongodb+srv://user:pass@cluster.mongodb.net/complaints_db
# MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/complaints_db")  # Add database URL here
# mongo_client = MongoClient(MONGO_URI)
# db = mongo_client.get_database("complaints_db")  # Database name
# complaints_collection = db["complaints"]  # Collection to store complaints

# # Set the provided token (in production, keep in .env only)
# os.environ["OKEYMETA_TOKEN"] = "okeyai_65b110107b482d2b58ba9192a05457565d278862a785c21208910defe653020e"

# # Your OkeyMeta LLM (unchanged)
# class OkeyMetaLLM(LLM):
#     token: str
#     base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

#     def __init__(self, token: str):
#         super().__init__(token=token)

#     def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
#         full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
#         headers = {"Authorization": f"Bearer {self.token}"}
#         try:
#             response = requests.get(full_url, headers=headers, timeout=30)
#             response.raise_for_status()
#             data = response.json()
#             output = data.get('output') or data.get('response', '')
#             if not output and 'choices' in data:
#                 output = data['choices'][0].get('text', '') if data['choices'] else ''
#             return output.strip()
#         except requests.exceptions.RequestException as e:
#             raise ValueError(f"OkeyMeta API error: {e}")

#     @property
#     def _llm_type(self) -> str:
#         return "okeymeta"

#     def _generate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
#         generations = []
#         for prompt in prompts:
#             text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
#             generations.append([Generation(text=text)])
#         return LLMResult(generations=generations)

# # LLM Setup (unchanged)
# okey_token = os.getenv("OKEYMETA_TOKEN")
# if okey_token:
#     model = OkeyMetaLLM(token=okey_token)
#     logger.info("Using OkeyMeta LLM")
# else:
#     print("No OKEYMETA_TOKEN—using Ollama.")
#     logger.info("Falling back to Ollama LLM")
#     model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# # Extract classification (unchanged)
# def extract_classification(response_text: str) -> str:
#     match = re.search(r'\b(customer|fibre)\b', response_text.lower())
#     return match.group(1) if match else "unknown"

# # Chains (slightly adapted for async if needed, but sync works fine)
# classification_template = """
# You are a support classification agent. Classify the complaint STRICTLY as "customer" or "fibre" ONLY. Do NOT add any other text.

# Examples:
# {examples}

# User Complaint: {complaint}

# Classification:"""
# classification_prompt = ChatPromptTemplate.from_template(classification_template)
# classification_chain = classification_prompt | model

# solution_template = """
# You are a helpful support agent. Based on the classified complaint type and the user's issue, provide a concise, empathetic solution or next steps. Keep it under 150 words.

# Classification: {classification}
# User Complaint: {complaint}

# Response:"""
# solution_prompt = ChatPromptTemplate.from_template(solution_template)
# solution_chain = solution_prompt | model

# # FastAPI App
# app = FastAPI(title="AI Complaint Classifier", version="1.0")

# # Pydantic models for input/output
# class ComplaintRequest(BaseModel):
#     complaint: str

# class ClassificationResponse(BaseModel):
#     classification: str
#     solution: str | None = None
#     confidence: str = "high"  # Optional: Add later

# class ComplaintItem(BaseModel):
#     id: str
#     complaint: str
#     classification: str
#     solution: str
#     timestamp: str

# class ComplaintsResponse(BaseModel):
#     complaints: List[ComplaintItem]

# @app.post("/classify", response_model=ClassificationResponse)
# async def classify_complaint(request: ComplaintRequest):
#     if not request.complaint.strip():
#         raise HTTPException(status_code=400, detail="Complaint cannot be empty")
    
#     logger.info(f"Processing complaint: {request.complaint}")

#     # Retrieve examples
#     examples = retriever.invoke(request.complaint)
#     all_examples = ""
#     for doc in examples:
#         category = doc.metadata.get("category", "unknown")
#         all_examples += f"{category.capitalize()}: {doc.page_content}\n"

#     # Classify
#     class_result = classification_chain.invoke({"examples": all_examples, "complaint": request.complaint})
#     class_text = class_result.content if hasattr(class_result, 'content') else class_result
#     classification = extract_classification(class_text)
    
#     logger.info(f"Classified as: {classification}")

#     if classification not in ["customer", "fibre"]:
#         logger.warning("Unable to classify complaint")
#         raise HTTPException(status_code=400, detail="Unable to classify complaint")

#     # Solution
#     sol_result = solution_chain.invoke({"classification": classification, "complaint": request.complaint})
#     sol_text = sol_result.content if hasattr(sol_result, 'content') else sol_result

#     # Store in MongoDB
#     complaint_data = {
#         "complaint": request.complaint,
#         "classification": classification,
#         "solution": sol_text.strip(),
#         "timestamp": str(pd.Timestamp.now())  # Assuming pandas is imported; else use datetime.now().isoformat()
#     }
#     result = complaints_collection.insert_one(complaint_data)
#     logger.info(f"Stored complaint with ID: {result.inserted_id}")

#     logger.info(f"Generated solution for {classification}")
#     return ClassificationResponse(classification=classification, solution=sol_text.strip())

# # New endpoint to get complaints
# @app.get("/getComplaints", response_model=ComplaintsResponse)
# async def get_complaints(limit: int = 10, skip: int = 0):
#     """
#     Retrieve a list of processed complaints.
#     - limit: Number of complaints to return (default: 10)
#     - skip: Number of complaints to skip (for pagination, default: 0)
#     """
#     cursor = complaints_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
#     complaints = []
#     for doc in cursor:
#         complaints.append(ComplaintItem(
#             id=str(doc["_id"]),
#             complaint=doc["complaint"],
#             classification=doc["classification"],
#             solution=doc["solution"],
#             timestamp=doc["timestamp"]
#         ))
#     return ComplaintsResponse(complaints=complaints)

# # Health check
# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from util import get_retriever  # Your vector DB util
from dotenv import load_dotenv
import os
import requests
from langchain_core.language_models import LLM
from langchain_core.outputs import Generation, LLMResult
import re
import logging  # For logging
from pymongo import MongoClient  # For MongoDB integration
from typing import List
from datetime import datetime  # Added for timestamp

# Load env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, filename='api.log', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
CUSTOMER_FILE = "data/customer_file.txt"
FIBRE_FILE = "data/fibre_file.txt"
retriever = get_retriever(CUSTOMER_FILE, FIBRE_FILE)

# MongoDB Setup - Yes, you'll need a database to store and retrieve complaints.
# Using MongoDB for persistence. Install with: pip install pymongo
# Add your MongoDB URI to .env as MONGO_URI=mongodb://localhost:27017/complaints_db
# Or for cloud: mongodb+srv://user:pass@cluster.mongodb.net/complaints_db
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/complaints_db")  # Add database URL here
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_database("complaints_db")  # Database name
complaints_collection = db["complaints"]  # Collection to store complaints

# Set the provided token (in production, keep in .env only)
os.environ["OKEYMETA_TOKEN"] = "okeyai_65b110107b482d2b58ba9192a05457565d278862a785c21208910defe653020e"

# Your OkeyMeta LLM (unchanged)
class OkeyMetaLLM(LLM):
    token: str
    base_url: str = "https://api.okeymeta.com.ng/api/ssailm/model/okeyai3.0-vanguard/okeyai"

    def __init__(self, token: str):
        super().__init__(token=token)

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        full_url = f"{self.base_url}?input={requests.utils.quote(prompt)}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(full_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            output = data.get('output') or data.get('response', '')
            if not output and 'choices' in data:
                output = data['choices'][0].get('text', '') if data['choices'] else ''
            return output.strip()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"OkeyMeta API error: {e}")

    @property
    def _llm_type(self) -> str:
        return "okeymeta"

    def _generate(self, prompts, stop=None, run_manager=None, **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

# LLM Setup (unchanged)
okey_token = os.getenv("OKEYMETA_TOKEN")
if okey_token:
    model = OkeyMetaLLM(token=okey_token)
    logger.info("Using OkeyMeta LLM")
else:
    print("No OKEYMETA_TOKEN—using Ollama.")
    logger.info("Falling back to Ollama LLM")
    model = ChatOllama(model="llama3.1:8b", base_url=OLLAMA_BASE_URL)

# Extract classification (unchanged)
def extract_classification(response_text: str) -> str:
    match = re.search(r'\b(customer|fibre)\b', response_text.lower())
    return match.group(1) if match else "unknown"

# Chains (slightly adapted for async if needed, but sync works fine)
classification_template = """
You are a support classification agent. Classify the complaint STRICTLY as "customer" or "fibre" ONLY. Do NOT add any other text.

Examples:
{examples}

User Complaint: {complaint}

Classification:"""
classification_prompt = ChatPromptTemplate.from_template(classification_template)
classification_chain = classification_prompt | model

solution_template = """
You are a helpful support agent. Based on the classified complaint type and the user's issue, provide a concise, empathetic solution or next steps. Keep it under 150 words.

Classification: {classification}
User Complaint: {complaint}

Response:"""
solution_prompt = ChatPromptTemplate.from_template(solution_template)
solution_chain = solution_prompt | model

# FastAPI App
app = FastAPI(title="AI Complaint Classifier", version="1.0")

# Pydantic models for input/output
class ComplaintRequest(BaseModel):
    complaint: str

class ClassificationResponse(BaseModel):
    classification: str
    solution: str | None = None
    confidence: str = "high"  # Optional: Add later

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
        raise HTTPException(status_code=400, detail="Complaint cannot be empty")
    
    logger.info(f"Processing complaint: {request.complaint}")

    # Retrieve examples
    examples = retriever.invoke(request.complaint)
    all_examples = ""
    for doc in examples:
        category = doc.metadata.get("category", "unknown")
        all_examples += f"{category.capitalize()}: {doc.page_content}\n"

    # Classify
    class_result = classification_chain.invoke({"examples": all_examples, "complaint": request.complaint})
    class_text = class_result.content if hasattr(class_result, 'content') else class_result
    classification = extract_classification(class_text)
    
    logger.info(f"Classified as: {classification}")

    if classification not in ["customer", "fibre"]:
        logger.warning("Unable to classify complaint")
        raise HTTPException(status_code=400, detail="Unable to classify complaint")

    # Solution
    sol_result = solution_chain.invoke({"classification": classification, "complaint": request.complaint})
    sol_text = sol_result.content if hasattr(sol_result, 'content') else sol_result

    # Store in MongoDB
    complaint_data = {
        "complaint": request.complaint,
        "classification": classification,
        "solution": sol_text.strip(),
        "timestamp": datetime.now().isoformat()  # Fixed: Use datetime instead of pd
    }
    result = complaints_collection.insert_one(complaint_data)
    logger.info(f"Stored complaint with ID: {result.inserted_id}")

    logger.info(f"Generated solution for {classification}")
    return ClassificationResponse(classification=classification, solution=sol_text.strip())

# New endpoint to get complaints
@app.get("/getComplaints", response_model=ComplaintsResponse)
async def get_complaints(limit: int = 10, skip: int = 0):
    """
    Retrieve a list of processed complaints.
    - limit: Number of complaints to return (default: 10)
    - skip: Number of complaints to skip (for pagination, default: 0)
    """
    cursor = complaints_collection.find().sort("timestamp", -1).skip(skip).limit(limit)
    complaints = []
    for doc in cursor:
        complaints.append(ComplaintItem(
            id=str(doc["_id"]),
            complaint=doc["complaint"],
            classification=doc["classification"],
            solution=doc["solution"],
            timestamp=doc["timestamp"]
        ))
    return ComplaintsResponse(complaints=complaints)

# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)