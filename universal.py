from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from util import get_retriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

OLLAMA_BASE_URL = "http://10.50.1.101:11434"

# Load vector retrievers for each category
customer_retriever = get_retriever("C:/Users/diloyanomon/Documents/customer issue.txt")
fibre_retriever = get_retriever("C:/Users/diloyanomon/Documents/fibre issues.txt")

# Load LLM
#model = ChatOllama(model="qwen3:8b",base_url=OLLAMA_BASE_URL)

# Get API key from env
openai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= openai_api_key,  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

template = """
You are a support classification agent.

You will be given examples of either customer complaints or fibre complaints.
Classify the user's complaint as one of the following: "customer" or "fibre".

Examples:
{examples}

User Complaint:
{complaint}

Respond with only one word: "customer" or "fibre".
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    complaint = input("Enter complaint (q to quit): ")
    if complaint.lower() == "q":
        break

    # Get relevant examples from both categories
    customer_examples = customer_retriever.invoke(complaint)
    fibre_examples = fibre_retriever.invoke(complaint)

    all_examples = ""
    for doc in customer_examples:
        all_examples += f"Customer: {doc.page_content}\n"
    for doc in fibre_examples:
        all_examples += f"Fibre: {doc.page_content}\n"

    result = chain.invoke({"examples": all_examples, "complaint": complaint})
    print(f"\nClassification: {result}")
