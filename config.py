# Central config for paths and settings
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CUSTOMER_FILE = os.path.join(BASE_DIR, "data", "customer_file.txt")
FIBRE_FILE = os.path.join(BASE_DIR, "data", "fibre_file.txt")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_index")
LOG_FILE = os.path.join(BASE_DIR, "api.log")

# App settings
API_PORT = 8000
API_HOST = "0.0.0.0"