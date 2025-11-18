# ---- Base image with Python 3.11 (smaller than 3.13, saves ~200 MB) ----
FROM python:3.11-slim-bookworm

# ---- System dependencies (keeps image small) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Copy only requirements first (caching magic) ----
COPY requirements.txt .

# ---- Install Python deps (this layer caches unless requirements change) ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy your code (do this after pip so it doesn't reinstall everything on code changes) ----
COPY . .

# ---- Memory optimizations for 512 MiB runtime ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# ---- Expose port (Render expects this) ----
EXPOSE 8000

# ---- Use your memory-optimized main.py with lazy loading (the one I gave you) ----
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]