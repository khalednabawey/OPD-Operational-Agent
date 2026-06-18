FROM python:3.11-slim

# Prevent Python from writing .pyc files and ensure logs are flushed immediately
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies required to build ChromaDB native extensions (like hnswlib)
COPY requirements.txt .

# Install requirements. 
# Note: If you use the RemoteEmbeddingFunction, you can remove torch 
# from requirements.txt to save ~500MB and avoid slow initialization.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Using python as entrypoint allows you to pass flags like --force 
# via docker-compose or terminal without rewriting the whole command.
ENTRYPOINT ["python"]
CMD ["policy_rag.py"]