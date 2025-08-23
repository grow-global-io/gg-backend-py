# Multi-stage build for model caching
FROM python:3.11-slim as model-cache

# Install dependencies for downloading models
RUN pip install --no-cache-dir transformers torch huggingface-hub

# Pre-download the model to cache it
RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the cached model from the previous stage
COPY --from=model-cache /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
