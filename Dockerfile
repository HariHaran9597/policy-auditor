# Streamlit app image for the Policy Auditor.
# Milvus itself runs via docker-compose.yml (separate services); this image
# talks to it over the compose network.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CPU-only torch keeps the image about 1.5 GB smaller than the default wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Bake the BGE models into the image so the first question is not a download
RUN python -c "from policy_auditor.indexing.embedder import get_embedder; get_embedder()" \
    && python -c "from policy_auditor.retrieval.reranker import get_reranker; get_reranker()"

EXPOSE 8501
CMD ["streamlit", "run", "src/policy_auditor/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
