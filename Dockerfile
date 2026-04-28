FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models during build to avoid timeout at startup
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"
RUN python3 -c "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup']))"
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY mcp_server.py .

ENV QDRANT_HOST=localhost
ENV QDRANT_PORT=6333
ENV MCP_PORT=8000
ENV PORT=8000

EXPOSE 8000

CMD ["python", "mcp_server.py"]
