FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastmcp==3.2.0 \
    qdrant-client==1.17.1 \
    fastembed==0.7.4 \
    sentence-transformers==5.3.0 \
    python-dotenv==1.2.2 \
    httpx==0.28.1 \
    aiohttp==3.13.5 \
    pymupdf==1.24.0

# Pre-download models during build to avoid timeout at startup
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-large-v2')"
RUN python3 -c "from fastembed import SparseTextEmbedding; list(SparseTextEmbedding('Qdrant/bm25').embed(['warmup']))"
RUN python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY mcp_server.py .

ENV QDRANT_HOST=localhost
ENV QDRANT_PORT=6333
ENV PORT=8000

EXPOSE 8000

CMD ["python", "mcp_server.py"]
