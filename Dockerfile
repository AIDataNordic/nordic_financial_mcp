FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_server.py .

ENV QDRANT_HOST=localhost
ENV QDRANT_PORT=6333
ENV PORT=8000

EXPOSE 8000

CMD ["python", "mcp_server.py"]
