FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install dependencies first so this layer is cached until pyproject.toml changes.
COPY pyproject.toml ./
RUN mkdir -p src/agentic_rag && touch src/agentic_rag/__init__.py \
    && pip install .

COPY src ./src
COPY data ./data
COPY evals ./evals
RUN pip install --no-deps .

RUN useradd --create-home appuser
USER appuser

ENV CORPUS_DIR=/app/data/corpus SALES_CSV=/app/data/sales.csv LLM_PROVIDER=mock
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "agentic_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
