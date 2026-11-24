.PHONY: install test lint eval run docker
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
eval:
	python -m agentic_rag eval | tee evals/results.md
run:
	uvicorn agentic_rag.api:create_app --factory --reload
docker:
	docker build -t agentic-rag-lab . && docker run -p 8000:8000 agentic-rag-lab
