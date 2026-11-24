.PHONY: install test lint eval run docker
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
eval:
	python -m forecastlens eval | tee evals/results.md
run:
	uvicorn forecastlens.api:create_app --factory --reload
docker:
	docker build -t forecastlens . && docker run -p 8000:8000 forecastlens
