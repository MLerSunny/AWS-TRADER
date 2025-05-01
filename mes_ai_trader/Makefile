.PHONY: dev-setup tests deploy-dev clean lint format materialize-dev

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	poetry install
	poetry run pre-commit install
	@echo "Creating .env file if not exists"
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Development setup complete!"

# Run tests
tests:
	@echo "Running tests..."
	poetry run pytest tests/ --cov=mes_ai_trader

# Code quality
lint:
	@echo "Running linters..."
	poetry run black . --check
	poetry run isort . --check
	poetry run flake8 .
	poetry run mypy .

# Format code
format:
	@echo "Formatting code..."
	poetry run black .
	poetry run isort .

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist/
	rm -rf build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type d -name "*.pyc" -exec rm -rf {} +

# Deploy to development environment
deploy-dev:
	@echo "Deploying to development environment..."
	docker-compose build
	docker-compose up -d
	@echo "Development deployment complete!"

# Feature store materialization
materialize-dev:
	@echo "Applying feature definitions and materializing features to online store..."
	cd models/feature_repo && \
	poetry run feast apply && \
	poetry run feast materialize-incremental $$(date +%F)
	@echo "Feature materialization complete!"

# Default target
all: format lint tests 