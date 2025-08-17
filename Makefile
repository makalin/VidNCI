.PHONY: help install install-dev install-ai test test-cov lint format clean build dist docs

help:  ## Show this help message
	@echo "VidNCI Development Commands"
	@echo "=========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e .[dev]

install-ai:  ## Install with AI capabilities
	pip install -e .[ai]

install-full:  ## Install with all dependencies
	pip install -e .[full]

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ --cov=vidnci --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 vidnci/ tests/ examples/
	mypy vidnci/

format:  ## Format code with black and isort
	black vidnci/ tests/ examples/
	isort vidnci/ tests/ examples/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build:  ## Build the package
	python setup.py build

dist: clean build  ## Create distribution packages
	python setup.py sdist bdist_wheel

docs:  ## Build documentation
	cd docs && make html

serve-docs:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

example:  ## Run the basic example
	python examples/basic_usage.py

cli-test:  ## Test the command-line interface
	vidnci info
	vidnci analyze --help
	vidnci batch --help
	vidnci compare --help

check-deps:  ## Check for outdated dependencies
	pip list --outdated

update-deps:  ## Update dependencies to latest versions
	pip install --upgrade -r requirements.txt

cython-build:  ## Build Cython extensions
	python setup.py build_ext --inplace

cython-clean:  ## Clean Cython build files
	find . -name "*.c" -delete
	find . -name "*.so" -delete
	find . -name "*.cpp" -delete

pre-commit:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

docker-build:  ## Build Docker image
	docker build -t vidnci .

docker-run:  ## Run Docker container
	docker run -it --rm vidnci

docker-test:  ## Run tests in Docker
	docker run -it --rm vidnci make test

# Development workflow
dev-setup: install-dev pre-commit  ## Complete development setup
	@echo "Development environment setup complete!"

# Quality checks
quality: format lint test  ## Run all quality checks

# Release preparation
release-prep: clean quality dist  ## Prepare for release
	@echo "Release preparation complete!"
	@echo "Check dist/ directory for packages"

# Help for common issues
troubleshoot:  ## Common troubleshooting commands
	@echo "Common troubleshooting commands:"
	@echo "  make clean          - Clean all build artifacts"
	@echo "  make cython-clean  - Clean Cython build files"
	@echo "  make install-dev   - Reinstall with dev dependencies"
	@echo "  make test          - Run tests to check installation"
	@echo "  python -c 'import vidnci; print(vidnci.__version__)' - Check import"

# Environment info
env-info:  ## Show environment information
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Pip version:"
	@pip --version
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(vidnci|numpy|opencv|scipy)"

# Performance testing
perf-test:  ## Run performance tests
	@echo "Running performance tests..."
	@python -c "
import time
import numpy as np
from vidnci import CodeGenerator, CodeExtractor

# Test code generation performance
print('Testing code generation...')
generator = CodeGenerator(seed=42)
start_time = time.time()
code = generator.generate_gaussian_code(10000)
gen_time = time.time() - start_time
print(f'Generated 10k Gaussian code in {gen_time:.4f}s')

# Test code extraction performance (simulated)
print('Testing code extraction...')
extractor = CodeExtractor(use_cython=False)
frames = np.random.random((100, 100, 100)).astype(np.float32)
start_time = time.time()
code_image = extractor.extract_code_image_from_frames(frames, code)
ext_time = time.time() - start_time
print(f'Extracted code image in {ext_time:.4f}s')
"
