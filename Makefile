.PHONY: help env install clean test

help:
	@echo "openqcp-lab Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make env      - Create virtual environment and install dependencies"
	@echo "  make install  - Install dependencies in current environment"
	@echo "  make clean    - Remove virtual environment and cache files"
	@echo "  make test     - Run basic import tests (if implemented)"

env:
	@echo "Creating virtual environment..."
	python3 -m venv venv
	@echo "Installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Environment created! Activate it with:"
	@echo "  source venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	@echo "Removing virtual environment and cache files..."
	rm -rf venv
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

test:
	@echo "Running basic import tests..."
	@python3 -c "import numpy; import scipy; import sympy; import matplotlib; import pennylane; import classiq; print('All core packages imported successfully.')" || echo "Some packages failed to import."
