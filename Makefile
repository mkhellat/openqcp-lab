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
	@echo ""
	@echo "Checking notebook syntax..."
	@python3 -c "import json, sys; [json.load(open(f)) for f in ['coupled_harmonic_oscillators/N_coupled_harmonic_oscillators_1_D_N_2.ipynb', 'minimize_expectation_value/minimize_vqc_output.ipynb', 'nonunitary_quantum_computing/lcu_2x2.ipynb', 'quantum_walk/quantum_walk_discrete_time_line_16nodes.ipynb', 'quantum_fourier_transform_abelian/qft_abelian_qpe_hadamard.ipynb', 'quantum_variational_algorithms/VA0_qubo_and_vqe.ipynb']]; print('All notebooks have valid JSON structure.')" 2>/dev/null || echo "Warning: Some notebooks may have JSON syntax issues."
