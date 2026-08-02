-include config.mk

VENV       ?= venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest

.PHONY: help env install clean distclean test test-import test-syntax test-run

find_notebooks = find . -name '*.ipynb' \
	-not -path './.git/*' \
	-not -path '*/.ipynb_checkpoints/*' \
	-not -name 'test.ipynb' \
	-not -name 'Untitled.ipynb'

help:
	@echo "openqcp-lab Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make env       - Run ./bootstrap to provision venv/ (Python $(if $(REQUIRED_PYTHON_VERSION),$(REQUIRED_PYTHON_VERSION),3.12))"
	@echo "  make install   - Install/refresh dependencies into an existing venv"
	@echo "  make test      - Run import checks, notebook JSON checks, and full notebook execution"
	@echo "  make clean     - Remove cache/bytecode files"
	@echo "  make distclean - Remove venv/, config.mk, config.log, and cache files"

env:
	./bootstrap

install:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true

distclean: clean
	rm -rf $(VENV) config.mk config.log

test: test-import test-syntax test-run

test-import:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	@echo "Checking core package imports..."
	$(PYTHON) -c "import numpy, scipy, sympy, matplotlib, pennylane, classiq; print('All core packages imported successfully.')"

test-syntax:
	@echo "Checking notebook JSON structure..."
	@$(find_notebooks) -print0 | \
	$(PYTHON) -c "\
import json, sys; \
files = sys.stdin.read().split(chr(0))[:-1]; \
[json.load(open(f)) for f in files]; \
print('All %d notebooks have valid JSON structure.' % len(files))"

test-run:
	@test -x "$(PYTEST)" || { echo "nbmake not installed - run 'make env' first." >&2; exit 1; }
	@echo "Executing notebooks with nbmake..."
	$(find_notebooks) -print0 | xargs -0 $(PYTEST) --nbmake --nbmake-timeout=600
