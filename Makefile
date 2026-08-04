-include config.mk

VENV       ?= venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest

.PHONY: help env install install-dev clean distclean test test-import test-syntax test-run test-tools docs-tools

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
	@echo "  make install-dev - Also install dev/profiling tools (requirements-dev.txt)"
	@echo "  make test      - Run import checks, notebook JSON checks, notebook execution, and tools/ test suites"
	@echo "  make test-tools - Run tools/*/tests/ (e.g. tools/paulikit) - installs each in editable mode first"
	@echo "  make docs-tools - Build Sphinx docs for every tool under tools/ that provides them"
	@echo "  make clean     - Remove cache/bytecode files"
	@echo "  make distclean - Remove venv/, config.mk, config.log, and cache files"

env:
	./bootstrap

install:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@for tool_dir in tools/*/; do \
		if [ -f "$$tool_dir/Makefile" ]; then $(MAKE) -C "$$tool_dir" clean; fi; \
	done

distclean: clean
	rm -rf $(VENV) config.mk config.log

test: test-import test-syntax test-run test-tools

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

test-tools:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	@for tool_dir in tools/*/; do \
		if [ -f "$$tool_dir/Makefile" ]; then \
			echo "Testing $$tool_dir..."; \
			$(MAKE) -C "$$tool_dir" test PYTHON="$(abspath $(PYTHON))" \
				PIP="$(abspath $(PYTHON)) -m pip" \
				PYTEST="$(abspath $(PYTEST))" || exit 1; \
		fi; \
	done

docs-tools:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	@for tool_dir in tools/*/; do \
		if [ -f "$$tool_dir/docs/conf.py" ] && [ -f "$$tool_dir/Makefile" ]; then \
			echo "Building docs for $$tool_dir..."; \
			$(MAKE) -C "$$tool_dir" docs PYTHON="$(abspath $(PYTHON))" \
				PIP="$(abspath $(PYTHON)) -m pip" || exit 1; \
		fi; \
	done
