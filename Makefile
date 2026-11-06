-include config.mk

VENV       ?= venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest

.PHONY: help env install install-requirements install-dev tools trotterkit clean distclean test test-import test-syntax test-run test-tools docs-tools

find_notebooks = find . -name '*.ipynb' \
	-not -path './.git/*' \
	-not -path '*/.ipynb_checkpoints/*' \
	-not -name 'test.ipynb' \
	-not -name 'Untitled.ipynb'

help:
	@echo "openqcp-lab Makefile"
	@echo ""
	@echo "Usage: make TARGET"
	@echo ""
	@echo "Environment:"
	@echo "  env                   Provision venv/ with ./bootstrap (Python $(if $(REQUIRED_PYTHON_VERSION),$(REQUIRED_PYTHON_VERSION),3.12))"
	@echo "  install               Refresh requirements.txt, then build the tools"
	@echo "  install-requirements  Refresh requirements.txt only"
	@echo "  install-dev           install, plus requirements-dev.txt"
	@echo "  tools                 Build every tool the tutorials import; run"
	@echo "                        again after pulling changes under tools/"
	@echo "  trotterkit            Build tools/trotterkit only (arbitrary-N notebook)"
	@echo ""
	@echo "Tests:"
	@echo "  test                  Run all of the test-* targets below"
	@echo "  test-import           Import the core packages and the tools"
	@echo "  test-syntax           Check that every notebook is valid JSON"
	@echo "  test-run              Execute the notebooks with nbmake"
	@echo "  test-tools            Run the test suite of each tool in tools/"
	@echo ""
	@echo "Other:"
	@echo "  docs-tools            Build the Sphinx docs of tools that have them"
	@echo "  clean                 Remove caches and bytecode"
	@echo "  distclean             clean, plus venv/, config.mk and config.log"

env:
	./bootstrap

# The tools build after the requirements, never alongside them (both
# use the venv's pip), so this stays correct under make -j.
install: install-requirements
	$(MAKE) tools

install-requirements:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

# The tools the tutorials import, built into the venv. A tool's build
# is incremental, so running this again after a pull is quick.
tools: trotterkit

# tools/trotterkit built into the venv; the arbitrary-N
# coupled-oscillator notebook evolves its state with it.
trotterkit:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	$(PIP) install meson ninja
	PATH="$(abspath $(VENV))/bin:$$PATH" && export PATH && \
	    cd tools/trotterkit && ./configure --venv="$(abspath $(VENV))" && \
	    $(MAKE) build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	@for tool_dir in tools/*/; do \
		if [ ! -f "$$tool_dir/Makefile" ] && [ -x "$$tool_dir/configure" ]; then \
			(cd "$$tool_dir" && ./configure >/dev/null); \
		fi; \
		if [ -f "$$tool_dir/Makefile" ]; then $(MAKE) -C "$$tool_dir" clean; fi; \
	done

distclean: clean
	rm -rf $(VENV) config.mk config.log

test: test-import test-syntax test-run test-tools

test-import:
	@test -x "$(PYTHON)" || { echo "No venv found at $(VENV) - run 'make env' first." >&2; exit 1; }
	@echo "Checking core package imports..."
	$(PYTHON) -c "import numpy, scipy, sympy, matplotlib, pennylane, classiq; print('All core packages imported successfully.')"
	@echo "Checking the tools the tutorials import..."
	@$(PYTHON) -c "import trotterkit; print('trotterkit backends:', ', '.join(trotterkit.backends()))" \
	    || { echo "trotterkit is missing or broken - run 'make tools'." >&2; exit 1; }

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
		if [ ! -f "$$tool_dir/Makefile" ] && [ -x "$$tool_dir/configure" ]; then \
			echo "Bootstrapping $$tool_dir via its own configure..."; \
			(cd "$$tool_dir" && ./configure >/dev/null); \
		fi; \
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
		if [ ! -f "$$tool_dir/Makefile" ] && [ -x "$$tool_dir/configure" ]; then \
			(cd "$$tool_dir" && ./configure >/dev/null); \
		fi; \
		if [ -f "$$tool_dir/docs/conf.py" ] && [ -f "$$tool_dir/Makefile" ]; then \
			echo "Building docs for $$tool_dir..."; \
			$(MAKE) -C "$$tool_dir" docs PYTHON="$(abspath $(PYTHON))" \
				PIP="$(abspath $(PYTHON)) -m pip" || exit 1; \
		fi; \
	done
