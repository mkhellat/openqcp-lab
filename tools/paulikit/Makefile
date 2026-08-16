# Package-level Makefile for paulikit.
#
# This sits between the repository root Makefile (which loops over
# tools/*/ and delegates here) and docs/Makefile (the standard
# Sphinx-generated build, delegated to for the docs target).
#
# Usage (from this directory):
#   make install    - editable install with test extras
#   make test       - run the pytest suite
#   make test-native - build and run the standalone C kernel checks
#   make docs       - build Sphinx HTML docs (docs/_build/html)
#   make clean      - remove build/test artifacts (including docs/_build)

PYTHON  ?= python3
PIP     ?= $(PYTHON) -m pip
PYTEST  ?= pytest
CC      ?= cc
CXX     ?= c++

.PHONY: help install install-docs test test-native docs clean

help:
	@echo "paulikit Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Editable install with test extras (pip install -e .[test])"
	@echo "  make install-docs - Editable install with docs extras (pip install -e .[docs])"
	@echo "  make test         - Run the pytest suite"
	@echo "  make test-native  - Build and run the standalone C kernel checks (no Python binding)"
	@echo "  make docs         - Build Sphinx HTML documentation (delegates to docs/Makefile)"
	@echo "  make clean        - Remove build/test artifacts, including docs/_build"

install:
	$(PIP) install -e ".[test]"

install-docs:
	$(PIP) install -e ".[docs]"

test: install
	$(PYTEST)

test-native:
	$(CC) -std=c99 -Wall -Wextra -Werror -o /tmp/paulikit_test_pauli_label \
		src/paulikit/_native/pauli_label.c \
		src/paulikit/_native/test_pauli_label.c
	/tmp/paulikit_test_pauli_label
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Werror -o /tmp/paulikit_test_pauli_label_parallel \
		src/paulikit/_native/pauli_label.c \
		src/paulikit/_native/pauli_label_parallel.cpp \
		src/paulikit/_native/test_pauli_label_parallel.cpp -ltbb
	/tmp/paulikit_test_pauli_label_parallel

docs: install-docs
	$(MAKE) -C docs html SPHINXBUILD="$(PYTHON) -m sphinx"

clean:
	rm -rf .pytest_cache src/paulikit.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	$(MAKE) -C docs clean
