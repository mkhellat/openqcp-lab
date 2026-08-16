"""Standalone build script for the Cython pauli_label binding.

Not part of the main paulikit package build (see PLAN.md Phase 3a:
each of the four binding techniques is built/benchmarked
independently before deciding which one paulikit actually ships
with). Build in place:

    python3 setup.py build_ext --inplace
"""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

NATIVE_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "paulikit" / "_native"

extension = Extension(
    name="pauli_label_cy",
    sources=[
        "pauli_label_cy.pyx",
        str(NATIVE_DIR / "pauli_label.c"),
        str(NATIVE_DIR / "pauli_label_parallel.cpp"),
    ],
    include_dirs=[str(NATIVE_DIR), np.get_include()],
    libraries=["tbb"],
    language="c++",
    extra_compile_args=["-std=c++17"],
)

setup(
    name="pauli_label_cy",
    ext_modules=cythonize([extension], language_level=3),
)
