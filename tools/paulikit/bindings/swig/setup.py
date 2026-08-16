"""Standalone build script for the SWIG pauli_label binding (Phase 3a,
binding 4/4). Not part of the main paulikit package build - see
bindings/README.md. Build in place:

    python3 setup.py build_ext --inplace
"""

from pathlib import Path

from setuptools import Extension, setup

NATIVE_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "paulikit" / "_native"

extension = Extension(
    name="_pauli_label_swig",
    sources=["pauli_label.i", str(NATIVE_DIR / "pauli_label.c")],
    include_dirs=[str(NATIVE_DIR)],
    swig_opts=["-I" + str(NATIVE_DIR)],
)

setup(
    name="pauli_label_swig",
    ext_modules=[extension],
    py_modules=["pauli_label_swig"],
)
