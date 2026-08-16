"""CFFI binding build script for the pauli_label C kernel (Phase 3a,
binding 2/4).

Uses CFFI's API mode (``set_source`` + a compiled extension), the
standard higher-performance CFFI pattern - not ABI mode (dlopen'ing a
pre-built .so at import time), which trades startup simplicity for
call-time overhead. Run once to generate and compile the extension:

    python3 build_pauli_label_cffi.py

Produces ``_pauli_label_cffi.cpython-<tag>.so`` (gitignored, like the
Cython binding's compiled output - build locally).
"""

from pathlib import Path

from cffi import FFI

NATIVE_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "paulikit" / "_native"

ffibuilder = FFI()

# Declarations CFFI needs to generate the Python-callable wrapper.
# Mirrors pauli_label.h exactly (CFFI's cdef parser doesn't handle
# #include, so the header can't just be pasted in directly).
ffibuilder.cdef("""
    void pauli_label(uint32_t x_mask, uint32_t z_mask, int n_qubits, char *out);
    void pauli_label_batch(
        const uint32_t *x_masks,
        const uint32_t *z_masks,
        int64_t n_terms,
        int n_qubits,
        char *out
    );
""")

ffibuilder.set_source(
    "_pauli_label_cffi",
    '#include "pauli_label.h"',
    sources=[str(NATIVE_DIR / "pauli_label.c")],
    include_dirs=[str(NATIVE_DIR)],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
