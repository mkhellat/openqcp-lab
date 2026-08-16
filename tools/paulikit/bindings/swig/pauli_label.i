/* SWIG interface for the pauli_label C kernel (Phase 3a, binding 4/4).
 *
 * Neither entry point in pauli_label.h maps cleanly onto SWIG's
 * default argument handling: both write into a caller-supplied
 * buffer via a raw `char *out` parameter, and the batch function
 * takes raw `const uint32_t *` array pointers with a separate length
 * argument - exactly the "extra typemap work for structs/arrays"
 * cost PLAN.md's binding-technique survey (Section 4) flagged for
 * SWIG going in. Hand-written typemaps below, rather than reusing
 * numpy.i (not installed on this machine, and reproducing its typemap
 * boilerplate by hand is itself the intended comparison point) or
 * changing the C API to be more SWIG-friendly (would make this an
 * unfair comparison against the other three bindings' identical C
 * calls).
 */

%module pauli_label_swig

%{
#include "pauli_label.h"
%}

%include "stdint.i"

/* pauli_label: allocate the output buffer internally sized by
 * n_qubits, then convert it to a Python str on the way out - SWIG's
 * `argout` typemap pattern for "C writes into a buffer we manage". */
%typemap(in, numinputs=0) (char *out_label, int n_qubits_dup) (char temp[33]) {
    $1 = temp;
    $2 = 33;
}
%typemap(argout) (char *out_label, int n_qubits_dup) {
    %append_output(PyUnicode_FromString($1));
}

%inline %{
static void pauli_label_str(uint32_t x_mask, uint32_t z_mask, int n_qubits,
                             char *out_label, int n_qubits_dup) {
    (void)n_qubits_dup;
    pauli_label(x_mask, z_mask, n_qubits, out_label);
}
%}

/* pauli_label_batch: takes the x/z arrays as raw Python buffers (via
 * a typemap converting a contiguous NumPy uint32 array's address),
 * and writes into a caller-preallocated Python bytes/bytearray
 * output buffer - avoids SWIG owning any allocation, matching the
 * kernel's actual no-allocation contract in pauli_label.h. */
%typemap(in) (const uint32_t *x_masks, int64_t n_terms_x) {
    Py_buffer view;
    if (PyObject_GetBuffer($input, &view, PyBUF_C_CONTIGUOUS) != 0) {
        SWIG_exception_fail(SWIG_TypeError, "expected a C-contiguous buffer for x_masks");
    }
    $1 = (uint32_t *)view.buf;
    $2 = view.len / sizeof(uint32_t);
    PyBuffer_Release(&view);
}
%typemap(in) (const uint32_t *z_masks) {
    Py_buffer view;
    if (PyObject_GetBuffer($input, &view, PyBUF_C_CONTIGUOUS) != 0) {
        SWIG_exception_fail(SWIG_TypeError, "expected a C-contiguous buffer for z_masks");
    }
    $1 = (uint32_t *)view.buf;
    PyBuffer_Release(&view);
}
%typemap(in) (char *out_buf) {
    Py_buffer view;
    if (PyObject_GetBuffer($input, &view, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) != 0) {
        SWIG_exception_fail(SWIG_TypeError, "expected a writable C-contiguous buffer for out_buf");
    }
    $1 = (char *)view.buf;
    PyBuffer_Release(&view);
}

%inline %{
static void pauli_label_batch_raw(const uint32_t *x_masks, int64_t n_terms_x,
                                   const uint32_t *z_masks,
                                   int n_qubits, char *out_buf) {
    pauli_label_batch(x_masks, z_masks, n_terms_x, n_qubits, out_buf);
}
%}
