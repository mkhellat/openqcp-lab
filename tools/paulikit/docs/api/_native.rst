paulikit._native
=====================

.. automodule:: paulikit._native
   :members:
   :undoc-members:
   :show-inheritance:

Optional compiled extension (Cython/C++) used by
``paulikit.algorithms.fwht`` when available, with a pure-Python
fallback otherwise. See the README's "Native extension" section and
:doc:`../plan` Phase 3c for build details and rationale.

Autodoc note: this page only renders members if the extension was
compiled at doc-build time (``-Dnative=auto`` or ``enabled``). With
``-Dnative=disabled``, ``paulikit._native.pauli_label_native`` doesn't
exist and this page will be empty aside from this note.
