"""Internal package: optional compiled fast-path extension.

``pauli_label_native`` (the Cython/C/oneTBB module) is only present
here if it was built - see ``meson.build``'s ``native`` feature
option and PLAN.md's packaging note. Do not import
``paulikit._native.pauli_label_native`` directly outside of
``paulikit.algorithms.fwht`` - go through ``fwht_pauli_terms``, which
handles the case where it isn't available.
"""
