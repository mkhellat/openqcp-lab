"""Tests for the binary chunk-framed checkpoint format
(docs/superpowers/specs/2026-09-08-binary-chunk-framed-checkpoint-design.md).

These test the format primitives directly - header encoding, frame
round-trip, and recovery from truncated or corrupt files - separately
from the decomposition behaviour that the four existing checkpoint
test files already pin.
"""

import struct

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _CHECKPOINT_HEADER_STRUCT,
    _CHECKPOINT_MAGIC,
    _CHECKPOINT_VERSION,
    _checkpoint_frame_header,
    _parse_checkpoint_frame_header,
)


def test_header_is_24_bytes():
    assert _CHECKPOINT_HEADER_STRUCT.size == 24


@pytest.mark.parametrize(
    "dtype", [np.dtype(np.uint16), np.dtype(np.uint32), np.dtype(np.intp)]
)
def test_header_round_trip_every_index_dtype(dtype):
    raw = _checkpoint_frame_header(chunk_index=7, n_terms=1234, idx_dtype=dtype)
    assert len(raw) == 24
    chunk_index, n_terms, got = _parse_checkpoint_frame_header(raw)
    assert (chunk_index, n_terms) == (7, 1234)
    assert got == dtype


def test_header_round_trip_large_values():
    # chunk_index and n_terms are u64; N=150 has ~91.6M terms overall
    # and thousands of chunks, so neither may be narrowed to u32.
    raw = _checkpoint_frame_header(
        chunk_index=2**40, n_terms=91_652_096, idx_dtype=np.dtype(np.uint32)
    )
    chunk_index, n_terms, _ = _parse_checkpoint_frame_header(raw)
    assert chunk_index == 2**40
    assert n_terms == 91_652_096


def test_parse_rejects_wrong_magic():
    raw = bytearray(
        _checkpoint_frame_header(0, 1, np.dtype(np.uint32))
    )
    raw[0:4] = b"XXXX"
    with pytest.raises(ValueError, match="magic"):
        _parse_checkpoint_frame_header(bytes(raw))


def test_parse_rejects_unknown_version():
    raw = _CHECKPOINT_HEADER_STRUCT.pack(
        _CHECKPOINT_MAGIC, _CHECKPOINT_VERSION + 1, 1, 0, 0, 1
    )
    with pytest.raises(ValueError, match="version"):
        _parse_checkpoint_frame_header(raw)


def test_parse_rejects_unknown_index_dtype_code():
    # An unknown width must raise rather than default to a guess:
    # guessing wrong silently wraps indices and yields wrong labels.
    raw = _CHECKPOINT_HEADER_STRUCT.pack(
        _CHECKPOINT_MAGIC, _CHECKPOINT_VERSION, 99, 0, 0, 1
    )
    with pytest.raises(ValueError, match="index dtype"):
        _parse_checkpoint_frame_header(raw)


def test_header_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="index dtype"):
        _checkpoint_frame_header(0, 1, np.dtype(np.float64))


def test_parse_rejects_short_buffer():
    with pytest.raises(ValueError, match="truncated"):
        _parse_checkpoint_frame_header(b"PKCP")
