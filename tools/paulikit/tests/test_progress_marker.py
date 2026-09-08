"""Tests for the append-only progress marker
(docs/superpowers/specs/2026-09-08-append-only-progress-marker-design.md).

These pin the record format and the recovery rule directly, separately
from the decomposition behaviour the four existing checkpoint test
files already cover.
"""

import numpy as np
import pytest

from paulikit.algorithms.fwht import (
    _PROGRESS_RECORD,
    _append_progress_record,
    _read_completed_indices,
)


def test_record_is_8_bytes():
    assert _PROGRESS_RECORD.size == 8


def test_round_trip_recovers_exactly_the_appended_set(tmp_path):
    path = tmp_path / "p.bin"
    for i in (0, 1, 2, 7):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 2, 7}


def test_absent_file_is_the_empty_set(tmp_path):
    assert _read_completed_indices(tmp_path / "absent.bin") == set()


def test_empty_file_is_the_empty_set(tmp_path):
    path = tmp_path / "p.bin"
    path.write_bytes(b"")
    assert _read_completed_indices(path) == set()


def test_records_need_not_be_sorted(tmp_path):
    # Parallel workers complete out of order, so the file is not
    # sorted. Recovery must not assume it is.
    path = tmp_path / "p.bin"
    for i in (5, 0, 3, 1):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {0, 1, 3, 5}


def test_duplicate_records_collapse(tmp_path):
    # A rollback-resume can record the same chunk twice; the file is
    # append-only and never compacted. Reading into a set handles it.
    path = tmp_path / "p.bin"
    for i in (2, 2, 2):
        _append_progress_record(path, i)
    assert _read_completed_indices(path) == {2}


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5, 6, 7])
def test_torn_final_record_is_discarded(tmp_path, cut):
    # A crash mid-append can only ever tear the LAST record, because
    # appends are 8 bytes in one call and never rewritten. Every
    # complete record must survive; the partial one must not appear.
    path = tmp_path / "p.bin"
    for i in (0, 1, 2):
        _append_progress_record(path, i)
    raw = path.read_bytes()
    path.write_bytes(raw[:-cut])
    assert _read_completed_indices(path) == {0, 1}


def test_large_chunk_index_survives(tmp_path):
    # u64, matching the width the frame header uses for chunk_index,
    # so the two cannot disagree about range.
    path = tmp_path / "p.bin"
    _append_progress_record(path, 2**40)
    assert _read_completed_indices(path) == {2**40}
