# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammadreza Khellat
# Licensed under the GNU GPL version 3 or later.
r"""Oriented simplices and sparse simplicial boundary operators.

A chain is a formal integer combination of oriented simplices. The
boundary $\partial_k$ sends a $k$-chain to a $(k-1)$-chain. That is a
linear map, so the encoding that matches the question is a sparse
matrix: one column per $k$-simplex, one row per $(k-1)$-simplex.

Use `boundary_matrix` when the question is linear algebra on chains.
Typical jobs: incidence signs, the identity
$\partial_{k-1}\circ\partial_k = 0$, and a matrix rank set next to a
face-count Euler characteristic. Use `SimplicialComplex.adjacency_list`
when the question is local on the 1-skeleton: the degree of a vertex,
a walk, a connected component, or whether a pure 1-complex is a
disjoint union of cycles. An adjacency list stores unordered
neighbour sets. It has no signs and no $k$-faces for $k > 1$, so it
cannot represent $\partial_k$. A boundary matrix is the wrong tool
for "which vertices are adjacent to this one?": the nonzero pattern
of $\partial_1$ is an oriented incidence matrix, not a walk list, and
higher $\partial_k$ are not graphs at all.

Positive orientation of a simplex is the increasing order of its
vertex ids. `oriented_simplices` lists those tuples in lexicographic
order, and that list is the ordered basis of $C_{\mathrm{dim}}$.
Deleting the vertex in position $i$ (0-based) contributes the sign
$(-1)^i$. Basis tuples are not rewritten to a coherent manifold
orientation. Those relative signs live only in the matrix entries.

The complex is unaugmented. The empty simplex is not a basis vector
(the same convention as `all_simplices`), so $\partial_0$ is the zero
map into a module of rank 0.

`boundary_matrix` returns a SciPy CSR matrix when SciPy imports. CSR
is the format to keep after the incidences have been scattered into
coordinates: the product $\partial_{k-1}\partial_k$ is then a sparse
matrix-matrix multiply. If SciPy is absent, the same nonzero pattern
is a `DictOfKeysMatrix`. Either object is sparse. `toarray` is only
for reading a tiny example; a dense block is not the stored operator.
"""

from __future__ import annotations

from .complex import SimplicialComplex

try:
    from scipy import sparse as _sparse
except ImportError:
    _sparse = None


class DictOfKeysMatrix:
    """Sparse matrix as a dictionary of nonzero integer entries.

    This is the fallback `boundary_matrix` returns when SciPy does not
    import. Keys are `(row, column)` pairs; missing keys are zero.
    Multiplication is the chain-map composition, the same role CSR
    plays when SciPy is present.

    Build one of these (or a CSR matrix) for a boundary. Do not expand
    it to a dense array in order to take degrees or to walk the
    1-skeleton; that is what `adjacency_list` is for, at cost $O(V+E)$
    in the edges that exist.
    """

    def __init__(
        self,
        entries: dict[tuple[int, int], int],
        shape: tuple[int, int],
    ) -> None:
        if (
            type(shape) is not tuple
            or len(shape) != 2
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise ValueError("shape must be a pair of nonnegative ints")
        nrows, ncols = shape
        stored: dict[tuple[int, int], int] = {}
        for key, value in entries.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or type(key[0]) is not int
                or type(key[1]) is not int
            ):
                raise TypeError("entries keys must be (row, column) int pairs")
            row, column = key
            if not (0 <= row < nrows and 0 <= column < ncols):
                raise ValueError(f"entry {key} lies outside shape {shape}")
            if type(value) is not int:
                raise TypeError("entry values must be int")
            if value != 0:
                stored[key] = value
        self.entries = stored
        self.shape = shape

    def toarray(self) -> list[list[int]]:
        """Dense rows, including explicit zeros.

        For a worked example small enough to read by hand. The stored
        operator remains `entries`.
        """
        nrows, ncols = self.shape
        array = [[0] * ncols for _ in range(nrows)]
        for (row, column), value in self.entries.items():
            array[row][column] = value
        return array

    def __matmul__(self, other: DictOfKeysMatrix) -> DictOfKeysMatrix:
        if type(other) is not DictOfKeysMatrix:
            return NotImplemented
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"cannot multiply shapes {self.shape} and {other.shape}"
            )
        by_column: dict[int, list[tuple[int, int]]] = {}
        for (row, column), value in self.entries.items():
            by_column.setdefault(column, []).append((row, value))
        accumulated: dict[tuple[int, int], int] = {}
        for (row, column), value in other.entries.items():
            for left_row, left_value in by_column.get(row, ()):
                key = (left_row, column)
                total = accumulated.get(key, 0) + left_value * value
                if total == 0:
                    accumulated.pop(key, None)
                else:
                    accumulated[key] = total
        return DictOfKeysMatrix(accumulated, (self.shape[0], other.shape[1]))


def oriented_simplices(K: SimplicialComplex, dim: int) -> list[tuple[int, ...]]:
    r"""Increasing vertex-id tuples for the simplices of dimension `dim`.

    The returned list is sorted lexicographically. It is the ordered
    basis of the chain group $C_{\mathrm{dim}}$: row `i` of
    `boundary_matrix(K, dim + 1)` and column `i` of
    `boundary_matrix(K, dim)` are this tuple.

    Why tuples, not neighbour sets: an oriented simplex is an ordered
    basis vector of a chain group. `adjacency_list` answers a different
    question (which vertices share an edge) and drops both order and
    every face of dimension other than 1.

    An out-of-range dimension, including the empty simplex at
    dimension $-1$, returns an empty list. The empty simplex is not a
    cell of this complex.
    """
    if type(dim) is not int:
        raise TypeError("dim must be an int")
    if dim < 0:
        return []
    faces = K.faces_by_dim().get(dim, ())
    return sorted(tuple(sorted(sigma)) for sigma in faces)


def boundary_matrix(K: SimplicialComplex, k: int):
    r"""Sparse $\partial_k \colon C_k \to C_{k-1}$.

    Columns are `oriented_simplices(K, k)` and rows are
    `oriented_simplices(K, k - 1)`, except that $k = 0$ has no rows:
    $C_{-1} = 0$ in the unaugmented complex. The entry in row $f$ and
    column $\sigma$ is $(-1)^i$ when face $f$ is $\sigma$ with the
    vertex in position $i$ deleted, and $0$ otherwise.

    Why a sparse incidence matrix: each column has exactly $k + 1$
    nonzeros, and the maps you compose are $\partial_{k-1}\partial_k$.
    SciPy CSR is used when `scipy` imports, because that product is a
    sparse multiply. Otherwise this returns a `DictOfKeysMatrix` with
    the same pattern. A dense matrix would store the zeros the
    identity $\partial^2 = 0$ is made of.

    When not to use it: vertex degree, graph distance, and cycle walks
    on the 1-skeleton. Those read `K.adjacency_list()` once. Scanning
    columns of $\partial_1$ rebuilds the edges with signs attached,
    which those walks do not need, and it still says nothing about
    higher faces that `adjacency_list` was never meant to hold.
    """
    if type(k) is not int:
        raise TypeError("k must be an int")
    if k < 0:
        raise ValueError("k must be nonnegative")
    domain = oriented_simplices(K, k)
    codomain = [] if k == 0 else oriented_simplices(K, k - 1)
    row_of = {face: index for index, face in enumerate(codomain)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    # k = 0 would delete the only vertex and land on the empty simplex,
    # which this unaugmented complex does not store. The map is zero
    # into a rank-0 module: no rows, and no entries.
    if k > 0:
        for column, simplex in enumerate(domain):
            for position in range(len(simplex)):
                face = simplex[:position] + simplex[position + 1 :]
                sign = 1 if position % 2 == 0 else -1
                rows.append(row_of[face])
                cols.append(column)
                data.append(sign)
    shape = (len(codomain), len(domain))
    if _sparse is not None:
        if data:
            return _sparse.csr_matrix(
                (data, (rows, cols)), shape=shape, dtype=int
            )
        return _sparse.csr_matrix(shape, dtype=int)
    return DictOfKeysMatrix(
        {(row, column): value for row, column, value in zip(rows, cols, data)},
        shape,
    )


def boundary_square_is_zero(
    K: SimplicialComplex,
    k: int,
    *,
    atol: float = 0.0,
) -> bool:
    r"""Whether $\partial_{k-1}\partial_k = 0$ as an integer matrix.

    `k` must be at least 1. The product is built from `boundary_matrix`,
    so it uses the same CSR-or-dict-of-keys choice. Nonzero entries of
    the product must all lie within `atol` of 0. The default `atol`
    asks for exact zeros, which holds for these $\pm 1$ incidence
    matrices over the integers.

    This is a chain-complex check. It does not replace
    `adjacency_list`: a graph walk cannot see the cancellation between
    the two paths around a triangle that makes the product vanish.
    """
    if type(k) is not int:
        raise TypeError("k must be an int")
    if k < 1:
        raise ValueError("k must be at least 1")
    product = boundary_matrix(K, k - 1) @ boundary_matrix(K, k)
    if isinstance(product, DictOfKeysMatrix):
        values = product.entries.values()
    else:
        values = product.data
    return all(abs(value) <= atol for value in values)
