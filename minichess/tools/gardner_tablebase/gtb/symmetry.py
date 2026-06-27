from __future__ import annotations

from functools import lru_cache

import numpy as np

from .indexer import PositionIndexer
from .material import MaterialSpec, rotate_swap_board

TRANSFORM_IDENTITY = 0
TRANSFORM_MIRROR_FILES = 1
TRANSFORM_ROTATE_SWAP = 2
TRANSFORM_ROTATE_SWAP_MIRROR = 3


def mirror_files(board):
    """Reflect a Gardner board across the vertical centre file."""
    source = np.asarray(board, dtype=np.int8)
    result = np.zeros(25, dtype=np.int8)
    for square in range(25):
        rank, file = divmod(square, 5)
        result[rank * 5 + (4 - file)] = source[square]
    return result


def apply_transform(board, turn: int, transform: int):
    result = np.asarray(board, dtype=np.int8)
    result_turn = int(turn)
    if transform & TRANSFORM_ROTATE_SWAP:
        result = rotate_swap_board(result)
        result_turn = -result_turn
    if transform & TRANSFORM_MIRROR_FILES:
        result = mirror_files(result)
    return result, result_turn


def transform_square(square: int, transform: int) -> int:
    square = int(square)
    rank, file = divmod(square, 5)
    if transform & TRANSFORM_ROTATE_SWAP:
        rank, file = 4 - rank, 4 - file
    if transform & TRANSFORM_MIRROR_FILES:
        file = 4 - file
    return rank * 5 + file


def transform_packed_move(value: int, transform: int) -> int:
    if not value or transform == TRANSFORM_IDENTITY:
        return int(value)
    origin = value & 31
    target = (value >> 5) & 31
    promotion = (value >> 10) & 7
    return transform_square(origin, transform) | (transform_square(target, transform) << 5) | (promotion << 10)


@lru_cache(maxsize=512)
def _indexer(signature: str) -> PositionIndexer:
    return PositionIndexer(MaterialSpec.parse(signature))


def canonical_sparse_position_with_transform(board, turn: int) -> tuple[MaterialSpec, int, int]:
    """Return exact sparse key and the involution mapping input to storage."""
    source = np.asarray(board, dtype=np.int8)
    best: tuple[str, int, MaterialSpec, int] | None = None
    for base_transform in range(4):
        candidate, candidate_turn = apply_transform(source, int(turn), base_transform)
        spec, swapped = MaterialSpec.from_board(candidate)
        effective_transform = base_transform
        if swapped:
            candidate = rotate_swap_board(candidate)
            candidate_turn = -candidate_turn
            effective_transform ^= TRANSFORM_ROTATE_SWAP
        index = _indexer(spec.signature).index_canonical(candidate, candidate_turn)
        key = (spec.signature, int(index), spec, effective_transform)
        if best is None or key[:2] < best[:2]:
            best = key
    assert best is not None
    return best[2], best[1], best[3]


def canonical_sparse_position(board, turn: int) -> tuple[MaterialSpec, int]:
    spec, index, _ = canonical_sparse_position_with_transform(board, turn)
    return spec, index


def unindex_sparse(spec: MaterialSpec, index: int):
    return _indexer(spec.signature).unindex(int(index))
