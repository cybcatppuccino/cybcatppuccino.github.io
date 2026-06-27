from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from .numba_compat import njit

from .constants import BLACK, WHITE
from .material import MaterialSpec, rotate_swap_board

COMB = np.zeros((26, 7), dtype=np.int64)
for n in range(26):
    for k in range(min(6, n) + 1):
        COMB[n, k] = comb(n, k)


@njit(cache=True, inline="always")
def _rank_selected_positions(positions, k, n):
    rank = 0
    previous = -1
    remaining = k
    for item in range(k):
        chosen = int(positions[item])
        for candidate in range(previous + 1, chosen):
            rank += int(COMB[n - candidate - 1, remaining - 1])
        previous = chosen
        remaining -= 1
    return rank


@njit(cache=True)
def fast_rank_board(board, turn, group_codes, group_counts, radices):
    available = np.arange(25, dtype=np.int16)
    available_count = 25
    rank_value = 0
    for group in range(group_codes.shape[0]):
        code = int(group_codes[group])
        count = int(group_counts[group])
        positions = np.empty(6, dtype=np.int16)
        selected_squares = np.empty(6, dtype=np.int16)
        selected_count = 0
        for position in range(available_count):
            square = int(available[position])
            if int(board[square]) == code:
                positions[selected_count] = position
                selected_squares[selected_count] = square
                selected_count += 1
        if selected_count != count:
            return -1
        combination_rank = _rank_selected_positions(positions, count, available_count)
        rank_value = rank_value * int(radices[group]) + combination_rank
        next_available = np.empty(25, dtype=np.int16)
        next_count = 0
        for position in range(available_count):
            square = int(available[position])
            keep = True
            for item in range(count):
                if square == int(selected_squares[item]):
                    keep = False
                    break
            if keep:
                next_available[next_count] = square
                next_count += 1
        available = next_available
        available_count = next_count
    return rank_value * 2 + (0 if turn == WHITE else 1)


@njit(cache=True)
def _unrank_positions(rank, n, k):
    result = np.empty(6, dtype=np.int16)
    result_count = 0
    next_value = 0
    remaining = k
    while remaining > 0:
        for candidate in range(next_value, n):
            count = int(COMB[n - candidate - 1, remaining - 1])
            if rank < count:
                result[result_count] = candidate
                result_count += 1
                next_value = candidate + 1
                remaining -= 1
                break
            rank -= count
    return result


@njit(cache=True)
def fast_unrank_board(index, group_codes, group_counts, radices):
    turn = WHITE if index % 2 == 0 else BLACK
    value = index // 2
    group_total = radices.shape[0]
    ranks = np.zeros(group_total, dtype=np.int64)
    for group in range(group_total - 1, -1, -1):
        radix = int(radices[group])
        ranks[group] = value % radix
        value //= radix
    board = np.zeros(25, dtype=np.int8)
    available = np.arange(25, dtype=np.int16)
    available_count = 25
    for group in range(group_total):
        count = int(group_counts[group])
        positions = _unrank_positions(int(ranks[group]), available_count, count)
        selected = np.empty(6, dtype=np.int16)
        for item in range(count):
            square = int(available[int(positions[item])])
            selected[item] = square
            board[square] = int(group_codes[group])
        next_available = np.empty(25, dtype=np.int16)
        next_count = 0
        for position in range(available_count):
            square = int(available[position])
            keep = True
            for item in range(count):
                if square == int(selected[item]):
                    keep = False
                    break
            if keep:
                next_available[next_count] = square
                next_count += 1
        available = next_available
        available_count = next_count
    return board, turn


def rank_combination(indices: list[int], n: int, k: int) -> int:
    values = np.asarray(indices + [0] * (6 - len(indices)), dtype=np.int16)
    return int(_rank_selected_positions(values, k, n))


def unrank_combination(rank: int, n: int, k: int) -> list[int]:
    return [int(value) for value in _unrank_positions(rank, n, k)[:k]]


@dataclass
class PositionIndexer:
    spec: MaterialSpec

    def __post_init__(self):
        self.group_codes = self.spec.group_codes
        self.group_counts = self.spec.group_counts
        remaining = 25
        radices = []
        for count in self.group_counts:
            radices.append(comb(remaining, count))
            remaining -= count
        self.radices = tuple(radices)
        self.group_codes_array = np.asarray(self.group_codes, dtype=np.int8)
        self.group_counts_array = np.asarray(self.group_counts, dtype=np.int8)
        self.radices_array = np.asarray(self.radices, dtype=np.int64)
        self.raw_count = self.spec.raw_count

    def index_canonical(self, board, turn: int) -> int:
        value = int(fast_rank_board(np.asarray(board, dtype=np.int8), int(turn), self.group_codes_array, self.group_counts_array, self.radices_array))
        if value < 0:
            raise ValueError("Position does not match material group counts.")
        return value

    def index(self, board, turn: int, canonicalize: bool = True) -> int:
        actual_spec, swapped = MaterialSpec.from_board(board)
        working = np.asarray(board, dtype=np.int8)
        working_turn = int(turn)
        if canonicalize and swapped:
            working = rotate_swap_board(working)
            working_turn = -working_turn
        if actual_spec != self.spec:
            raise ValueError(f"Position material {actual_spec.signature} does not match {self.spec.signature}.")
        return self.index_canonical(working, working_turn)

    def unindex(self, index: int):
        if index < 0 or index >= self.raw_count:
            raise IndexError(index)
        return fast_unrank_board(int(index), self.group_codes_array, self.group_counts_array, self.radices_array)
