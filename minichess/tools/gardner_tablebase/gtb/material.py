from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import comb
from typing import Iterable

from .constants import (
    BLACK,
    CHAR_TO_TYPE,
    KING,
    MATERIAL_ORDER,
    PAWN,
    TYPE_TO_CHAR,
    WHITE,
)


def _normalize_side(text: str) -> str:
    symbols = [char.upper() for char in text.strip() if char.isalpha()]
    if symbols.count("K") != 1:
        raise ValueError(f"Each material side must contain exactly one king: {text!r}")
    if any(symbol not in CHAR_TO_TYPE for symbol in symbols):
        raise ValueError(f"Unsupported material symbols in {text!r}")
    extras = "".join(symbol * symbols.count(symbol) for symbol in MATERIAL_ORDER)
    return "K" + extras


def _side_key(side: str) -> tuple[int, ...]:
    return tuple(side.count(symbol) for symbol in MATERIAL_ORDER)


def rotate_swap_board(board):
    """Rotate 180 degrees and exchange colors."""
    import numpy as np

    transformed = np.zeros(25, dtype=np.int8)
    for square, piece in enumerate(board):
        if piece:
            transformed[24 - square] = -int(piece)
    return transformed


@dataclass(frozen=True)
class MaterialSpec:
    white: str
    black: str

    @classmethod
    def parse(cls, signature: str, canonical: bool = True) -> "MaterialSpec":
        if "v" not in signature.lower():
            raise ValueError("Material signature must look like KQvK or KRPvKR.")
        left, right = signature.upper().split("V", 1)
        spec = cls(_normalize_side(left), _normalize_side(right))
        return spec.canonical()[0] if canonical else spec

    def canonical(self) -> tuple["MaterialSpec", bool]:
        left = _side_key(self.white)
        right = _side_key(self.black)
        if left > right or (left == right and self.white >= self.black):
            return self, False
        return MaterialSpec(self.black, self.white), True

    @property
    def signature(self) -> str:
        return f"{self.white}v{self.black}"

    @property
    def piece_count(self) -> int:
        return len(self.white) + len(self.black)

    @property
    def pawn_count(self) -> int:
        return self.white.count("P") + self.black.count("P")

    @property
    def group_codes(self) -> tuple[int, ...]:
        groups = [WHITE * KING, BLACK * KING]
        for color, side in ((WHITE, self.white), (BLACK, self.black)):
            for symbol in MATERIAL_ORDER:
                count = side.count(symbol)
                if count:
                    groups.append(color * CHAR_TO_TYPE[symbol])
        return tuple(groups)

    @property
    def group_counts(self) -> tuple[int, ...]:
        counts = [1, 1]
        for side in (self.white, self.black):
            for symbol in MATERIAL_ORDER:
                count = side.count(symbol)
                if count:
                    counts.append(count)
        return tuple(counts)

    @property
    def expanded_codes(self) -> tuple[int, ...]:
        values: list[int] = []
        for code, count in zip(self.group_codes, self.group_counts):
            values.extend([code] * count)
        return tuple(values)

    @property
    def raw_count(self) -> int:
        remaining = 25
        total = 1
        for count in self.group_counts:
            total *= comb(remaining, count)
            remaining -= count
        return total * 2

    def dependencies(self) -> set["MaterialSpec"]:
        result: set[MaterialSpec] = set()
        for side_name in ("white", "black"):
            side = getattr(self, side_name)
            opponent = self.black if side_name == "white" else self.white
            for index, symbol in enumerate(side[1:], start=1):
                reduced = side[:index] + side[index + 1:]
                candidate = MaterialSpec(reduced, opponent) if side_name == "white" else MaterialSpec(opponent, reduced)
                result.add(candidate.canonical()[0])
            if "P" in side:
                pawn_index = side.index("P")
                for promotion in "QRBN":
                    promoted = side[:pawn_index] + promotion + side[pawn_index + 1:]
                    promoted = _normalize_side(promoted)
                    candidate = MaterialSpec(promoted, opponent) if side_name == "white" else MaterialSpec(opponent, promoted)
                    result.add(candidate.canonical()[0])
        result.discard(self)
        return result

    @classmethod
    def from_board(cls, board) -> tuple["MaterialSpec", bool]:
        white = ["K"]
        black = ["K"]
        wk = bk = 0
        for piece in board:
            piece = int(piece)
            if not piece:
                continue
            symbol = TYPE_TO_CHAR[abs(piece)]
            if piece > 0:
                if symbol == "K":
                    wk += 1
                else:
                    white.append(symbol)
            else:
                if symbol == "K":
                    bk += 1
                else:
                    black.append(symbol)
        if wk != 1 or bk != 1:
            raise ValueError("A tablebase position must contain exactly one king per side.")
        spec = cls(_normalize_side("".join(white)), _normalize_side("".join(black)))
        return spec.canonical()


def dependency_closure(targets: Iterable[MaterialSpec]) -> list[MaterialSpec]:
    found: set[MaterialSpec] = set()
    stack = list(targets)
    while stack:
        spec = stack.pop().canonical()[0]
        if spec in found:
            continue
        found.add(spec)
        stack.extend(spec.dependencies())
    return sorted(found, key=lambda item: (item.piece_count, item.pawn_count, item.signature))


def _count_vectors(total: int, slots: int = 5):
    if slots == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _count_vectors(total - first, slots - 1):
            yield (first,) + rest


def _side_from_counts(counts: tuple[int, ...]) -> str:
    return "K" + "".join(symbol * count for symbol, count in zip(MATERIAL_ORDER, counts))


def all_materials(max_pieces: int) -> list[MaterialSpec]:
    if max_pieces < 2 or max_pieces > 6:
        raise ValueError("This generator supports 2 through 6 pieces.")
    result: set[MaterialSpec] = set()
    for extras in range(max_pieces - 1):
        for white_extras in range(extras + 1):
            black_extras = extras - white_extras
            for left, right in product(_count_vectors(white_extras), _count_vectors(black_extras)):
                spec = MaterialSpec(_side_from_counts(left), _side_from_counts(right)).canonical()[0]
                result.add(spec)
    return sorted(result, key=lambda item: (item.piece_count, item.pawn_count, item.signature))
