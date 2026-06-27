from __future__ import annotations

BOARD_SIZE = 5
SQUARES = 25
WHITE = 1
BLACK = -1

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

TYPE_TO_CHAR = {
    PAWN: "P",
    KNIGHT: "N",
    BISHOP: "B",
    ROOK: "R",
    QUEEN: "Q",
    KING: "K",
}
CHAR_TO_TYPE = {value: key for key, value in TYPE_TO_CHAR.items()}
MATERIAL_ORDER = "QRBNP"
PROMOTIONS = (QUEEN, ROOK, BISHOP, KNIGHT)

UNKNOWN = 9
LOSS = -1
DRAW = 0
WIN = 1
INVALID = 2

FORMAT_VERSION = 1
DEFAULT_BLOCK_SIZE = 1 << 18
MAX_MOVES = 128

KNIGHT_DELTAS = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
KING_DELTAS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS
