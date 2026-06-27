from __future__ import annotations

import numpy as np

from .numba_compat import njit
from .constants import BISHOP, BLACK, KING, KNIGHT, MAX_MOVES, PAWN, QUEEN, ROOK, WHITE
from .board import (
    BISHOP_DF,
    BISHOP_DR,
    KING_DF,
    KING_DR,
    KNIGHT_DF,
    KNIGHT_DR,
    QUEEN_DF,
    QUEEN_DR,
    ROOK_DF,
    ROOK_DR,
    file_of,
    inside,
    rank_of,
    valid_position,
)
from .indexer import fast_rank_board


@njit(cache=True)
def reverse_quiet_predecessor_indices(
    board: np.ndarray,
    turn: int,
    group_codes: np.ndarray,
    group_counts: np.ndarray,
    radices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return canonical raw indices for same-material quiet predecessors.

    This is the hot retrograde kernel.  The older path returned full predecessor
    boards to Python and then ranked each board one by one.  Ranking inside the
    JIT kernel removes thousands/millions of Python calls during exact 5-piece
    generation while preserving the same legal-position checks and indexing.
    """
    previous_turn = -turn
    indices = np.zeros(MAX_MOVES, dtype=np.uint32)
    count = 0
    for target in range(25):
        piece = int(board[target])
        if piece == 0 or (1 if piece > 0 else -1) != previous_turn:
            continue
        piece_type = abs(piece)
        target_file = file_of(target)
        target_rank = rank_of(target)
        origins = np.full(24, -1, dtype=np.int16)
        origin_count = 0
        if piece_type == PAWN:
            direction = 1 if previous_turn == WHITE else -1
            origin_rank = target_rank - direction
            if 0 <= origin_rank < 5 and target_rank != (4 if previous_turn == WHITE else 0):
                origins[origin_count] = origin_rank * 5 + target_file
                origin_count += 1
        elif piece_type == KNIGHT or piece_type == KING:
            dfs = KNIGHT_DF if piece_type == KNIGHT else KING_DF
            drs = KNIGHT_DR if piece_type == KNIGHT else KING_DR
            for step in range(8):
                origin_file = target_file - int(dfs[step])
                origin_rank = target_rank - int(drs[step])
                if inside(origin_file, origin_rank):
                    origins[origin_count] = origin_rank * 5 + origin_file
                    origin_count += 1
        else:
            if piece_type == BISHOP:
                dfs, drs, length = BISHOP_DF, BISHOP_DR, 4
            elif piece_type == ROOK:
                dfs, drs, length = ROOK_DF, ROOK_DR, 4
            else:
                dfs, drs, length = QUEEN_DF, QUEEN_DR, 8
            for direction_index in range(length):
                origin_file = target_file - int(dfs[direction_index])
                origin_rank = target_rank - int(drs[direction_index])
                while inside(origin_file, origin_rank):
                    origin = origin_rank * 5 + origin_file
                    if board[origin] != 0:
                        break
                    origins[origin_count] = origin
                    origin_count += 1
                    origin_file -= int(dfs[direction_index])
                    origin_rank -= int(drs[direction_index])
        for origin_index in range(origin_count):
            origin = int(origins[origin_index])
            if origin < 0 or board[origin] != 0:
                continue
            predecessor = board.copy()
            predecessor[origin] = predecessor[target]
            predecessor[target] = 0
            if valid_position(predecessor, previous_turn):
                raw_index = fast_rank_board(predecessor, previous_turn, group_codes, group_counts, radices)
                if raw_index >= 0 and count < indices.shape[0]:
                    indices[count] = raw_index
                    count += 1
    return indices, count
