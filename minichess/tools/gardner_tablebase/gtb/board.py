from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numba_compat import njit

from .constants import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    MAX_MOVES,
    PAWN,
    PROMOTIONS,
    QUEEN,
    ROOK,
    WHITE,
)

KNIGHT_DF = np.asarray((1, 2, 2, 1, -1, -2, -2, -1), dtype=np.int8)
KNIGHT_DR = np.asarray((2, 1, -1, -2, -2, -1, 1, 2), dtype=np.int8)
KING_DF = np.asarray((-1, 0, 1, -1, 1, -1, 0, 1), dtype=np.int8)
KING_DR = np.asarray((-1, -1, -1, 0, 0, 1, 1, 1), dtype=np.int8)
BISHOP_DF = np.asarray((1, 1, -1, -1), dtype=np.int8)
BISHOP_DR = np.asarray((1, -1, 1, -1), dtype=np.int8)
ROOK_DF = np.asarray((1, -1, 0, 0), dtype=np.int8)
ROOK_DR = np.asarray((0, 0, 1, -1), dtype=np.int8)
QUEEN_DF = np.asarray((1, 1, -1, -1, 1, -1, 0, 0), dtype=np.int8)
QUEEN_DR = np.asarray((1, -1, 1, -1, 0, 0, 1, -1), dtype=np.int8)
PROMOTION_ARRAY = np.asarray(PROMOTIONS, dtype=np.int8)


@njit(cache=True, inline="always")
def file_of(square: int) -> int:
    return square % 5


@njit(cache=True, inline="always")
def rank_of(square: int) -> int:
    return square // 5


@njit(cache=True, inline="always")
def inside(file: int, rank: int) -> bool:
    return 0 <= file < 5 and 0 <= rank < 5


@njit(cache=True)
def king_square(board: np.ndarray, color: int) -> int:
    target = color * KING
    for square in range(25):
        if board[square] == target:
            return square
    return -1


@njit(cache=True)
def is_attacked(board: np.ndarray, target: int, by_color: int) -> bool:
    target_file = file_of(target)
    target_rank = rank_of(target)
    for square in range(25):
        piece = int(board[square])
        if piece == 0 or (1 if piece > 0 else -1) != by_color:
            continue
        piece_type = abs(piece)
        file = file_of(square)
        rank = rank_of(square)
        if piece_type == PAWN:
            direction = 1 if by_color == WHITE else -1
            if target_rank == rank + direction and abs(target_file - file) == 1:
                return True
        elif piece_type == KNIGHT:
            for index in range(8):
                if file + int(KNIGHT_DF[index]) == target_file and rank + int(KNIGHT_DR[index]) == target_rank:
                    return True
        elif piece_type == KING:
            if abs(target_file - file) <= 1 and abs(target_rank - rank) <= 1:
                return True
        else:
            if piece_type == BISHOP:
                dfs, drs, length = BISHOP_DF, BISHOP_DR, 4
            elif piece_type == ROOK:
                dfs, drs, length = ROOK_DF, ROOK_DR, 4
            else:
                dfs, drs, length = QUEEN_DF, QUEEN_DR, 8
            for direction_index in range(length):
                next_file = file + int(dfs[direction_index])
                next_rank = rank + int(drs[direction_index])
                while inside(next_file, next_rank):
                    candidate = next_rank * 5 + next_file
                    if candidate == target:
                        return True
                    if board[candidate] != 0:
                        break
                    next_file += int(dfs[direction_index])
                    next_rank += int(drs[direction_index])
    return False


@njit(cache=True)
def in_check(board: np.ndarray, color: int) -> bool:
    king = king_square(board, color)
    return king >= 0 and is_attacked(board, king, -color)


@njit(cache=True)
def valid_position(board: np.ndarray, turn: int) -> bool:
    white_king = king_square(board, WHITE)
    black_king = king_square(board, BLACK)
    if white_king < 0 or black_king < 0:
        return False
    if abs(file_of(white_king) - file_of(black_king)) <= 1 and abs(rank_of(white_king) - rank_of(black_king)) <= 1:
        return False
    for square in range(25):
        piece = int(board[square])
        if abs(piece) == PAWN and (rank_of(square) == 0 or rank_of(square) == 4):
            return False
    # The side that made the preceding move may not be left in check.
    if in_check(board, -turn):
        return False
    return True


@njit(cache=True)
def apply_move(board: np.ndarray, move: np.ndarray) -> np.ndarray:
    result = board.copy()
    origin = int(move[0])
    target = int(move[1])
    promotion = int(move[2])
    piece = int(result[origin])
    result[origin] = 0
    result[target] = (1 if piece > 0 else -1) * promotion if promotion else piece
    return result


@njit(cache=True, inline="always")
def _push_move(moves: np.ndarray, count: int, origin: int, target: int, promotion: int, captured: int) -> int:
    if count >= moves.shape[0]:
        return count
    moves[count, 0] = origin
    moves[count, 1] = target
    moves[count, 2] = promotion
    moves[count, 3] = captured
    return count + 1


@njit(cache=True)
def pseudo_moves(board: np.ndarray, turn: int) -> tuple[np.ndarray, int]:
    moves = np.zeros((MAX_MOVES, 4), dtype=np.int16)
    count = 0
    for origin in range(25):
        piece = int(board[origin])
        if piece == 0 or (1 if piece > 0 else -1) != turn:
            continue
        piece_type = abs(piece)
        file = file_of(origin)
        rank = rank_of(origin)
        if piece_type == PAWN:
            direction = 1 if turn == WHITE else -1
            next_rank = rank + direction
            promotion_rank = 4 if turn == WHITE else 0
            if inside(file, next_rank):
                target = next_rank * 5 + file
                if board[target] == 0:
                    if next_rank == promotion_rank:
                        for promotion in PROMOTION_ARRAY:
                            count = _push_move(moves, count, origin, target, int(promotion), 0)
                    else:
                        count = _push_move(moves, count, origin, target, 0, 0)
                for delta_file in (-1, 1):
                    capture_file = file + delta_file
                    if not inside(capture_file, next_rank):
                        continue
                    capture_target = next_rank * 5 + capture_file
                    captured = int(board[capture_target])
                    if captured and (1 if captured > 0 else -1) == -turn and abs(captured) != KING:
                        if next_rank == promotion_rank:
                            for promotion in PROMOTION_ARRAY:
                                count = _push_move(moves, count, origin, capture_target, int(promotion), captured)
                        else:
                            count = _push_move(moves, count, origin, capture_target, 0, captured)
            continue
        if piece_type == KNIGHT or piece_type == KING:
            dfs = KNIGHT_DF if piece_type == KNIGHT else KING_DF
            drs = KNIGHT_DR if piece_type == KNIGHT else KING_DR
            for index in range(8):
                next_file = file + int(dfs[index])
                next_rank = rank + int(drs[index])
                if not inside(next_file, next_rank):
                    continue
                target = next_rank * 5 + next_file
                captured = int(board[target])
                if captured == 0:
                    count = _push_move(moves, count, origin, target, 0, 0)
                elif (1 if captured > 0 else -1) == -turn and abs(captured) != KING:
                    count = _push_move(moves, count, origin, target, 0, captured)
            continue
        if piece_type == BISHOP:
            dfs, drs, length = BISHOP_DF, BISHOP_DR, 4
        elif piece_type == ROOK:
            dfs, drs, length = ROOK_DF, ROOK_DR, 4
        else:
            dfs, drs, length = QUEEN_DF, QUEEN_DR, 8
        for direction_index in range(length):
            next_file = file + int(dfs[direction_index])
            next_rank = rank + int(drs[direction_index])
            while inside(next_file, next_rank):
                target = next_rank * 5 + next_file
                captured = int(board[target])
                if captured == 0:
                    count = _push_move(moves, count, origin, target, 0, 0)
                else:
                    if (1 if captured > 0 else -1) == -turn and abs(captured) != KING:
                        count = _push_move(moves, count, origin, target, 0, captured)
                    break
                next_file += int(dfs[direction_index])
                next_rank += int(drs[direction_index])
    return moves, count


@njit(cache=True)
def legal_moves(board: np.ndarray, turn: int) -> tuple[np.ndarray, int]:
    candidates, candidate_count = pseudo_moves(board, turn)
    moves = np.zeros((MAX_MOVES, 4), dtype=np.int16)
    count = 0
    for index in range(candidate_count):
        next_board = apply_move(board, candidates[index])
        if not in_check(next_board, turn):
            moves[count] = candidates[index]
            count += 1
    return moves, count


def reverse_quiet_predecessors(board: np.ndarray, turn: int):
    """Yield same-material legal predecessors connected by a quiet non-promotion move."""
    previous_turn = -int(turn)
    occupied = {index for index, piece in enumerate(board) if int(piece)}
    for target, raw_piece in enumerate(board):
        piece = int(raw_piece)
        if not piece or (1 if piece > 0 else -1) != previous_turn:
            continue
        piece_type = abs(piece)
        target_file, target_rank = target % 5, target // 5
        origins: list[int] = []
        if piece_type == PAWN:
            direction = 1 if previous_turn == WHITE else -1
            origin_rank = target_rank - direction
            if 0 <= origin_rank < 5 and target_rank not in (0, 4):
                origins.append(origin_rank * 5 + target_file)
        elif piece_type in (KNIGHT, KING):
            dfs = KNIGHT_DF if piece_type == KNIGHT else KING_DF
            drs = KNIGHT_DR if piece_type == KNIGHT else KING_DR
            for df, dr in zip(dfs.tolist(), drs.tolist()):
                origin_file = target_file - int(df)
                origin_rank = target_rank - int(dr)
                if 0 <= origin_file < 5 and 0 <= origin_rank < 5:
                    origins.append(origin_rank * 5 + origin_file)
        else:
            if piece_type == BISHOP:
                directions = zip(BISHOP_DF.tolist(), BISHOP_DR.tolist())
            elif piece_type == ROOK:
                directions = zip(ROOK_DF.tolist(), ROOK_DR.tolist())
            else:
                directions = zip(QUEEN_DF.tolist(), QUEEN_DR.tolist())
            for df, dr in directions:
                origin_file = target_file - int(df)
                origin_rank = target_rank - int(dr)
                while 0 <= origin_file < 5 and 0 <= origin_rank < 5:
                    origin = origin_rank * 5 + origin_file
                    if origin in occupied:
                        break
                    origins.append(origin)
                    origin_file -= int(df)
                    origin_rank -= int(dr)

        for origin in origins:
            if origin in occupied:
                continue
            predecessor = board.copy()
            predecessor[origin] = predecessor[target]
            predecessor[target] = 0
            if not valid_position(predecessor, previous_turn):
                continue
            moves, count = legal_moves(predecessor, previous_turn)
            for index in range(count):
                move = moves[index]
                if int(move[0]) == origin and int(move[1]) == target and int(move[2]) == 0 and int(move[3]) == 0:
                    yield predecessor, previous_turn
                    break


def move_to_uci(move) -> str:
    files = "abcde"
    origin, target, promotion = int(move[0]), int(move[1]), int(move[2])
    text = f"{files[origin % 5]}{origin // 5 + 1}{files[target % 5]}{target // 5 + 1}"
    if promotion:
        text += {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}[promotion]
    return text


def _compress_cells(cells):
    compact = ""
    empty = 0
    for cell in cells:
        if cell is None:
            empty += 1
        else:
            if empty:
                compact += str(empty)
                empty = 0
            compact += cell
    if empty:
        compact += str(empty)
    return compact


def _extract_padded_rectangle(expanded, top, bottom, left, right):
    inside = []
    outside_pieces = 0
    for row_index, row in enumerate(expanded):
        row_inside = top <= row_index <= bottom
        if row_inside:
            inside.append(row[left:right + 1])
        for file_index, symbol in enumerate(row):
            if symbol is None:
                continue
            file_inside = left <= file_index <= right
            if not row_inside or not file_inside:
                outside_pieces += 1
    return outside_pieces == 0 and len(inside) == 5, [_compress_cells(row) for row in inside]


def parse_fen(text: str):
    parts = text.strip().split()
    rows = parts[0].split("/")
    turn = WHITE if len(parts) < 2 or parts[1] == "w" else BLACK
    if len(rows) == 8:
        # v12.2: compact A1-E5 FEN is canonical. 8x8 input is kept as a
        # compatibility reader for old b2-f6 study FEN and optional standard
        # A1-E5 padded FEN. Tablebase/database indexing remains square-based.
        expanded = []
        for row in rows:
            cells = []
            for char in row:
                if char.isdigit():
                    cells.extend([None] * int(char))
                else:
                    cells.append(char)
            if len(cells) != 8:
                raise ValueError("Invalid 8x8 Gardner FEN.")
            expanded.append(cells)
        legacy_ok, legacy_rows = _extract_padded_rectangle(expanded, 2, 6, 1, 5)
        standard_ok, standard_rows = _extract_padded_rectangle(expanded, 3, 7, 0, 4)
        if legacy_ok:
            rows = legacy_rows
        elif standard_ok:
            rows = standard_rows
        else:
            raise ValueError("Pieces outside supported Gardner 5x5 areas.")
    if len(rows) != 5:
        raise ValueError("Gardner FEN requires five compact A1-E5 ranks or a compatible 8x8 padded FEN.")
    board = np.zeros(25, dtype=np.int8)
    mapping = {"p": PAWN, "n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN, "k": KING}
    for row_index, row in enumerate(rows):
        rank = 4 - row_index
        file = 0
        for char in row:
            if char.isdigit():
                file += int(char)
            else:
                if char.lower() not in mapping or file >= 5:
                    raise ValueError("Invalid piece placement.")
                board[rank * 5 + file] = (WHITE if char.isupper() else BLACK) * mapping[char.lower()]
                file += 1
        if file != 5:
            raise ValueError("Each Gardner rank must contain five squares.")
    return board, turn


def to_fen(board: np.ndarray, turn: int) -> str:
    reverse = {PAWN: "p", KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q", KING: "k"}
    rows = []
    for rank in range(4, -1, -1):
        row = ""
        empty = 0
        for file in range(5):
            piece = int(board[rank * 5 + file])
            if not piece:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            symbol = reverse[abs(piece)]
            row += symbol.upper() if piece > 0 else symbol
        if empty:
            row += str(empty)
        rows.append(row)
    return f"{'/'.join(rows)} {'w' if turn == WHITE else 'b'} - - 0 1"

@njit(cache=True)
def reverse_quiet_predecessor_array(board: np.ndarray, turn: int) -> tuple[np.ndarray, int]:
    """JIT kernel for same-material quiet predecessors.

    The current position is assumed legal. Captures and promotions change the
    material signature and are intentionally handled as external table edges.
    """
    previous_turn = -turn
    predecessors = np.zeros((MAX_MOVES, 25), dtype=np.int8)
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
            for index in range(8):
                origin_file = target_file - int(dfs[index])
                origin_rank = target_rank - int(drs[index])
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
                predecessors[count] = predecessor
                count += 1
    return predecessors, count
