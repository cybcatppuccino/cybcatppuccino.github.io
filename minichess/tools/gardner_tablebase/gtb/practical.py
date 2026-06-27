from __future__ import annotations

import gzip
import json
import math
import os
import random
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import apply_move, in_check, legal_moves, parse_fen, valid_position
from .constants import BISHOP, BLACK, CHAR_TO_TYPE, DRAW, KING, KNIGHT, LOSS, PAWN, QUEEN, ROOK, UNKNOWN, WHITE, WIN
from .material import MaterialSpec
from .practical_storage import pack_move, pack_value, update_practical_manifest, write_sparse_table
from .storage import FlatTablebase, atomic_json
from .symmetry import canonical_sparse_position, unindex_sparse

PIECE_VALUE = {PAWN: 100, KNIGHT: 315, BISHOP: 325, ROOK: 500, QUEEN: 900, KING: 0}
DEFAULT_TARGET_BYTES = 96 * 1024 * 1024


@dataclass
class PracticalOptions:
    output_root: Path
    work_root: Path
    seed_file: Path
    exact_core_pieces: int = 3
    node_limit: int = 750_000
    rollouts: int = 3_000
    rollout_plies: int = 100
    hours: float = 2.0
    target_bytes: int = DEFAULT_TARGET_BYTES
    block_records: int = 65_536
    commit_every: int = 100
    random_seed: int = 0x47544271
    refine_passes: int = 3
    shell_seeds: int = 2_000_000
    coverage_mode: str = "coverage"
    min_fill_ratio: float = 0.85
    allow_underfilled: bool = False
    rebuild: bool = False


class PracticalGenerator:
    """Reachability-guided, sparse verified tablebase builder.

    It is intentionally not an exhaustive six-piece Syzygy replacement.  The
    configured small core (2–3 pieces by default) remains exhaustive. Higher
    4–6-piece positions are selected from real PGNs and reachable capture-biased
    playouts. The graph
    is expanded in priority order and retrograded.  Only mathematically proved
    WDL records are exported.  Their DTM value is a verified mate upper bound,
    which can be tightened by later runs without invalidating older results.
    """

    def __init__(self, options: PracticalOptions):
        self.options = options
        self.output = options.output_root
        self.work = options.work_root / "practical"
        self.work.mkdir(parents=True, exist_ok=True)
        self.db_path = self.work / "practical.sqlite3"
        if options.rebuild:
            self.db_path.unlink(missing_ok=True)
            shutil.rmtree(self.output / "practical", ignore_errors=True)
            (self.output / "practical-manifest.json").unlink(missing_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=60)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA cache_size=-131072")  # ~128 MiB for faster graph expansion
        self.connection.execute("PRAGMA mmap_size=67108864")
        self._schema()
        existing_core = self._meta("exactCorePieces", None)
        if existing_core is not None and int(existing_core) != int(options.exact_core_pieces):
            raise RuntimeError(
                "The practical checkpoint was created with a different exact core size. "
                "Use --rebuild-practical when changing --core-pieces."
            )
        self._set_meta("exactCorePieces", int(options.exact_core_pieces))
        self.connection.commit()
        # Sparse expansion probes the exact core extremely often. Keep core
        # tables flat in memory to avoid repeated gzip block decompression.
        self.core = FlatTablebase(self.output, max_cached_blocks=8, max_flat_bytes=256 * 1024 * 1024)
        self.material_ids: dict[str, int] = {}
        self.material_specs: dict[int, MaterialSpec] = {}
        self._load_materials()
        self.deadline = time.monotonic() + max(0.001, options.hours) * 3600

    def close(self):
        self.connection.close()

    def _schema(self):
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY,
                signature TEXT NOT NULL UNIQUE,
                piece_count INTEGER NOT NULL,
                pawn_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                material_id INTEGER NOT NULL,
                pos_index INTEGER NOT NULL,
                priority REAL NOT NULL DEFAULT 0,
                frequency INTEGER NOT NULL DEFAULT 0,
                distance INTEGER NOT NULL DEFAULT 0,
                expanded INTEGER NOT NULL DEFAULT 0,
                degree INTEGER NOT NULL DEFAULT 0,
                frontier INTEGER NOT NULL DEFAULT 0,
                external_win_count INTEGER NOT NULL DEFAULT 0,
                external_max_dtm INTEGER NOT NULL DEFAULT 0,
                external_loss_dtm INTEGER,
                external_loss_move INTEGER NOT NULL DEFAULT 0,
                terminal_state INTEGER,
                terminal_dtm INTEGER NOT NULL DEFAULT 0,
                state INTEGER NOT NULL DEFAULT 9,
                dtm INTEGER NOT NULL DEFAULT 0,
                best_move INTEGER NOT NULL DEFAULT 0,
                hint_move INTEGER NOT NULL DEFAULT 0,
                hint_score INTEGER NOT NULL DEFAULT 0,
                remaining INTEGER NOT NULL DEFAULT 0,
                max_child_dtm INTEGER NOT NULL DEFAULT 0,
                UNIQUE(material_id, pos_index)
            );
            CREATE INDEX IF NOT EXISTS nodes_expand_idx ON nodes(expanded, priority DESC, distance, id);
            CREATE INDEX IF NOT EXISTS nodes_state_idx ON nodes(state);
            CREATE TABLE IF NOT EXISTS edges (
                parent INTEGER NOT NULL,
                move INTEGER NOT NULL,
                child INTEGER NOT NULL,
                PRIMARY KEY(parent, move)
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS edges_child_idx ON edges(child, parent);
            CREATE TABLE IF NOT EXISTS retro_queue (
                node_id INTEGER PRIMARY KEY,
                state INTEGER NOT NULL,
                dtm INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS retro_distance_idx ON retro_queue(dtm, node_id);
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(nodes)")}
        if "hint_move" not in columns:
            self.connection.execute("ALTER TABLE nodes ADD COLUMN hint_move INTEGER NOT NULL DEFAULT 0")
        if "hint_score" not in columns:
            self.connection.execute("ALTER TABLE nodes ADD COLUMN hint_score INTEGER NOT NULL DEFAULT 0")
        self.connection.commit()

    def _meta(self, key: str, default=None):
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def _set_meta(self, key: str, value):
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, separators=(",", ":"))),
        )

    def _load_materials(self):
        for material_id, signature in self.connection.execute("SELECT id,signature FROM materials"):
            spec = MaterialSpec.parse(signature)
            self.material_ids[signature] = int(material_id)
            self.material_specs[int(material_id)] = spec

    def _material_id(self, spec: MaterialSpec) -> int:
        signature = spec.signature
        known = self.material_ids.get(signature)
        if known is not None:
            return known
        cursor = self.connection.execute(
            "INSERT INTO materials(signature,piece_count,pawn_count) VALUES(?,?,?)",
            (signature, spec.piece_count, spec.pawn_count),
        )
        material_id = int(cursor.lastrowid)
        self.material_ids[signature] = material_id
        self.material_specs[material_id] = spec
        return material_id

    @staticmethod
    def _piece_count(board) -> int:
        return int(np.count_nonzero(board))

    @staticmethod
    def _material_balance(board) -> int:
        score = 0
        for piece in board:
            value = PIECE_VALUE.get(abs(int(piece)), 0)
            score += value if int(piece) > 0 else -value
        return abs(score)

    def _position_priority(self, board, frequency: int, min_ply: int, source_bonus: float = 0.0) -> float:
        pieces = self._piece_count(board)
        balance = self._material_balance(board)
        balanced_bonus = max(0.0, 20.0 - balance / 55.0)
        pawn_bonus = sum(1 for piece in board if abs(int(piece)) == PAWN) * 2.0
        late_bonus = min(20.0, max(0.0, (min_ply - 20) * 0.25))
        return 1000.0 + 90.0 * math.log1p(max(1, frequency)) + balanced_bonus + pawn_bonus + late_bonus + source_bonus + (7 - pieces) * 2.0

    def _insert_position(self, board, turn: int, priority: float, frequency: int = 0, distance: int = 0) -> tuple[int, bool]:
        spec, index = canonical_sparse_position(board, turn)
        minimum = int(self.options.exact_core_pieces) + 1
        if spec.piece_count < minimum or spec.piece_count > 6:
            raise ValueError(f"Practical graph only stores {minimum}- through six-piece positions.")
        material_id = self._material_id(spec)
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO nodes(material_id,pos_index,priority,frequency,distance) VALUES(?,?,?,?,?)",
            (material_id, int(index), float(priority), int(frequency), int(distance)),
        )
        created = cursor.rowcount > 0
        if not created:
            self.connection.execute(
                "UPDATE nodes SET priority=MAX(priority,?), frequency=frequency+?, distance=MIN(distance,?) WHERE material_id=? AND pos_index=?",
                (float(priority), int(frequency), int(distance), material_id, int(index)),
            )
        row = self.connection.execute(
            "SELECT id FROM nodes WHERE material_id=? AND pos_index=?",
            (material_id, int(index)),
        ).fetchone()
        return int(row[0]), created

    def _move_hint_score(self, board, move, successor) -> int:
        captured = abs(int(move[3]))
        mover_piece = abs(int(board[int(move[0])]))
        score = 10
        if captured:
            score += 1000 + PIECE_VALUE.get(captured, 0) - PIECE_VALUE.get(mover_piece, 0) // 12
        if int(move[2]):
            score += 900 + PIECE_VALUE.get(abs(int(move[2])), 0) // 4
        if in_check(successor, -1 if int(board[int(move[0])]) > 0 else 1):
            score += 180
        if mover_piece == PAWN:
            score += 40
        score += max(0, 6 - self._piece_count(successor)) * 12
        return int(score)

    def _material_seed_plan(self) -> list[tuple[MaterialSpec, float]]:
        weights: dict[str, float] = {}
        if self.options.seed_file.exists():
            try:
                with gzip.open(self.options.seed_file, "rt", encoding="utf-8") as handle:
                    payload = json.load(handle)
                for entry in payload.get("positions", []):
                    board, turn = parse_fen(entry["fen"])
                    pieces = self._piece_count(board)
                    if self.options.exact_core_pieces < pieces <= 6 and valid_position(board, turn):
                        spec, _ = MaterialSpec.from_board(board)
                        freq = max(1, int(entry.get("frequency", 1)))
                        weights[spec.signature] = weights.get(spec.signature, 0.0) + 4.0 + math.log1p(freq) * 8.0
            except Exception as exc:
                print(f"Warning: could not read seed corpus for material weights: {exc}", flush=True)
        # Always include common practical mini-chess endings.  These are not
        # random padding: they are balanced pawn/rook/queen/minor-piece shells
        # that frequently decide real Gardner games and engine searches.
        common = {
            "KRPvKRP": 180, "KQvKR": 160, "KRvKRP": 145, "KRPvKR": 145,
            "KQvKQ": 140, "KQvKRP": 130, "KRPvKP": 120, "KPPvKPP": 115,
            "KRPvKBP": 110, "KRPvKNP": 110, "KBPvKBP": 95, "KNPvKNP": 95,
            "KRBvKRP": 90, "KRNvKRP": 90, "KQPvKRP": 85, "KRRvKRP": 80,
            "KPPPvKPP": 75, "KRPPvKRP": 75, "KQPPvKRP": 70, "KQRvKRP": 70,
            "KBNvKRP": 60, "KBBvKRP": 60, "KNNvKRP": 55, "KRBvKRN": 55,
        }
        for signature, weight in common.items():
            try:
                spec = MaterialSpec.parse(signature)
            except ValueError:
                continue
            if self.options.exact_core_pieces < spec.piece_count <= 6:
                weights[spec.signature] = weights.get(spec.signature, 0.0) + float(weight)
        if not weights:
            for signature in ("KRPvKRP", "KQvKR", "KQvKQ", "KRvKRP", "KPPvKPP"):
                spec = MaterialSpec.parse(signature)
                if self.options.exact_core_pieces < spec.piece_count <= 6:
                    weights[spec.signature] = 1.0
        result = [(MaterialSpec.parse(signature), weight) for signature, weight in weights.items()]
        result.sort(key=lambda item: (-item[1], item[0].piece_count, item[0].signature))
        return result

    @staticmethod
    def _sanitize_material_seed_plan(plan: list[tuple[MaterialSpec, float]]) -> list[tuple[MaterialSpec, float]]:
        """Return a non-empty, finite weighted material plan for shell seeding.

        v2 used random.choices(specs, weights=weights).  In long resumable runs,
        any accidental desynchronisation or non-finite weight can turn a resumable
        coverage pass into a hard crash.  Keep the weighted population as one list
        of (spec, weight) pairs and sanitize it once before sampling.
        """
        cleaned: list[tuple[MaterialSpec, float]] = []
        seen: set[str] = set()
        for spec, weight in plan:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(w) or w <= 0.0:
                continue
            if spec.signature in seen:
                # _material_seed_plan normally coalesces signatures, but keep this
                # guard so future changes cannot create duplicate populations.
                continue
            seen.add(spec.signature)
            cleaned.append((spec, w))
        if cleaned:
            return cleaned
        # Last-resort fallback: keep the generator resumable even if the external
        # seed corpus and all configured shells were filtered out.
        fallback: list[tuple[MaterialSpec, float]] = []
        for signature in ("KRPvKRP", "KQvKR", "KQvKQ", "KRvKRP", "KPPvKPP"):
            try:
                spec = MaterialSpec.parse(signature)
            except ValueError:
                continue
            fallback.append((spec, 1.0))
        return fallback

    @staticmethod
    def _weighted_material_choice(plan: list[tuple[MaterialSpec, float]], rng: random.Random) -> tuple[MaterialSpec, float]:
        """Deterministic safe replacement for random.choices(..., weights=...)."""
        total = 0.0
        for _, weight in plan:
            total += float(weight)
        if not plan or not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("Material-shell seed plan is empty or has no positive finite weights.")
        threshold = rng.random() * total
        cumulative = 0.0
        for spec, weight in plan:
            cumulative += float(weight)
            if threshold <= cumulative:
                return spec, float(weight)
        spec, weight = plan[-1]
        return spec, float(weight)

    def _sample_material_position(self, spec: MaterialSpec, rng: random.Random):
        codes = list(spec.expanded_codes)
        # Keep kings first but randomize non-king placements. This preserves the
        # material class while producing broad legal coverage inside the class.
        for _ in range(80):
            squares = rng.sample(range(25), len(codes))
            board = np.zeros(25, dtype=np.int8)
            for square, code in zip(squares, codes):
                board[int(square)] = int(code)
            turn = WHITE if rng.random() < 0.5 else BLACK
            if not valid_position(board, turn):
                continue
            moves, count = legal_moves(board, turn)
            if count == 0:
                # Terminal positions are already handled by exact/sparse proof;
                # shell seeding should spend space on playable practical nodes.
                continue
            # Prefer positions that are not obviously absurd material blowouts.
            if self._material_balance(board) > 1450 and rng.random() > 0.05:
                continue
            best_score = -10**9
            best_move = 0
            for move_number in range(count):
                move = moves[move_number]
                successor = apply_move(board, move)
                score = self._move_hint_score(board, move, successor)
                if score > best_score:
                    best_score = score
                    best_move = pack_move(move)
            return board, turn, best_move, best_score
        return None

    def seed_from_material_shells(self):
        target = int(max(0, self.options.shell_seeds))
        if target <= 0:
            return
        start_sample = int(self._meta("shellSeedNext", 0))
        if start_sample >= target:
            self._set_meta("shellSeedTarget", target)
            self._set_meta("shellSeeded", True)
            self.connection.commit()
            return
        plan = self._sanitize_material_seed_plan(self._material_seed_plan())
        unique_before = self._node_count()
        current_nodes = unique_before
        completed = start_sample
        accepted = int(self._meta("shellSeedAccepted", 0))
        for sample_no in range(start_sample, target):
            if current_nodes >= int(self.options.node_limit):
                break
            if time.monotonic() >= self.deadline - 5.0:
                break
            rng = random.Random(self.options.random_seed ^ 0x51EED5EED ^ (sample_no * 0x9E3779B1))
            spec, chosen_weight = self._weighted_material_choice(plan, rng)
            sampled = self._sample_material_position(spec, rng)
            if sampled is None:
                completed = sample_no + 1
                continue
            board, turn, hint_move, hint_score = sampled
            priority = self._position_priority(board, 1, 28, source_bonus=24.0 + math.log1p(max(1.0, float(chosen_weight))))
            node_id, created = self._insert_position(board, turn, priority, 1, 0)
            if created:
                accepted += 1
                current_nodes += 1
                self.connection.execute(
                    "UPDATE nodes SET hint_move=?,hint_score=MAX(hint_score,?) WHERE id=?",
                    (int(hint_move), int(hint_score), int(node_id)),
                )
            completed = sample_no + 1
            if completed % 5000 == 0:
                self._set_meta("shellSeedNext", completed)
                self._set_meta("shellSeedAccepted", accepted)
                self.connection.commit()
                print(f"Material-shell coverage seeds: {completed:,}/{target:,} samples, {current_nodes:,}/{self.options.node_limit:,} nodes", flush=True)
        self._set_meta("shellSeedNext", completed)
        self._set_meta("shellSeedTarget", target)
        self._set_meta("shellSeedAccepted", accepted)
        self._set_meta("shellSeeded", bool(completed >= target))
        self.connection.commit()
        unique_after = current_nodes
        print(f"Material-shell seeds: {unique_after - unique_before:,} new unique positions this run ({completed:,}/{target:,} samples complete)", flush=True)

    def seed_from_corpus(self):
        if self._meta("corpusSeeded", False):
            return
        if not self.options.seed_file.exists():
            print(f"Practical seed corpus not found at {self.options.seed_file}; continuing with rollout/material-shell seeds.", flush=True)
            self._set_meta("corpusSeeded", True)
            self._set_meta("corpusSeedCount", 0)
            self.connection.commit()
            return
        with gzip.open(self.options.seed_file, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        accepted = 0
        for entry in payload.get("positions", []):
            board, turn = parse_fen(entry["fen"])
            pieces = self._piece_count(board)
            if pieces <= self.options.exact_core_pieces or pieces > 6 or not valid_position(board, turn):
                continue
            position_spec, _ = MaterialSpec.from_board(board)
            if self.core.has(position_spec):
                continue
            priority = self._position_priority(
                board,
                int(entry.get("frequency", 1)),
                int(entry.get("minPly", 0)),
                source_bonus=35.0,
            )
            self._insert_position(board, turn, priority, int(entry.get("frequency", 1)), 0)
            accepted += 1
        self._set_meta("corpusSeeded", True)
        self._set_meta("corpusSeedCount", accepted)
        self.connection.commit()
        print(f"Practical seeds: {accepted:,} reachable PGN positions", flush=True)

    @staticmethod
    def _layout_fens(rng: random.Random) -> list[str]:
        standard = list("RNBQK")

        def fen(white, black):
            return f"{''.join(black).lower()}/ppppp/5/PPPPP/{''.join(white).upper()} w - - 0 1"

        layouts = [
            fen(standard, standard),
            fen(standard, list(reversed(standard))),
            fen(list("RNKQN"), list("RBKQB")),
        ]
        for _ in range(12):
            white = standard[:]
            rng.shuffle(white)
            layouts.append(fen(white, white))
            layouts.append(fen(white, list(reversed(white))))
            black = standard[:]
            rng.shuffle(black)
            layouts.append(fen(white, black))
        return layouts

    def seed_from_rollouts(self):
        layout_rng = random.Random(self.options.random_seed)
        layouts = self._layout_fens(layout_rng)
        start_rollout = int(self._meta("rolloutNext", 0))
        accepted = int(self._meta("rolloutAcceptedVisits", 0))

        # Older v7.1 checkpoints stored rolloutsSeeded=True as soon as the
        # then-current rollout target was reached.  That made later "resume with
        # a larger --rollouts/--node-limit" runs no-op, even though README says
        # coverage can be increased without rebuilding.  Treat rolloutNext as the
        # authoritative progress counter: if the new target is larger, continue
        # from the saved deterministic rollout number.
        if start_rollout >= int(self.options.rollouts):
            self._set_meta("rolloutAttempts", int(self.options.rollouts))
            self._set_meta("rolloutsSeeded", True)
            self.connection.commit()
            return

        previous_target = int(self._meta("rolloutAttempts", 0))
        if previous_target and int(self.options.rollouts) > previous_target:
            print(
                f"Synthetic reachability: extending playout target "
                f"from {previous_target:,} to {int(self.options.rollouts):,}; "
                f"resuming at {start_rollout:,}",
                flush=True,
            )
        elif self._meta("rolloutsSeeded", False):
            # Defensive repair for checkpoints that have rolloutsSeeded=True but
            # a smaller rolloutNext than the requested target.
            self._set_meta("rolloutsSeeded", False)
            self.connection.commit()

        unique_before = self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        completed = start_rollout
        for rollout in range(start_rollout, int(self.options.rollouts)):
            if time.monotonic() >= self.deadline - 5.0:
                break
            rng = random.Random(self.options.random_seed + rollout * 0x9E3779B1)
            board, turn = parse_fen(layouts[rollout % len(layouts)])
            for ply in range(self.options.rollout_plies):
                moves, count = legal_moves(board, turn)
                if count == 0:
                    break
                weighted = []
                for move_number in range(count):
                    move = moves[move_number]
                    successor = apply_move(board, move)
                    weight = 1.0
                    if int(move[3]):
                        weight += 9.0 + PIECE_VALUE.get(abs(int(move[3])), 0) / 130.0
                    if int(move[2]):
                        weight += 14.0
                    if in_check(successor, -turn):
                        weight += 4.0
                    if self._piece_count(successor) <= 6:
                        weight += 18.0
                    if abs(int(board[int(move[0])])) == PAWN:
                        weight += 1.5
                    weighted.append(weight)
                chosen = rng.choices(range(count), weights=weighted, k=1)[0]
                board = apply_move(board, moves[chosen])
                turn = -turn
                pieces = self._piece_count(board)
                if self.options.exact_core_pieces < pieces <= 6 and valid_position(board, turn):
                    balance = self._material_balance(board)
                    if balance <= 950 or rng.random() < 0.16:
                        priority = self._position_priority(board, 1, ply, source_bonus=10.0 - min(9.0, balance / 150.0))
                        self._insert_position(board, turn, priority, 1, 0)
                        accepted += 1
                if pieces <= self.options.exact_core_pieces:
                    break
            completed = rollout + 1
            if completed % 250 == 0:
                self._set_meta("rolloutNext", completed)
                self._set_meta("rolloutAcceptedVisits", accepted)
                self.connection.commit()
                print(f"Synthetic reachability: {completed:,}/{self.options.rollouts:,} playouts", flush=True)
        self._set_meta("rolloutNext", completed)
        self._set_meta("rolloutAttempts", int(self.options.rollouts))
        self._set_meta("rolloutAcceptedVisits", accepted)
        self._set_meta("rolloutsSeeded", bool(completed >= int(self.options.rollouts)))
        self.connection.commit()
        unique_after = self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        print(f"Synthetic seeds: {unique_after - unique_before:,} new unique positions this run ({completed:,}/{int(self.options.rollouts):,} playouts complete)", flush=True)

    def _node_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def expand_graph(self):
        """Expand the most useful reachable neighbourhood within the budget.

        Child existence is resolved once per parent with one indexed SQLite
        lookup instead of one query per legal move.  This matters for the
        practical profile: hundreds of thousands of nodes can otherwise turn
        into millions of Python/SQLite round trips without improving coverage.
        """
        reserve_seconds = min(900.0, max(5.0, self.options.hours * 3600 * 0.20))
        expansion_deadline = self.deadline - reserve_seconds
        processed = 0
        total_nodes = self._node_count()
        # Reaching the node cap stops *new* insertions, not expansion. Existing
        # stored nodes still need terminal classification and known-child edges;
        # the v7 prototype stopped here too early and left most of its budgeted
        # graph unexpanded.
        while time.monotonic() < expansion_deadline:
            row = self.connection.execute(
                "SELECT id,material_id,pos_index,priority,distance FROM nodes WHERE expanded=0 ORDER BY priority DESC,distance,id LIMIT 1"
            ).fetchone()
            if row is None:
                break
            node_id, material_id, pos_index, priority, distance = row
            spec = self.material_specs[int(material_id)]
            board, turn = unindex_sparse(spec, int(pos_index))
            moves, move_count = legal_moves(board, turn)
            terminal_state = None
            terminal_dtm = 0
            if move_count == 0:
                terminal_state = LOSS if in_check(board, turn) else DRAW
            external_win_count = 0
            external_max_dtm = 0
            external_loss_dtm = None
            external_loss_move = 0
            frontier = 0
            hint_move = 0
            hint_score = -1

            # First classify every legal edge. Lower-material edges are probed
            # from the exhaustive core; same-layer children are collected and
            # looked up in one batch below.
            pending: list[tuple[int, float, int, int, int]] = []
            unique_keys: list[tuple[int, int]] = []
            seen_keys: set[tuple[int, int]] = set()
            for move_number in range(move_count):
                move = moves[move_number]
                move_code = pack_move(move)
                successor = apply_move(board, move)
                score = self._move_hint_score(board, move, successor)
                if score > hint_score:
                    hint_score = score
                    hint_move = move_code
                successor_turn = -turn
                child_spec, _ = MaterialSpec.from_board(successor)
                if self.core.has(child_spec):
                    child_wdl, child_dtm = self.core.probe(successor, successor_turn)
                    if child_wdl == WIN:
                        external_win_count += 1
                        external_max_dtm = max(external_max_dtm, int(child_dtm))
                    elif child_wdl == LOSS:
                        candidate = int(child_dtm) + 1
                        if external_loss_dtm is None or candidate < external_loss_dtm:
                            external_loss_dtm = candidate
                            external_loss_move = move_code
                    continue
                if child_spec.piece_count <= self.options.exact_core_pieces:
                    # A missing required dependency is unknown, never silently
                    # draw or win. It blocks any false LOSS proof.
                    frontier += 1
                    continue

                child_priority = float(priority) * 0.985 - 1.0 - self._material_balance(successor) / 5000.0
                child_distance = int(distance) + 1
                child_canonical_spec, child_index = canonical_sparse_position(successor, successor_turn)
                child_material_id = self._material_id(child_canonical_spec)
                key = (child_material_id, int(child_index))
                pending.append((move_code, child_priority, child_distance, key[0], key[1]))
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_keys.append(key)

            existing_by_key: dict[tuple[int, int], int] = {}
            if unique_keys:
                # 128 legal moves is the hard move buffer, so this remains well
                # below SQLite's normal parameter limit.
                clauses = " OR ".join("(material_id=? AND pos_index=?)" for _ in unique_keys)
                parameters = [value for key in unique_keys for value in key]
                for child_id, child_material_id, child_index in self.connection.execute(
                    f"SELECT id,material_id,pos_index FROM nodes WHERE {clauses}", parameters
                ):
                    existing_by_key[(int(child_material_id), int(child_index))] = int(child_id)

            updates_by_id: dict[int, tuple[float, int]] = {}
            for move_code, child_priority, child_distance, child_material_id, child_index in pending:
                key = (child_material_id, child_index)
                child_id = existing_by_key.get(key)
                if child_id is None:
                    if total_nodes >= self.options.node_limit:
                        frontier += 1
                        continue
                    cursor = self.connection.execute(
                        "INSERT INTO nodes(material_id,pos_index,priority,frequency,distance) VALUES(?,?,?,?,?)",
                        (child_material_id, child_index, child_priority, 0, child_distance),
                    )
                    child_id = int(cursor.lastrowid)
                    existing_by_key[key] = child_id
                    total_nodes += 1
                else:
                    previous = updates_by_id.get(child_id)
                    if previous is None:
                        updates_by_id[child_id] = (child_priority, child_distance)
                    else:
                        updates_by_id[child_id] = (max(previous[0], child_priority), min(previous[1], child_distance))
                self.connection.execute(
                    "INSERT OR IGNORE INTO edges(parent,move,child) VALUES(?,?,?)",
                    (int(node_id), int(move_code), int(child_id)),
                )

            if updates_by_id:
                self.connection.executemany(
                    "UPDATE nodes SET priority=MAX(priority,?),distance=MIN(distance,?) WHERE id=?",
                    [(values[0], values[1], child_id) for child_id, values in updates_by_id.items()],
                )

            self.connection.execute(
                """UPDATE nodes SET expanded=1,degree=?,frontier=?,external_win_count=?,external_max_dtm=?,
                   external_loss_dtm=?,external_loss_move=?,terminal_state=?,terminal_dtm=?,
                   hint_move=?,hint_score=MAX(hint_score,?) WHERE id=?""",
                (
                    int(move_count), int(frontier), int(external_win_count), int(external_max_dtm),
                    external_loss_dtm, int(external_loss_move), terminal_state, terminal_dtm,
                    int(hint_move), int(max(0, hint_score)), int(node_id),
                ),
            )
            processed += 1
            if processed % self.options.commit_every == 0:
                self.connection.commit()
                expanded = self.connection.execute("SELECT COUNT(*) FROM nodes WHERE expanded=1").fetchone()[0]
                print(f"Practical graph: {expanded:,} expanded / {total_nodes:,} stored", flush=True)
        self.connection.commit()
        if processed:
            self._set_meta("retroInitialized", False)
        self._set_meta("graphNodeLimit", self.options.node_limit)
        self._set_meta("graphExpanded", int(self.connection.execute("SELECT COUNT(*) FROM nodes WHERE expanded=1").fetchone()[0]))
        self._set_meta("graphNodes", self._node_count())
        self.connection.commit()

    def initialize_retrograde(self, reset: bool = False):
        if self._meta("retroInitialized", False) and not reset:
            return
        self.connection.execute("DELETE FROM retro_queue")
        self.connection.execute(
            """UPDATE nodes SET state=9,dtm=0,best_move=0,
               remaining=degree-external_win_count,max_child_dtm=external_max_dtm"""
        )
        self.connection.execute(
            "UPDATE nodes SET state=terminal_state,dtm=terminal_dtm WHERE expanded=1 AND terminal_state IS NOT NULL"
        )
        self.connection.execute(
            """UPDATE nodes SET state=1,dtm=external_loss_dtm,best_move=external_loss_move
               WHERE expanded=1 AND terminal_state IS NULL AND external_loss_dtm IS NOT NULL"""
        )
        self.connection.execute(
            """INSERT OR REPLACE INTO retro_queue(node_id,state,dtm)
               SELECT id,state,dtm FROM nodes WHERE expanded=1 AND state IN (-1,1)"""
        )
        self._set_meta("retroInitialized", True)
        self.connection.commit()

    def retrograde(self, batch_size: int = 800):
        self.initialize_retrograde()
        processed = 0
        while time.monotonic() < self.deadline:
            batch = self.connection.execute(
                "SELECT node_id,state,dtm FROM retro_queue ORDER BY dtm,node_id LIMIT ?",
                (batch_size,),
            ).fetchall()
            if not batch:
                break
            child_map = {int(node_id): (int(state), int(dtm)) for node_id, state, dtm in batch}
            ids = list(child_map)
            placeholders = ",".join("?" for _ in ids)
            predecessor_rows = self.connection.execute(
                f"SELECT parent,move,child FROM edges WHERE child IN ({placeholders})",
                ids,
            ).fetchall()
            effects: dict[int, dict] = {}
            for parent, move, child in predecessor_rows:
                child_state, child_dtm = child_map[int(child)]
                effect = effects.setdefault(int(parent), {"win": None, "winMove": 0, "decrement": 0, "max": 0})
                if child_state == LOSS:
                    candidate = child_dtm + 1
                    if effect["win"] is None or candidate < effect["win"]:
                        effect["win"] = candidate
                        effect["winMove"] = int(move)
                elif child_state == WIN:
                    effect["decrement"] += 1
                    effect["max"] = max(effect["max"], child_dtm)

            new_queue = []
            if effects:
                parent_ids = list(effects)
                for start in range(0, len(parent_ids), 900):
                    chunk = parent_ids[start:start + 900]
                    marks = ",".join("?" for _ in chunk)
                    rows = self.connection.execute(
                        f"SELECT id,state,remaining,max_child_dtm FROM nodes WHERE id IN ({marks})",
                        chunk,
                    ).fetchall()
                    for parent_id, state, remaining, max_child in rows:
                        if int(state) != UNKNOWN:
                            continue
                        effect = effects[int(parent_id)]
                        if effect["win"] is not None:
                            next_state = WIN
                            next_dtm = int(effect["win"])
                            next_move = int(effect["winMove"])
                            self.connection.execute(
                                "UPDATE nodes SET state=1,dtm=?,best_move=? WHERE id=? AND state=9",
                                (next_dtm, next_move, int(parent_id)),
                            )
                            new_queue.append((int(parent_id), next_state, next_dtm))
                            continue
                        next_remaining = max(0, int(remaining) - int(effect["decrement"]))
                        next_max = max(int(max_child), int(effect["max"]))
                        if next_remaining == 0:
                            next_state = LOSS
                            next_dtm = next_max + 1
                            self.connection.execute(
                                "UPDATE nodes SET state=-1,dtm=?,remaining=0,max_child_dtm=? WHERE id=? AND state=9",
                                (next_dtm, next_max, int(parent_id)),
                            )
                            new_queue.append((int(parent_id), next_state, next_dtm))
                        else:
                            self.connection.execute(
                                "UPDATE nodes SET remaining=?,max_child_dtm=? WHERE id=? AND state=9",
                                (next_remaining, next_max, int(parent_id)),
                            )
            self.connection.executemany("DELETE FROM retro_queue WHERE node_id=?", [(node_id,) for node_id in ids])
            self.connection.executemany(
                "INSERT OR REPLACE INTO retro_queue(node_id,state,dtm) VALUES(?,?,?)",
                new_queue,
            )
            self.connection.commit()
            processed += len(batch)
            if processed % 8000 < batch_size:
                solved = self.connection.execute("SELECT COUNT(*) FROM nodes WHERE state IN (-1,0,1)").fetchone()[0]
                pending = self.connection.execute("SELECT COUNT(*) FROM retro_queue").fetchone()[0]
                print(f"Retrograde: {processed:,} events, {solved:,} proved nodes, {pending:,} queued", flush=True)
        self._set_meta("retroProcessedLastRun", processed)
        self.connection.commit()

    def refine_distances(self):
        """Tighten verified mate upper bounds with bulk graph relaxations."""
        for pass_number in range(self.options.refine_passes):
            if time.monotonic() >= self.deadline:
                break
            changed = 0
            winner_rows = self.connection.execute(
                """SELECT parent,candidate,move FROM (
                       SELECT e.parent AS parent,c.dtm+1 AS candidate,e.move AS move,
                              ROW_NUMBER() OVER(PARTITION BY e.parent ORDER BY c.dtm+1,e.move) AS rn
                       FROM edges e JOIN nodes c ON c.id=e.child JOIN nodes p ON p.id=e.parent
                       WHERE p.state=1 AND c.state=-1
                   ) WHERE rn=1"""
            ).fetchall()
            for parent, candidate, move in winner_rows:
                row = self.connection.execute(
                    "SELECT dtm,external_loss_dtm,external_loss_move FROM nodes WHERE id=?",
                    (int(parent),),
                ).fetchone()
                old_dtm, external_dtm, external_move = row
                best_dtm = int(candidate)
                best_move = int(move)
                if external_dtm is not None and int(external_dtm) <= best_dtm:
                    best_dtm = int(external_dtm)
                    best_move = int(external_move or 0)
                if best_dtm < int(old_dtm):
                    self.connection.execute("UPDATE nodes SET dtm=?,best_move=? WHERE id=?", (best_dtm, best_move, int(parent)))
                    changed += 1
            loser_rows = self.connection.execute(
                """SELECT p.id,p.dtm,p.external_max_dtm,COALESCE(MAX(c.dtm),0)
                   FROM nodes p LEFT JOIN edges e ON e.parent=p.id LEFT JOIN nodes c ON c.id=e.child AND c.state=1
                   WHERE p.state=-1 GROUP BY p.id"""
            ).fetchall()
            for node_id, old_dtm, external_max, child_max in loser_rows:
                candidate = max(int(external_max), int(child_max)) + 1
                if candidate < int(old_dtm):
                    self.connection.execute("UPDATE nodes SET dtm=? WHERE id=?", (candidate, int(node_id)))
                    changed += 1
            self.connection.commit()
            print(f"DTM upper-bound refinement pass {pass_number + 1}: {changed:,} improvements", flush=True)
            if changed == 0:
                break

    def _core_bytes(self) -> int:
        total = 0
        for path in self.output.rglob("*"):
            if path.is_file() and "practical" not in path.parts:
                total += path.stat().st_size
        return total

    def export(self):
        """Export the highest-value verified records under a hard size cap.

        The first estimate is deliberately conservative, then the actual gzip
        directory size is measured.  If filesystem/metadata overhead still
        crosses the requested target, the lowest-priority tail is trimmed and
        the practical layer is rebuilt automatically.  The exact core is never
        deleted or weakened.
        """
        practical_root = self.output / "practical"
        core_bytes = self._core_bytes()
        if core_bytes > int(self.options.target_bytes):
            raise RuntimeError(
                f"The exact <= {self.options.exact_core_pieces} core alone is {core_bytes / (1024 * 1024):.2f} MiB, "
                f"above the requested {self.options.target_bytes / (1024 * 1024):.2f} MiB target."
            )
        solved = int(self.connection.execute(
            "SELECT COUNT(*) FROM nodes WHERE expanded=1 AND state IN (-1,0,1)"
        ).fetchone()[0])
        coverage_mode = str(self.options.coverage_mode or "coverage").lower()
        if coverage_mode not in {"exact", "coverage"}:
            raise ValueError("coverage_mode must be 'exact' or 'coverage'.")
        candidate_where = "expanded=1 AND state IN (-1,0,1)" if coverage_mode == "exact" else "1=1"
        candidates = int(self.connection.execute(f"SELECT COUNT(*) FROM nodes WHERE {candidate_where}").fetchone()[0])

        # Start by exporting every candidate.  If that exceeds the hard cap,
        # trim the lowest-priority tail.  This fixes the v7.1 failure mode where
        # the generator claimed completion even though only a tiny solved subset
        # was available and the 96 MiB practical budget remained unused.
        selected = candidates

        def write_selection(limit: int):
            shutil.rmtree(practical_root, ignore_errors=True)
            (self.output / "practical-manifest.json").unlink(missing_ok=True)
            self.connection.execute("DROP TABLE IF EXISTS selected_export")
            if coverage_mode == "exact":
                where = "expanded=1 AND state IN (-1,0,1)"
            else:
                where = "1=1"
            self.connection.execute(
                f"""CREATE TEMP TABLE selected_export AS
                   SELECT id,material_id,pos_index,
                          CASE WHEN expanded=1 AND state IN (-1,0,1) THEN state ELSE 9 END AS export_state,
                          CASE WHEN expanded=1 AND state IN (-1,0,1) THEN dtm ELSE 0 END AS export_dtm,
                          CASE WHEN expanded=1 AND state IN (-1,0,1) AND best_move<>0 THEN best_move ELSE hint_move END AS export_move,
                          priority,frequency,distance
                   FROM nodes WHERE {where}
                   ORDER BY
                       CASE WHEN expanded=1 AND state IN (-1,0,1) THEN 0 ELSE 1 END,
                       priority DESC,frequency DESC,distance,id LIMIT ?""",
                (int(limit),),
            )
            self.connection.execute("CREATE INDEX selected_material_idx ON selected_export(material_id,pos_index)")
            tables = []
            material_rows = self.connection.execute(
                "SELECT DISTINCT material_id FROM selected_export ORDER BY material_id"
            ).fetchall()
            for (material_id,) in material_rows:
                spec = self.material_specs[int(material_id)]
                rows = [
                    (
                        int(pos_index),
                        pack_value(
                            int(state),
                            int(dtm),
                            int(best_move),
                            upper_bound=(int(state) != UNKNOWN),
                        ),
                    )
                    for pos_index, state, dtm, best_move in self.connection.execute(
                        "SELECT pos_index,export_state,export_dtm,export_move FROM selected_export WHERE material_id=? ORDER BY pos_index",
                        (int(material_id),),
                    )
                ]
                if rows:
                    tables.append(write_sparse_table(self.output, spec, rows, self.options.block_records))
            return tables

        # Usually one pass. The loop makes the size promise a real invariant,
        # not an optimistic estimate.
        attempts = 0
        tables = []
        while True:
            attempts += 1
            tables = write_selection(selected)
            preliminary_profile = {
                "name": "practical-v2.0-coverage",
                "exactCorePieces": int(self.options.exact_core_pieces),
                "sparsePieces": list(range(int(self.options.exact_core_pieces) + 1, 7)),
                "seedCorpus": self.options.seed_file.name,
                "nodeLimit": self.options.node_limit,
                "graphNodes": self._node_count(),
                "graphExpanded": int(self.connection.execute("SELECT COUNT(*) FROM nodes WHERE expanded=1").fetchone()[0]),
                "provedNodes": solved,
                "coverageCandidates": candidates,
                "exportedRecords": selected,
                "coverageMode": coverage_mode,
                "targetBytes": self.options.target_bytes,
                "coreBytes": core_bytes,
                "selection": "reachable PGN positions + capture-biased legal playouts + material-shell legal coverage + priority-expanded legal neighbourhood",
                "guarantee": "Solved records have proved WDL. UNKNOWN coverage records are search hints only and must fall back to normal engine search.",
            }
            update_practical_manifest(self.output, tables, preliminary_profile)
            final_bytes = sum(path.stat().st_size for path in self.output.rglob("*") if path.is_file())
            if final_bytes <= int(self.options.target_bytes) or selected == 0:
                fill_ratio = final_bytes / max(1, int(self.options.target_bytes))
                profile = preliminary_profile | {
                    "finalBytes": final_bytes,
                    "fillRatio": fill_ratio,
                    "minFillRatio": float(self.options.min_fill_ratio),
                    "underfilled": bool(fill_ratio < float(self.options.min_fill_ratio)),
                    "sizeTrimPasses": attempts - 1,
                    "indexEncoding": "delta-u32+gzip",
                }
                update_practical_manifest(self.output, tables, profile)
                break
            practical_bytes = max(1, final_bytes - core_bytes)
            allowed_practical = max(0, int(self.options.target_bytes) - core_bytes - (256 << 10))
            next_selected = int(selected * allowed_practical / practical_bytes * 0.94)
            if next_selected >= selected:
                next_selected = selected - max(1, selected // 20)
            selected = max(0, next_selected)
            print(
                f"Practical export exceeded target ({final_bytes / (1024 * 1024):.2f} MiB); "
                f"trimming to {selected:,} highest-priority records.",
                flush=True,
            )

        self._set_meta("lastExport", profile)
        self.connection.commit()
        solved_exported = int(self.connection.execute("SELECT COUNT(*) FROM selected_export WHERE export_state IN (-1,0,1)").fetchone()[0]) if selected else 0
        unknown_exported = max(0, int(selected) - solved_exported)
        print(f"Exported {selected:,} practical records ({solved_exported:,} proved, {unknown_exported:,} covered-unknown) across {len(tables):,} material tables", flush=True)
        print(f"Final table directory: {profile['finalBytes'] / (1024 * 1024):.2f} MiB (hard target {self.options.target_bytes / (1024 * 1024):.0f} MiB; fill {profile['fillRatio']:.1%})", flush=True)
        if profile.get("underfilled"):
            print("Not enough meaningful records were generated to fill the requested practical target yet. Rerun with a larger --node-limit/--shell-seeds/--hours, or set --allow-underfilled for a small trial build.", flush=True)
        return profile

    def run(self):
        self.seed_from_corpus()
        self.seed_from_rollouts()
        self.seed_from_material_shells()
        self.expand_graph()
        self.retrograde()
        self.refine_distances()
        return self.export()
