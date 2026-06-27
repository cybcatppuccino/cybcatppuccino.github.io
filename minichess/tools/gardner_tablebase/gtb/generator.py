from __future__ import annotations

import json
import os
import pickle
import time
import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .board import apply_move, in_check, legal_moves, valid_position
from .constants import DRAW, INVALID, LOSS, UNKNOWN, WIN
from .indexer import PositionIndexer
from .material import MaterialSpec
from .storage import FlatTablebase, atomic_json, update_manifest, write_table
from .kernels import reverse_quiet_predecessor_indices


@dataclass
class GeneratorOptions:
    work_root: Path
    output_root: Path
    block_size: int = 1 << 18
    init_chunk: int = 20_000
    retro_batch: int = 50_000
    keep_work: bool = False
    dependency_cache_mb: int = 512


class MaterialGenerator:
    def __init__(self, spec: MaterialSpec, options: GeneratorOptions):
        self.spec = spec.canonical()[0]
        self.options = options
        self.indexer = PositionIndexer(self.spec)
        self.work = options.work_root / self.spec.signature
        self.work.mkdir(parents=True, exist_ok=True)
        self.bucket_dir = self.work / "buckets"
        self.bucket_dir.mkdir(exist_ok=True)
        self.status_path = self.work / "status.json"
        self.wal_path = self.work / "transaction.pkl"
        self.dependencies = FlatTablebase(
            options.output_root,
            max_cached_blocks=24,
            max_flat_bytes=max(0, int(options.dependency_cache_mb)) * 1024 * 1024,
        )
        self.status = self._load_status()
        self._open_arrays()
        self._recover()

    def _load_status(self):
        if self.status_path.exists():
            status = json.loads(self.status_path.read_text(encoding="utf-8"))
            if status.get("signature") != self.spec.signature or int(status.get("rawCount", -1)) != self.spec.raw_count:
                raise RuntimeError("Checkpoint does not match the requested material.")
            return status
        status = {
            "signature": self.spec.signature,
            "rawCount": self.spec.raw_count,
            "stage": "init",
            "initNext": 0,
            "heads": {},
            "sizes": {},
            "lastTxn": 0,
            "processed": 0,
            "createdAt": time.time(),
        }
        atomic_json(self.status_path, status)
        return status

    def _array(self, name: str, dtype, fill):
        path = self.work / f"{name}.dat"
        mode = "r+" if path.exists() else "w+"
        array = np.memmap(path, dtype=dtype, mode=mode, shape=(self.spec.raw_count,))
        if mode == "w+":
            array[:] = fill
            array.flush()
        return array

    def _open_arrays(self):
        self.wdl = self._array("wdl", np.int8, UNKNOWN)
        self.degree = self._array("degree", np.uint8, 0)
        self.dtm = self._array("dtm", np.uint16, 0)
        self.max_dtm = self._array("max_dtm", np.uint16, 0)

    def _bucket_path(self, distance: int) -> Path:
        return self.bucket_dir / f"d{distance:05d}.bin"

    def _truncate_buckets(self):
        known = {int(key): int(value) for key, value in self.status.get("sizes", {}).items()}
        for path in self.bucket_dir.glob("d*.bin"):
            distance = int(path.stem[1:])
            size = known.get(distance, 0)
            with path.open("r+b") as handle:
                handle.truncate(size * 4)

    def _recover(self):
        self._truncate_buckets()
        if not self.wal_path.exists():
            return
        transaction = pickle.loads(self.wal_path.read_bytes())
        if int(self.status.get("lastTxn", 0)) >= int(transaction["id"]):
            self.wal_path.unlink(missing_ok=True)
            return
        self._apply_transaction(transaction)

    def _write_wal(self, transaction):
        temporary = self.wal_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(transaction, handle, protocol=5)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.wal_path)

    def _commit(self, updates: dict[int, tuple[int, int, int, int]], appends: dict[int, list[int]], status_changes: dict):
        transaction_id = int(self.status.get("lastTxn", 0)) + 1
        indices = np.fromiter(updates.keys(), dtype=np.uint32, count=len(updates))
        values = np.asarray(list(updates.values()), dtype=np.int64) if updates else np.empty((0, 4), dtype=np.int64)
        transaction = {
            "id": transaction_id,
            "indices": indices,
            "values": values,
            "appends": {int(distance): np.asarray(items, dtype=np.uint32) for distance, items in appends.items() if items},
            "status": status_changes,
            "baseSizes": {int(key): int(value) for key, value in self.status.get("sizes", {}).items()},
        }
        self._write_wal(transaction)
        self._apply_transaction(transaction)

    def _apply_transaction(self, transaction):
        indices = transaction["indices"]
        values = transaction["values"]
        if len(indices):
            self.wdl[indices] = values[:, 0].astype(np.int8)
            self.degree[indices] = values[:, 1].astype(np.uint8)
            self.dtm[indices] = values[:, 2].astype(np.uint16)
            self.max_dtm[indices] = values[:, 3].astype(np.uint16)
        base_sizes = {int(key): int(value) for key, value in transaction.get("baseSizes", {}).items()}
        next_sizes = dict(base_sizes)
        for distance, items in transaction.get("appends", {}).items():
            distance = int(distance)
            path = self._bucket_path(distance)
            old_size = base_sizes.get(distance, 0)
            with path.open("a+b") as handle:
                handle.truncate(old_size * 4)
                handle.seek(old_size * 4)
                handle.write(np.asarray(items, dtype="<u4").tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            next_sizes[distance] = old_size + len(items)
        self.wdl.flush(); self.degree.flush(); self.dtm.flush(); self.max_dtm.flush()
        self.status.update(transaction.get("status", {}))
        self.status["sizes"] = {str(key): value for key, value in next_sizes.items()}
        self.status["lastTxn"] = int(transaction["id"])
        atomic_json(self.status_path, self.status)
        self.wal_path.unlink(missing_ok=True)

    def _external_probe(self, board, turn):
        return self.dependencies.probe(board, turn)

    def initialize(self):
        if self.status["stage"] != "init":
            return
        total = self.spec.raw_count
        start = int(self.status.get("initNext", 0))
        while start < total:
            stop = min(total, start + self.options.init_chunk)
            updates: dict[int, tuple[int, int, int, int]] = {}
            appends: dict[int, list[int]] = {}
            for index in range(start, stop):
                board, turn = self.indexer.unindex(index)
                if not valid_position(board, turn):
                    updates[index] = (INVALID, 0, 0, 0)
                    continue
                moves, move_count = legal_moves(board, turn)
                if move_count == 0:
                    if in_check(board, turn):
                        updates[index] = (LOSS, 0, 0, 0)
                        appends.setdefault(0, []).append(index)
                    else:
                        updates[index] = (DRAW, 0, 0, 0)
                    continue
                remaining = int(move_count)
                best_win_dtm = None
                worst_loss_dtm = 0
                for move_index in range(move_count):
                    move = moves[move_index]
                    # Quiet non-promotion moves stay inside this material table.
                    # Avoid constructing a successor board and re-scanning its
                    # material for the overwhelmingly common internal edge.
                    if int(move[2]) == 0 and int(move[3]) == 0:
                        continue
                    successor = apply_move(board, move)
                    successor_turn = -turn
                    child_wdl, child_dtm = self._external_probe(successor, successor_turn)
                    if child_wdl == LOSS:
                        candidate = child_dtm + 1
                        best_win_dtm = candidate if best_win_dtm is None else min(best_win_dtm, candidate)
                    elif child_wdl == WIN:
                        remaining -= 1
                        worst_loss_dtm = max(worst_loss_dtm, child_dtm)
                if best_win_dtm is not None:
                    updates[index] = (WIN, remaining, best_win_dtm, worst_loss_dtm)
                    appends.setdefault(best_win_dtm, []).append(index)
                elif remaining == 0:
                    distance = worst_loss_dtm + 1
                    updates[index] = (LOSS, 0, distance, worst_loss_dtm)
                    appends.setdefault(distance, []).append(index)
                else:
                    updates[index] = (UNKNOWN, remaining, 0, worst_loss_dtm)
            self._commit(updates, appends, {"initNext": stop, "stage": "init" if stop < total else "retrograde"})
            start = stop
            print(f"[{self.spec.signature}] initialized {stop:,}/{total:,}", flush=True)

    def _next_bucket(self):
        sizes = {int(key): int(value) for key, value in self.status.get("sizes", {}).items()}
        heads = {int(key): int(value) for key, value in self.status.get("heads", {}).items()}
        available = [distance for distance, size in sizes.items() if heads.get(distance, 0) < size]
        return min(available) if available else None

    def retrograde(self):
        if self.status["stage"] not in ("retrograde", "finalize"):
            return
        self.status["stage"] = "retrograde"
        atomic_json(self.status_path, self.status)
        while True:
            distance = self._next_bucket()
            if distance is None:
                break
            heads = {int(key): int(value) for key, value in self.status.get("heads", {}).items()}
            sizes = {int(key): int(value) for key, value in self.status.get("sizes", {}).items()}
            head = heads.get(distance, 0)
            stop = min(sizes[distance], head + self.options.retro_batch)
            with self._bucket_path(distance).open("rb") as handle:
                handle.seek(head * 4)
                current_indices = np.frombuffer(handle.read((stop - head) * 4), dtype="<u4").copy()
            updates: dict[int, tuple[int, int, int, int]] = {}
            appends: dict[int, list[int]] = {}

            def current_values(index: int):
                if index in updates:
                    return updates[index]
                return int(self.wdl[index]), int(self.degree[index]), int(self.dtm[index]), int(self.max_dtm[index])

            for current_index in current_indices:
                current_index = int(current_index)
                outcome = int(self.wdl[current_index])
                if outcome not in (WIN, LOSS):
                    continue
                board, turn = self.indexer.unindex(current_index)
                predecessor_indices, predecessor_count = reverse_quiet_predecessor_indices(
                    board,
                    turn,
                    self.indexer.group_codes_array,
                    self.indexer.group_counts_array,
                    self.indexer.radices_array,
                )
                for predecessor_number in range(predecessor_count):
                    predecessor_index = int(predecessor_indices[predecessor_number])
                    pred_wdl, pred_degree, pred_dtm, pred_max = current_values(predecessor_index)
                    if pred_wdl != UNKNOWN:
                        continue
                    if outcome == LOSS:
                        next_distance = distance + 1
                        updates[predecessor_index] = (WIN, pred_degree, next_distance, pred_max)
                        appends.setdefault(next_distance, []).append(predecessor_index)
                    else:
                        next_degree = max(0, pred_degree - 1)
                        next_max = max(pred_max, distance)
                        if next_degree == 0:
                            next_distance = next_max + 1
                            updates[predecessor_index] = (LOSS, 0, next_distance, next_max)
                            appends.setdefault(next_distance, []).append(predecessor_index)
                        else:
                            updates[predecessor_index] = (UNKNOWN, next_degree, pred_dtm, next_max)
            heads[distance] = stop
            self._commit(
                updates,
                appends,
                {
                    "heads": {str(key): value for key, value in heads.items()},
                    "processed": int(self.status.get("processed", 0)) + len(current_indices),
                    "stage": "retrograde",
                },
            )
            print(f"[{self.spec.signature}] DTM {distance}: {stop:,}/{sizes[distance]:,} queue states", flush=True)
        self.status["stage"] = "finalize"
        atomic_json(self.status_path, self.status)


    def _close_arrays(self):
        for name in ("wdl", "degree", "dtm", "max_dtm"):
            array = getattr(self, name, None)
            if array is None:
                continue
            try:
                array.flush()
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            except Exception:
                pass
            setattr(self, name, None)
        gc.collect()

    def _discard_completed_work(self):
        """Remove large checkpoint payloads only after final files and status commit.

        Interrupted material classes keep every memmap and queue. Completed tables
        retain only status.json unless --keep-work is requested. This bounds peak
        checkpoint space to the largest material class instead of the whole plan.
        """
        self._close_arrays()
        for name in ("wdl.dat", "degree.dat", "dtm.dat", "max_dtm.dat", "transaction.pkl"):
            (self.work / name).unlink(missing_ok=True)
        shutil.rmtree(self.bucket_dir, ignore_errors=True)

    def finalize(self):
        if self.status["stage"] == "done":
            return
        if self.status["stage"] != "finalize":
            raise RuntimeError(f"Cannot finalize from stage {self.status['stage']}")
        chunk = 1 << 20
        for start in range(0, self.spec.raw_count, chunk):
            stop = min(self.spec.raw_count, start + chunk)
            view = self.wdl[start:stop]
            view[view == UNKNOWN] = DRAW
        self.wdl.flush()
        metadata = write_table(self.options.output_root, self.spec, self.wdl, self.dtm, self.options.block_size)
        update_manifest(self.options.output_root, metadata)
        self.status["stage"] = "done"
        self.status["finishedAt"] = time.time()
        atomic_json(self.status_path, self.status)
        print(f"[{self.spec.signature}] complete: {len(metadata['blocks'])} lazy-load blocks", flush=True)
        if not self.options.keep_work:
            self._discard_completed_work()

    def run(self):
        self.initialize()
        self.retrograde()
        self.finalize()
