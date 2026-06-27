from __future__ import annotations

import gzip
import json
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .constants import DRAW, LOSS, UNKNOWN, WIN
from .material import MaterialSpec
from .storage import Tablebase, atomic_json, sha256_file
from .symmetry import canonical_sparse_position_with_transform, transform_packed_move

PRACTICAL_FORMAT = "GardnerPracticalTB"
PRACTICAL_VERSION = 3


def pack_move(move) -> int:
    if move is None:
        return 0
    origin, target, promotion = int(move[0]), int(move[1]), int(move[2])
    return origin | (target << 5) | (promotion << 10)


def unpack_move(value: int) -> tuple[int, int, int]:
    return value & 31, (value >> 5) & 31, (value >> 10) & 7


def pack_value(state: int, dtm: int, best_move: int = 0, upper_bound: bool = True) -> int:
    # Codes 0/1/2 keep the v7.1 solved-record layout intact.  Code 3 is new
    # in v2 and means: covered practical position, but exact WDL is not proved
    # inside the selective graph.  It is a useful search/policy hint, never a
    # false draw.
    state_code = {LOSS: 0, DRAW: 1, WIN: 2, UNKNOWN: 3}[int(state)]
    distance = max(0, min(1023, int(dtm)))
    flags = 1 if upper_bound else 0
    return state_code | (distance << 2) | ((int(best_move) & 0xFFFF) << 12) | (flags << 28)


def unpack_value(value: int) -> dict:
    states = (LOSS, DRAW, WIN, UNKNOWN)
    state = states[value & 3]
    return {
        "wdl": state,
        "dtmPly": 0 if state == UNKNOWN else ((value >> 2) & 1023),
        "bestMove": (value >> 12) & 0xFFFF,
        "dtmUpperBound": bool((value >> 28) & 1) if state != UNKNOWN else False,
    }


def write_sparse_table(root: Path, spec: MaterialSpec, rows: list[tuple[int, int]], block_records: int = 65536) -> dict:
    table_dir = root / "practical" / spec.signature
    table_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda item: item[0])
    blocks = []
    for block_id, start in enumerate(range(0, len(rows), block_records)):
        portion = rows[start:start + block_records]
        indices = np.asarray([item[0] for item in portion], dtype="<u4")
        # Sparse combinatorial indexes are monotone. Delta coding before gzip
        # preserves exact random access after one cumsum while materially
        # reducing the final database footprint for practical clusters.
        encoded_indices = np.empty_like(indices)
        encoded_indices[0] = indices[0]
        if len(indices) > 1:
            encoded_indices[1:] = np.diff(indices)
        values = np.asarray([item[1] for item in portion], dtype="<u4")
        index_name = f"{block_id:05d}.idx.gz"
        value_name = f"{block_id:05d}.val.gz"
        index_path = table_dir / index_name
        value_path = table_dir / value_name
        with gzip.open(index_path, "wb", compresslevel=6) as handle:
            handle.write(encoded_indices.tobytes())
        with gzip.open(value_path, "wb", compresslevel=6) as handle:
            handle.write(values.tobytes())
        blocks.append({
            "id": block_id,
            "count": len(portion),
            "minIndex": int(indices[0]),
            "maxIndex": int(indices[-1]),
            "indices": index_name,
            "values": value_name,
            "indexBytes": index_path.stat().st_size,
            "valueBytes": value_path.stat().st_size,
            "indexSha256": sha256_file(index_path),
            "valueSha256": sha256_file(value_path),
        })
    metadata = {
        "format": PRACTICAL_FORMAT,
        "version": PRACTICAL_VERSION,
        "signature": spec.signature,
        "board": "5x5",
        "metric": "exact WDL for solved records + UNKNOWN search-hint coverage records",
        "pieceCount": spec.piece_count,
        "pawnCount": spec.pawn_count,
        "recordCount": len(rows),
        "recordEncoding": "delta-coded uint32 combinatorial index + packed uint32 value; WDL code 3 is UNKNOWN coverage",
        "indexEncoding": "delta-u32",
        "symmetry": "material colour canonicalization, file mirror, and colour-rotation where applicable",
        "blocks": blocks,
    }
    atomic_json(table_dir / "metadata.json", metadata)
    return metadata


def update_practical_manifest(root: Path, tables: list[dict], profile: dict) -> None:
    path = root / "practical-manifest.json"
    manifest = {
        "format": PRACTICAL_FORMAT,
        "version": PRACTICAL_VERSION,
        "board": "5x5",
        "metric": "exact WDL for solved records + UNKNOWN search-hint coverage records",
        "fiftyMoveRule": False,
        "profile": profile,
        "tables": {},
    }
    for metadata in tables:
        manifest["tables"][metadata["signature"]] = {
            "path": f"practical/{metadata['signature']}/metadata.json",
            "recordCount": metadata["recordCount"],
            "pieceCount": metadata["pieceCount"],
            "pawnCount": metadata["pawnCount"],
            "blocks": len(metadata["blocks"]),
            "compressedBytes": sum(block["indexBytes"] + block["valueBytes"] for block in metadata["blocks"]),
        }
    atomic_json(path, manifest)


class PracticalTablebase:
    """Probe the exact small core first, then sparse verified higher-piece records."""

    def __init__(self, root: str | Path, max_cached_blocks: int = 12):
        self.root = Path(root)
        self.core = Tablebase(self.root, max_cached_blocks=max_cached_blocks)
        path = self.root / "practical-manifest.json"
        self.manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"tables": {}}
        self.max_cached_blocks = max(1, int(max_cached_blocks))
        self._metadata: dict[str, dict] = {}
        self._cache: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _metadata_for(self, signature: str) -> dict:
        if signature not in self._metadata:
            entry = self.manifest.get("tables", {}).get(signature)
            if not entry:
                raise KeyError(signature)
            self._metadata[signature] = json.loads((self.root / entry["path"]).read_text(encoding="utf-8"))
        return self._metadata[signature]

    def _load_block(self, signature: str, block_id: int):
        key = (signature, block_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        metadata = self._metadata_for(signature)
        block = metadata["blocks"][block_id]
        table_dir = self.root / "practical" / signature
        with gzip.open(table_dir / block["indices"], "rb") as handle:
            encoded = np.frombuffer(handle.read(), dtype="<u4").copy()
        if metadata.get("indexEncoding") == "delta-u32":
            indices = np.cumsum(encoded, dtype=np.uint64).astype(np.uint32)
        else:  # Backward-compatible with the v1 sparse blocks.
            indices = encoded
        with gzip.open(table_dir / block["values"], "rb") as handle:
            values = np.frombuffer(handle.read(), dtype="<u4").copy()
        self._cache[key] = (indices, values)
        while len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return indices, values

    def probe(self, board, turn: int) -> dict:
        spec, _ = MaterialSpec.from_board(board)
        if self.core.has(spec):
            wdl, dtm = self.core.probe(board, turn)
            return {"wdl": wdl, "dtmPly": dtm, "bestMove": 0, "dtmUpperBound": False, "source": "exact-core"}
        canonical_spec, index, transform = canonical_sparse_position_with_transform(board, turn)
        entry = self.manifest.get("tables", {}).get(canonical_spec.signature)
        if not entry:
            raise KeyError(f"No practical record for {canonical_spec.signature}")
        metadata = self._metadata_for(canonical_spec.signature)
        maxima = [block["maxIndex"] for block in metadata["blocks"]]
        block_id = bisect_left(maxima, index)
        if block_id >= len(metadata["blocks"]):
            raise KeyError("Position not present in practical tablebase")
        block = metadata["blocks"][block_id]
        if index < block["minIndex"]:
            raise KeyError("Position not present in practical tablebase")
        indices, values = self._load_block(canonical_spec.signature, block_id)
        offset = int(np.searchsorted(indices, np.uint32(index)))
        if offset >= len(indices) or int(indices[offset]) != index:
            raise KeyError("Position not present in practical tablebase")
        result = unpack_value(int(values[offset]))
        result["bestMove"] = transform_packed_move(result["bestMove"], transform)
        result["source"] = "practical-verified" if result["wdl"] != UNKNOWN else "practical-covered-unknown"
        return result
