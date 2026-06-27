from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .constants import DRAW, FORMAT_VERSION, INVALID, LOSS, WIN
from .indexer import PositionIndexer
from .material import MaterialSpec, rotate_swap_board


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack_wdl(values: np.ndarray) -> bytes:
    mapping = np.empty(values.shape[0], dtype=np.uint8)
    mapping[values == LOSS] = 0
    mapping[values == DRAW] = 1
    mapping[values == WIN] = 2
    mapping[values == INVALID] = 3
    padding = (-mapping.size) % 4
    if padding:
        mapping = np.pad(mapping, (0, padding), constant_values=3)
    packed = mapping[0::4] | (mapping[1::4] << 2) | (mapping[2::4] << 4) | (mapping[3::4] << 6)
    return packed.tobytes()


def unpack_wdl(payload: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(packed.size * 4, dtype=np.int8)
    reverse = np.asarray((LOSS, DRAW, WIN, INVALID), dtype=np.int8)
    values[0::4] = reverse[packed & 3]
    values[1::4] = reverse[(packed >> 2) & 3]
    values[2::4] = reverse[(packed >> 4) & 3]
    values[3::4] = reverse[(packed >> 6) & 3]
    return values[:count]


def write_table(output_root: Path, spec: MaterialSpec, wdl: np.ndarray, dtm: np.ndarray, block_size: int) -> dict:
    table_dir = output_root / spec.signature
    table_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for block_id, start in enumerate(range(0, spec.raw_count, block_size)):
        stop = min(spec.raw_count, start + block_size)
        wdl_name = f"{block_id:06d}.wdl.gz"
        dtm_name = f"{block_id:06d}.dtm.gz"
        wdl_path = table_dir / wdl_name
        dtm_path = table_dir / dtm_name
        with gzip.open(wdl_path, "wb", compresslevel=6) as handle:
            handle.write(pack_wdl(np.asarray(wdl[start:stop], dtype=np.int8)))
        with gzip.open(dtm_path, "wb", compresslevel=6) as handle:
            handle.write(np.asarray(dtm[start:stop], dtype="<u2").tobytes())
        blocks.append({
            "id": block_id,
            "start": start,
            "count": stop - start,
            "wdl": wdl_name,
            "dtm": dtm_name,
            "wdlBytes": wdl_path.stat().st_size,
            "dtmBytes": dtm_path.stat().st_size,
            "wdlSha256": sha256_file(wdl_path),
            "dtmSha256": sha256_file(dtm_path),
        })
    metadata = {
        "format": "GardnerTB",
        "version": FORMAT_VERSION,
        "signature": spec.signature,
        "board": "5x5",
        "metric": "WDL+DTM",
        "fiftyMoveRule": False,
        "rawCount": spec.raw_count,
        "pieceCount": spec.piece_count,
        "pawnCount": spec.pawn_count,
        "blockSize": block_size,
        "wdlEncoding": "2-bit: loss=0 draw=1 win=2 invalid=3",
        "dtmEncoding": "little-endian uint16 plies; zero for draw/invalid",
        "blocks": blocks,
    }
    atomic_json(table_dir / "metadata.json", metadata)
    return metadata


def update_manifest(output_root: Path, metadata: dict) -> None:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "format": "GardnerTB",
            "version": FORMAT_VERSION,
            "board": "5x5",
            "metric": "WDL+DTM",
            "fiftyMoveRule": False,
            "tables": {},
        }
    manifest["tables"][metadata["signature"]] = {
        "path": f"{metadata['signature']}/metadata.json",
        "rawCount": metadata["rawCount"],
        "pieceCount": metadata["pieceCount"],
        "pawnCount": metadata["pawnCount"],
        "blocks": len(metadata["blocks"]),
        "compressedBytes": sum(block["wdlBytes"] + block["dtmBytes"] for block in metadata["blocks"]),
    }
    atomic_json(manifest_path, manifest)


class Tablebase:
    def __init__(self, root: str | Path, max_cached_blocks: int = 16):
        self.root = Path(root)
        self.max_cached_blocks = max(1, max_cached_blocks)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"tables": {}}
        self._metadata: dict[str, dict] = {}
        self._indexers: dict[str, PositionIndexer] = {}
        self._cache: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def has(self, signature: str | MaterialSpec) -> bool:
        spec = signature if isinstance(signature, MaterialSpec) else MaterialSpec.parse(signature)
        return spec.signature in self.manifest.get("tables", {})

    def metadata(self, spec: MaterialSpec) -> dict:
        signature = spec.signature
        if signature not in self._metadata:
            entry = self.manifest.get("tables", {}).get(signature)
            if not entry:
                raise KeyError(f"Missing table {signature}")
            self._metadata[signature] = json.loads((self.root / entry["path"]).read_text(encoding="utf-8"))
        return self._metadata[signature]

    def indexer(self, spec: MaterialSpec) -> PositionIndexer:
        if spec.signature not in self._indexers:
            self._indexers[spec.signature] = PositionIndexer(spec)
        return self._indexers[spec.signature]

    def _load_block(self, spec: MaterialSpec, block_id: int):
        key = (spec.signature, block_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        metadata = self.metadata(spec)
        block = metadata["blocks"][block_id]
        table_dir = self.root / spec.signature
        with gzip.open(table_dir / block["wdl"], "rb") as handle:
            wdl = unpack_wdl(handle.read(), block["count"])
        with gzip.open(table_dir / block["dtm"], "rb") as handle:
            dtm = np.frombuffer(handle.read(), dtype="<u2").copy()
        self._cache[key] = (wdl, dtm)
        while len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return wdl, dtm

    def probe(self, board, turn: int) -> tuple[int, int]:
        spec, swapped = MaterialSpec.from_board(board)
        metadata = self.metadata(spec)
        working = np.asarray(board, dtype=np.int8)
        working_turn = int(turn)
        if swapped:
            working = rotate_swap_board(working)
            working_turn = -working_turn
        index = self.indexer(spec).index_canonical(working, working_turn)
        block_id, offset = divmod(index, metadata["blockSize"])
        wdl, dtm = self._load_block(spec, block_id)
        value = int(wdl[offset])
        if value == INVALID:
            raise ValueError("The position is not legal in this tablebase.")
        return value, int(dtm[offset])


class FlatTablebase(Tablebase):
    """Tablebase variant that keeps whole dependency tables in memory.

    Exact generation probes lower-material and promotion-dependency tables from
    the initialize phase.  Lazy gzip block loading is memory-efficient for
    interactive probing, but for generation it can thrash badly because captures
    and promotions jump across many blocks.  This class loads a table once into
    contiguous numpy arrays, with a soft LRU byte budget, and otherwise exposes
    the same probe() interface as Tablebase.
    """

    def __init__(self, root: str | Path, max_cached_blocks: int = 16, max_flat_bytes: int = 512 * 1024 * 1024):
        super().__init__(root, max_cached_blocks=max_cached_blocks)
        self.max_flat_bytes = max(0, int(max_flat_bytes))
        self._flat_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        self._flat_bytes = 0

    def _load_flat(self, spec: MaterialSpec) -> tuple[np.ndarray, np.ndarray]:
        signature = spec.signature
        cached = self._flat_cache.pop(signature, None)
        if cached is not None:
            wdl, dtm, size = cached
            self._flat_cache[signature] = cached
            return wdl, dtm
        metadata = self.metadata(spec)
        wdl = np.empty(int(metadata["rawCount"]), dtype=np.int8)
        dtm = np.empty(int(metadata["rawCount"]), dtype=np.uint16)
        table_dir = self.root / signature
        for block in metadata["blocks"]:
            start = int(block["start"])
            stop = start + int(block["count"])
            with gzip.open(table_dir / block["wdl"], "rb") as handle:
                wdl[start:stop] = unpack_wdl(handle.read(), int(block["count"]))
            with gzip.open(table_dir / block["dtm"], "rb") as handle:
                dtm[start:stop] = np.frombuffer(handle.read(), dtype="<u2")
        size = int(wdl.nbytes + dtm.nbytes)
        self._flat_cache[signature] = (wdl, dtm, size)
        self._flat_bytes += size
        while self.max_flat_bytes and self._flat_bytes > self.max_flat_bytes and len(self._flat_cache) > 1:
            _, (_, _, old_size) = self._flat_cache.popitem(last=False)
            self._flat_bytes -= int(old_size)
        return wdl, dtm

    def probe(self, board, turn: int) -> tuple[int, int]:
        spec, swapped = MaterialSpec.from_board(board)
        working = np.asarray(board, dtype=np.int8)
        working_turn = int(turn)
        if swapped:
            working = rotate_swap_board(working)
            working_turn = -working_turn
        index = self.indexer(spec).index_canonical(working, working_turn)
        wdl, dtm = self._load_flat(spec)
        value = int(wdl[index])
        if value == INVALID:
            raise ValueError("The position is not legal in this tablebase.")
        return value, int(dtm[index])
