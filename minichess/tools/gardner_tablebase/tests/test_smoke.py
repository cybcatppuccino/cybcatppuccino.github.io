from __future__ import annotations

import tempfile
from pathlib import Path

from gtb.board import parse_fen
from gtb.generator import GeneratorOptions, MaterialGenerator
from gtb.material import MaterialSpec, dependency_closure
from gtb.storage import Tablebase


def test_kqvk_generation():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        options = GeneratorOptions(root / "work", root / "tables", block_size=4096, init_chunk=4096, retro_batch=2048)
        for spec in dependency_closure([MaterialSpec.parse("KQvK")]):
            MaterialGenerator(spec, options).run()
        tb = Tablebase(root / "tables")
        board, turn = parse_fen("4k/5/5/5/KQ3 w - - 0 1")
        wdl, dtm = tb.probe(board, turn)
        assert wdl == 1
        assert dtm > 0
