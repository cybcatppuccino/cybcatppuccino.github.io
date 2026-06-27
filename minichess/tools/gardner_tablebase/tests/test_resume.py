from __future__ import annotations

import tempfile
from pathlib import Path

from gtb.board import parse_fen
from gtb.generator import GeneratorOptions, MaterialGenerator
from gtb.material import MaterialSpec, dependency_closure
from gtb.storage import Tablebase


def test_write_ahead_log_resume_and_cleanup():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        options = GeneratorOptions(
            root / "work", root / "tables",
            block_size=4096, init_chunk=4096, retro_batch=2048,
        )
        plan = dependency_closure([MaterialSpec.parse("KQvK")])
        # Generate the dependency normally.
        MaterialGenerator(plan[0], options).run()

        generator = MaterialGenerator(plan[-1], options)
        original_apply = generator._apply_transaction
        crashed = False

        def crash_once(transaction):
            nonlocal crashed
            if not crashed:
                crashed = True
                raise RuntimeError("simulated power loss after durable WAL")
            return original_apply(transaction)

        generator._apply_transaction = crash_once
        try:
            generator.initialize()
        except RuntimeError as error:
            assert "simulated power loss" in str(error)
        finally:
            generator._close_arrays()

        assert (root / "work" / "KQvK" / "transaction.pkl").exists()

        # Construction replays the WAL; run then resumes the remaining batches.
        resumed = MaterialGenerator(plan[-1], options)
        resumed.run()
        assert not (root / "work" / "KQvK" / "transaction.pkl").exists()
        assert not (root / "work" / "KQvK" / "wdl.dat").exists()
        assert (root / "work" / "KQvK" / "status.json").exists()

        tablebase = Tablebase(root / "tables")
        board, turn = parse_fen("4k/5/5/5/KQ3 w - - 0 1")
        wdl, dtm = tablebase.probe(board, turn)
        assert wdl == 1
        assert dtm > 0
