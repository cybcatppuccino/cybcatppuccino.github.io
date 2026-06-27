from __future__ import annotations

import gzip
import json
from pathlib import Path

from gtb.board import apply_move, in_check, legal_moves, move_to_uci, parse_fen
from gtb.constants import WIN
from gtb.practical import PracticalGenerator, PracticalOptions
from gtb.practical_storage import PracticalTablebase, unpack_move
from gtb.symmetry import canonical_sparse_position, mirror_files


def test_sparse_symmetry_and_verified_mate(tmp_path: Path):
    seed = tmp_path / "seed.json.gz"
    fen = "k4/2Q2/1K3/4p/4R w - - 0 1"
    with gzip.open(seed, "wt", encoding="utf-8") as handle:
        json.dump({"positions": [{"fen": fen, "frequency": 10, "minPly": 40, "sources": ["test"]}]}, handle)

    output = tmp_path / "tables"
    work = tmp_path / "work"
    generator = PracticalGenerator(
        PracticalOptions(
            output_root=output,
            work_root=work,
            seed_file=seed,
            node_limit=100,
            rollouts=0,
            rollout_plies=0,
            hours=0.02,
            target_bytes=2 * 1024 * 1024,
            block_records=16,
            commit_every=4,
            refine_passes=1,
            rebuild=True,
        )
    )
    try:
        profile = generator.run()
    finally:
        generator.close()

    assert profile["provedNodes"] > 0
    board, turn = parse_fen(fen)
    tablebase = PracticalTablebase(output)
    result = tablebase.probe(board, turn)
    assert result["wdl"] == WIN
    assert result["dtmPly"] == 1
    move = unpack_move(result["bestMove"])
    next_board = apply_move(board, move)
    _, count = legal_moves(next_board, -turn)
    assert count == 0 and in_check(next_board, -turn)

    mirrored = mirror_files(board)
    mirrored_result = tablebase.probe(mirrored, turn)
    assert mirrored_result["wdl"] == WIN
    mirrored_move = unpack_move(mirrored_result["bestMove"])
    assert move_to_uci(mirrored_move) != move_to_uci(move)
    mirrored_next = apply_move(mirrored, mirrored_move)
    _, mirrored_count = legal_moves(mirrored_next, -turn)
    assert mirrored_count == 0 and in_check(mirrored_next, -turn)

    assert canonical_sparse_position(board, turn) == canonical_sparse_position(mirrored, turn)


def test_practical_resume(tmp_path: Path):
    seed = tmp_path / "seed.json.gz"
    with gzip.open(seed, "wt", encoding="utf-8") as handle:
        json.dump({"positions": [{"fen": "k4/2Q2/1K3/4p/4R w - - 0 1", "frequency": 1, "minPly": 1}]}, handle)
    options = PracticalOptions(
        output_root=tmp_path / "tables",
        work_root=tmp_path / "work",
        seed_file=seed,
        node_limit=40,
        rollouts=0,
        rollout_plies=0,
        hours=0.01,
        target_bytes=1024 * 1024,
        block_records=16,
        commit_every=2,
        refine_passes=0,
        rebuild=True,
    )
    first = PracticalGenerator(options)
    try:
        first.seed_from_corpus()
        first.expand_graph()
        count_before = first._node_count()
    finally:
        first.close()

    options.rebuild = False
    second = PracticalGenerator(options)
    try:
        second.seed_from_corpus()
        assert second._node_count() == count_before
        second.retrograde()
        profile = second.export()
        assert profile["graphNodes"] == count_before
    finally:
        second.close()


def test_frontier_is_unknown_not_false_loss(tmp_path: Path):
    """A capped sparse graph must never relabel unseen legal moves as wins."""
    seed = tmp_path / "seed.json.gz"
    fen = "k4/5/2q2/2R2/K3R w - - 0 1"
    with gzip.open(seed, "wt", encoding="utf-8") as handle:
        json.dump({"positions": [{"fen": fen, "frequency": 1, "minPly": 20}]}, handle)
    output = tmp_path / "tables"
    generator = PracticalGenerator(
        PracticalOptions(
            output_root=output,
            work_root=tmp_path / "work",
            seed_file=seed,
            node_limit=1,
            rollouts=0,
            rollout_plies=0,
            hours=0.005,
            target_bytes=1024 * 1024,
            block_records=8,
            commit_every=1,
            refine_passes=0,
            rebuild=True,
        )
    )
    try:
        profile = generator.run()
    finally:
        generator.close()
    assert profile["finalBytes"] <= profile["targetBytes"]
    board, turn = parse_fen(fen)
    try:
        PracticalTablebase(output).probe(board, turn)
    except KeyError:
        pass
    else:
        raise AssertionError("An unresolved frontier position was exported as if proved")


def test_four_piece_sparse_layer_with_fast_three_piece_core_profile(tmp_path: Path):
    seed = tmp_path / "seed.json.gz"
    fen = "k4/2Q2/1K3/5/4R w - - 0 1"
    with gzip.open(seed, "wt", encoding="utf-8") as handle:
        json.dump({"positions": [{"fen": fen, "frequency": 5, "minPly": 35}]}, handle)
    output = tmp_path / "tables"
    generator = PracticalGenerator(
        PracticalOptions(
            output_root=output,
            work_root=tmp_path / "work",
            seed_file=seed,
            exact_core_pieces=3,
            node_limit=120,
            rollouts=0,
            rollout_plies=0,
            hours=0.02,
            target_bytes=1024 * 1024,
            block_records=16,
            commit_every=8,
            refine_passes=1,
            rebuild=True,
        )
    )
    try:
        profile = generator.run()
    finally:
        generator.close()
    assert profile["exactCorePieces"] == 3
    assert 4 in profile["sparsePieces"]
    board, turn = parse_fen(fen)
    result = PracticalTablebase(output).probe(board, turn)
    assert result["wdl"] == WIN and result["dtmPly"] == 1
