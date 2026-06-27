from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

from .board import apply_move, legal_moves, move_to_uci, parse_fen, to_fen
from .constants import DRAW, LOSS, UNKNOWN, WIN
from .generator import GeneratorOptions, MaterialGenerator
from .material import MaterialSpec, all_materials, dependency_closure
from .presets import get_preset, preset_names, preset_plan, preset_targets
from .storage import Tablebase
from .practical import DEFAULT_TARGET_BYTES, PracticalGenerator, PracticalOptions
from .practical_storage import PracticalTablebase, unpack_move


def human_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = 0
    while value >= 1024 and index < len(units) - 1:
        value /= 1024
        index += 1
    return f"{value:.2f} {units[index]}"


def selected_targets(args) -> list[MaterialSpec]:
    if getattr(args, "preset", None):
        return preset_targets(args.preset)
    if getattr(args, "all_up_to", None):
        return all_materials(args.all_up_to)
    return [MaterialSpec.parse(item) for item in args.target]


def selected_plan(args) -> list[MaterialSpec]:
    if getattr(args, "all_up_to", None):
        return all_materials(args.all_up_to)
    if getattr(args, "preset", None):
        return preset_plan(args.preset)
    return dependency_closure(selected_targets(args))


def estimate_plan(plan: list[MaterialSpec]):
    raw = sum(spec.raw_count for spec in plan)
    max_table = max((spec.raw_count for spec in plan), default=0)
    # Four memmaps (6 bytes/slot) plus a conservative 4-byte queue entry.
    # Completed checkpoints are deleted by default, so peak work space follows
    # the largest material table. --keep-work retains the full-plan amount.
    temporary_peak = max_table * 10
    temporary_keep_all = raw * 10
    packed_wdl = math.ceil(raw / 4)
    dtm_raw = raw * 2
    return {
        "tables": len(plan),
        "rawSlots": raw,
        "largestRawTable": max_table,
        "temporaryPeakBytes": temporary_peak,
        "temporaryKeepAllBytes": temporary_keep_all,
        "uncompressedOutputBytes": packed_wdl + dtm_raw,
    }


def command_estimate(args):
    plan = selected_plan(args)
    report = estimate_plan(plan)
    print(f"Material tables: {report['tables']:,}")
    print(f"Raw indexed slots: {report['rawSlots']:,}")
    print(f"Largest single table: {report['largestRawTable']:,} slots")
    print(f"Peak resumable checkpoint disk (default cleanup): {human_bytes(report['temporaryPeakBytes'])}")
    print(f"Checkpoint disk with --keep-work: {human_bytes(report['temporaryKeepAllBytes'])}")
    print(f"Uncompressed packed WDL+DTM output: {human_bytes(report['uncompressedOutputBytes'])}")
    print("Actual gzip output depends heavily on legality, draws and DTM structure.")
    if args.details:
        for spec in plan:
            print(f"{spec.signature:14} pieces={spec.piece_count} pawns={spec.pawn_count} raw={spec.raw_count:,}")


def command_plan(args):
    plan = selected_targets(args) if getattr(args, "targets_only", False) else selected_plan(args)
    for spec in plan:
        print(spec.signature)


def command_presets(args):
    for name in preset_names():
        preset = get_preset(name)
        targets = preset.target_specs()
        closure = preset.closure()
        print(f"{name}: {preset.title}")
        print(f"  {preset.description}")
        print(f"  targets ({len(targets)}): {', '.join(spec.signature for spec in targets)}")
        print(f"  dependency closure: {len(closure)} tables; "
              f"5-piece={sum(1 for spec in closure if spec.piece_count == 5)}, "
              f"<=4-piece={sum(1 for spec in closure if spec.piece_count <= 4)}")


def _completed_missing(plan: list[MaterialSpec], output: Path) -> tuple[list[MaterialSpec], list[MaterialSpec]]:
    complete: list[MaterialSpec] = []
    missing: list[MaterialSpec] = []
    for spec in plan:
        if (output / spec.signature / "metadata.json").exists():
            complete.append(spec)
        else:
            missing.append(spec)
    return complete, missing


def command_generate(args):
    plan = selected_plan(args)
    output = Path(args.output).resolve()
    work = Path(args.work).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output}")
    print(f"Checkpoint work: {work}")
    print(f"Plan: {len(plan)} material tables")
    start = time.time()
    for number, spec in enumerate(plan, start=1):
        metadata = output / spec.signature / "metadata.json"
        if metadata.exists() and not args.rebuild:
            print(f"[{number}/{len(plan)}] {spec.signature}: already complete, skipped")
            continue
        if args.rebuild:
            shutil.rmtree(output / spec.signature, ignore_errors=True)
            shutil.rmtree(work / spec.signature, ignore_errors=True)
        print(f"[{number}/{len(plan)}] {spec.signature}: {spec.raw_count:,} raw slots")
        options = GeneratorOptions(
            work_root=work,
            output_root=output,
            block_size=args.block_size,
            init_chunk=args.init_chunk,
            retro_batch=args.retro_batch,
            keep_work=args.keep_work,
            dependency_cache_mb=getattr(args, "dependency_cache_mb", 512),
        )
        MaterialGenerator(spec, options).run()
    print(f"Finished in {(time.time() - start) / 3600:.2f} hours.")


def command_watchdog_exact_generate(args):
    plan = selected_plan(args)
    output = Path(args.output).resolve()
    work = Path(args.work).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    print("Exact-table watchdog enabled: completed metadata is never rebuilt unless --rebuild is explicitly supplied.", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Checkpoint work: {work}", flush=True)
    print(f"Plan: {len(plan)} material tables", flush=True)
    if getattr(args, "preset", None):
        targets = selected_targets(args)
        print(f"Preset {args.preset}: {len(targets)} requested 5-piece targets; dependency closure will be generated as needed.", flush=True)
    restarts = 0
    pass_number = 0
    while True:
        pass_number += 1
        complete, missing = _completed_missing(plan, output)
        print(f"\nExact watchdog pass {pass_number}: {len(complete)}/{len(plan)} complete, {len(missing)} missing.", flush=True)
        if not missing:
            print("Exact watchdog accepted the build: all requested/dependency metadata files exist.", flush=True)
            return 0
        caught_exception = None
        exit_code = 0
        try:
            command_generate(args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            caught_exception = exc
            exit_code = 99
            print(
                "Exact watchdog caught a resumable generator error. The committed checkpoint is retained; "
                "the watchdog will inspect current progress and then resume after the configured sleep.",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
        # Rebuild must be one-shot; later watchdog passes resume instead of deleting finished work.
        args.rebuild = False
        complete, missing = _completed_missing(plan, output)
        print(f"Exact watchdog status: {len(complete)}/{len(plan)} complete, {len(missing)} missing.", flush=True)
        if not missing:
            print("Exact watchdog accepted the build. Running verify is safe now.", flush=True)
            return 0
        if exit_code not in (0,):
            if caught_exception is not None and bool(getattr(args, "watchdog_ignore_errors", True)):
                print(
                    f"Exact watchdog: generator raised {caught_exception.__class__.__name__}; "
                    "treating it as resumable and continuing from the last committed checkpoint.",
                    flush=True,
                )
            else:
                print(f"Exact watchdog stopped because generate returned fatal exit code {exit_code}.", file=sys.stderr)
                return int(exit_code)
        restarts += 1
        if int(args.watchdog_max_restarts) > 0 and restarts > int(args.watchdog_max_restarts):
            print(
                "Exact watchdog restart limit reached before all requested material tables completed. "
                "The checkpoint is preserved; rerun the same command to continue.",
                file=sys.stderr,
            )
            return 3
        sleep_seconds = max(0.0, float(args.watchdog_sleep))
        preview = ", ".join(spec.signature for spec in missing[:10])
        if len(missing) > 10:
            preview += f", ... +{len(missing) - 10} more"
        print(f"Exact watchdog: still missing {preview}. Sleeping {sleep_seconds:.1f}s, then resuming.", flush=True)
        time.sleep(sleep_seconds)


def command_probe(args):
    board, turn = parse_fen(args.fen)
    tablebase = Tablebase(args.tables, max_cached_blocks=args.cache_blocks)
    value, dtm = tablebase.probe(board, turn)
    names = {LOSS: "loss", DRAW: "draw", WIN: "win", UNKNOWN: "unknown"}
    print(json.dumps({"fen": to_fen(board, turn), "wdl": names[value], "dtmPly": dtm}, indent=2))
    if args.moves:
        candidates, count = legal_moves(board, turn)
        rows = []
        for index in range(count):
            move = candidates[index]
            successor = apply_move(board, move)
            try:
                child_wdl, child_dtm = tablebase.probe(successor, -turn)
            except (KeyError, ValueError):
                continue
            rows.append({
                "move": move_to_uci(move),
                "resultForMover": {LOSS: "win", DRAW: "draw", WIN: "loss"}[child_wdl],
                "dtmPly": child_dtm + 1 if child_wdl != DRAW else 0,
            })
        priority = {"win": 0, "draw": 1, "loss": 2}
        rows.sort(key=lambda row: (priority[row["resultForMover"]], row["dtmPly"] if row["resultForMover"] == "win" else -row["dtmPly"]))
        print(json.dumps(rows, indent=2))


def command_verify(args):
    root = Path(args.tables)
    tablebase = Tablebase(root)
    checked = 0
    for signature, entry in tablebase.manifest.get("tables", {}).items():
        metadata = tablebase.metadata(MaterialSpec.parse(signature))
        table_dir = root / signature
        for block in metadata["blocks"]:
            for field, checksum_field in (("wdl", "wdlSha256"), ("dtm", "dtmSha256")):
                import hashlib
                digest = hashlib.sha256((table_dir / block[field]).read_bytes()).hexdigest()
                if digest != block[checksum_field]:
                    raise SystemExit(f"Checksum mismatch: {signature}/{block[field]}")
                checked += 1
    practical_path = root / "practical-manifest.json"
    if practical_path.exists():
        practical = json.loads(practical_path.read_text(encoding="utf-8"))
        import hashlib
        for signature, entry in practical.get("tables", {}).items():
            metadata = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
            table_dir = root / "practical" / signature
            for block in metadata.get("blocks", []):
                for field, checksum_field in (("indices", "indexSha256"), ("values", "valueSha256")):
                    digest = hashlib.sha256((table_dir / block[field]).read_bytes()).hexdigest()
                    if digest != block[checksum_field]:
                        raise SystemExit(f"Checksum mismatch: practical/{signature}/{block[field]}")
                    checked += 1
    print(f"Verified {checked} compressed block files.")


def command_clean(args):
    path = Path(args.work).resolve()
    if not path.exists():
        print("No checkpoint directory exists.")
        return
    if args.signature:
        spec = MaterialSpec.parse(args.signature)
        target = path / spec.signature
        shutil.rmtree(target, ignore_errors=True)
        print(f"Removed checkpoint {target}")
    elif args.all:
        shutil.rmtree(path)
        print(f"Removed {path}")
    else:
        raise SystemExit("Use --signature MATERIAL or --all.")


def add_selection(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", action="append", help="Target material, e.g. KQvK. Repeat for multiple targets.")
    group.add_argument("--all-up-to", type=int, choices=range(2, 7), help="All canonical material classes up to this many pieces.")
    group.add_argument("--preset", choices=preset_names(), help="Curated target preset, e.g. common-5 for practical exact five-piece endings.")



def parse_bytes(text: str) -> int:
    value = text.strip().upper().replace("IB", "B")
    multipliers = {"B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2, "G": 1024**3, "GB": 1024**3}
    for suffix in ("GB", "MB", "KB", "G", "M", "K", "B"):
        if value.endswith(suffix):
            number = value[:-len(suffix)] or "0"
            return int(float(number) * multipliers[suffix])
    return int(float(value))


def project_seed_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "data" / "practical-seeds.json.gz",          # standalone package
        here.parents[3] / "data" / "practical-seeds.json.gz",          # inside minichess/tools/gardner_tablebase
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_exact_core(output: Path, work: Path, args) -> None:
    plan = all_materials(int(args.core_pieces))
    for number, spec in enumerate(plan, start=1):
        metadata = output / spec.signature / "metadata.json"
        if metadata.exists() and not args.rebuild_core:
            continue
        if args.rebuild_core:
            shutil.rmtree(output / spec.signature, ignore_errors=True)
            shutil.rmtree(work / spec.signature, ignore_errors=True)
        print(f"[exact core {number}/{len(plan)}] {spec.signature}: {spec.raw_count:,} raw slots")
        options = GeneratorOptions(
            work_root=work,
            output_root=output,
            block_size=args.block_size,
            init_chunk=args.init_chunk,
            retro_batch=args.retro_batch,
            keep_work=False,
            dependency_cache_mb=getattr(args, "dependency_cache_mb", 512),
        )
        MaterialGenerator(spec, options).run()


def command_quick_estimate(args):
    core = estimate_plan(all_materials(int(args.core_pieces)))
    target = parse_bytes(args.target_size)
    print("Practical v2 profile")
    print(f"  Exact core: every legal 2-{args.core_pieces} piece position (WDL+DTM)")
    print(f"  Coverage layer: {int(args.core_pieces)+1}-6 piece practical positions from PGNs, rollouts and material-shell sampling")
    print(f"  Graph node cap: {args.node_limit:,}")
    print(f"  Final directory hard target: {human_bytes(target)}")
    print(f"  Exact-core uncompressed maximum: {human_bytes(core['uncompressedOutputBytes'])}")
    print(f"  Exact-core peak checkpoint arrays: {human_bytes(core['temporaryPeakBytes'])}")
    print(f"  Practical target is a real fill target in v2; if the generated coverage is too small, quick-generate reports UNDERFILLED instead of pretending completion.")
    print(f"  Rough sparse allowance after core: {max(0, (target-core['uncompressedOutputBytes'])//8):,}+ records before compression/metadata.")
    print("  Probe-time memory: normally 2-20 MiB depending on lazy block cache")
    print("  Generator process memory: commonly 0.5-1.0 GiB because Python/Numba JIT itself is not part of the tablebase footprint")
    print("  Typical total time: v2 defaults are intentionally heavier; expect many hours, and rerun to resume/enlarge until the fill target is met.")


def command_quick_generate(args):
    output = Path(args.output).resolve()
    work = Path(args.work).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    target_bytes = parse_bytes(args.target_size)
    started = time.time()
    print(f"Output: {output}")
    print(f"Checkpoint work: {work}")
    print(f"Final size target: {human_bytes(target_bytes)}")
    ensure_exact_core(output, work, args)
    elapsed_hours = (time.time() - started) / 3600.0
    remaining_hours = max(0.08, float(args.hours) - elapsed_hours)
    seed_file = Path(args.seed_file).resolve() if args.seed_file else project_seed_path()
    options = PracticalOptions(
        output_root=output,
        work_root=work,
        seed_file=seed_file,
        exact_core_pieces=args.core_pieces,
        node_limit=args.node_limit,
        rollouts=args.rollouts,
        rollout_plies=args.rollout_plies,
        hours=remaining_hours,
        target_bytes=target_bytes,
        block_records=args.practical_block_records,
        commit_every=args.commit_every,
        random_seed=args.random_seed,
        refine_passes=args.refine_passes,
        shell_seeds=args.shell_seeds,
        coverage_mode=args.coverage_mode,
        min_fill_ratio=args.min_fill_ratio,
        allow_underfilled=args.allow_underfilled,
        rebuild=args.rebuild_practical,
    )
    generator = PracticalGenerator(options)
    try:
        profile = generator.run()
    finally:
        generator.close()
    print(json.dumps(profile, indent=2))
    print(f"Total elapsed: {(time.time() - started) / 3600:.2f} hours")
    if profile.get("underfilled") and not args.allow_underfilled:
        print("UNDERFILLED: the run is valid as a checkpoint, but not accepted as a full 96M practical build. Rerun the same command with more --hours, or increase --node-limit/--shell-seeds.", file=sys.stderr)
        return 2
    return 0



def _directory_file_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _read_practical_status(output: Path, target_bytes: int, min_fill_ratio: float) -> dict:
    output = Path(output).resolve()
    manifest_path = output / "practical-manifest.json"
    profile = {}
    manifest_exists = manifest_path.exists()
    if manifest_exists:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            profile = dict(manifest.get("profile", {}) or {})
        except Exception as exc:  # keep watchdog alive even if a partial write was observed
            profile = {"manifestReadError": str(exc), "underfilled": True}
    actual_bytes = _directory_file_bytes(output)
    reported_bytes = int(profile.get("finalBytes", actual_bytes) or actual_bytes)
    measured_bytes = max(actual_bytes, reported_bytes)
    fill_ratio = measured_bytes / max(1, int(target_bytes))
    reported_underfilled = bool(profile.get("underfilled", True))
    reached = bool(manifest_exists and not reported_underfilled and fill_ratio >= float(min_fill_ratio))
    status = {
        "manifestExists": manifest_exists,
        "actualBytes": actual_bytes,
        "reportedBytes": reported_bytes,
        "measuredBytes": measured_bytes,
        "fillRatio": fill_ratio,
        "targetBytes": int(target_bytes),
        "minFillRatio": float(min_fill_ratio),
        "reportedUnderfilled": reported_underfilled,
        "reached": reached,
        "graphNodes": int(profile.get("graphNodes", 0) or 0),
        "graphExpanded": int(profile.get("graphExpanded", 0) or 0),
        "provedNodes": int(profile.get("provedNodes", 0) or 0),
        "coverageCandidates": int(profile.get("coverageCandidates", 0) or 0),
        "exportedRecords": int(profile.get("exportedRecords", 0) or 0),
    }
    if "manifestReadError" in profile:
        status["manifestReadError"] = profile["manifestReadError"]
    return status


def _watchdog_progress_key(status: dict) -> tuple:
    return (
        int(status.get("measuredBytes", 0)),
        int(status.get("graphNodes", 0)),
        int(status.get("graphExpanded", 0)),
        int(status.get("provedNodes", 0)),
        int(status.get("coverageCandidates", 0)),
        int(status.get("exportedRecords", 0)),
    )


def _grow_watchdog_budget(args) -> None:
    factor = max(1.01, float(args.watchdog_grow_factor))
    old_node_limit = int(args.node_limit)
    old_rollouts = int(args.rollouts)
    old_shell_seeds = int(args.shell_seeds)
    args.node_limit = max(
        old_node_limit + int(args.watchdog_grow_min_nodes),
        int(math.ceil(old_node_limit * factor)),
    )
    args.rollouts = max(
        old_rollouts + int(args.watchdog_grow_min_rollouts),
        int(math.ceil(old_rollouts * factor)),
    )
    args.shell_seeds = max(
        old_shell_seeds + int(args.watchdog_grow_min_shell_seeds),
        int(math.ceil(old_shell_seeds * factor)),
    )
    print(
        "Watchdog: no additional table growth was detected; increasing resume budgets "
        f"node-limit {old_node_limit:,}->{args.node_limit:,}, "
        f"rollouts {old_rollouts:,}->{args.rollouts:,}, "
        f"shell-seeds {old_shell_seeds:,}->{args.shell_seeds:,}.",
        flush=True,
    )


def command_watchdog_generate(args):
    target_bytes = parse_bytes(args.target_size)
    watchdog_min_fill = float(args.watchdog_min_fill_ratio)
    # Make quick-generate itself stricter under watchdog, so a small table cannot
    # be reported as accepted merely because the old quick-generate threshold was looser.
    args.min_fill_ratio = max(float(args.min_fill_ratio), watchdog_min_fill)
    print("Watchdog enabled: false COMPLETE/UNDERFILLED results will sleep 1s and resume from the same checkpoint.", flush=True)
    print(
        f"Watchdog acceptance: >= {watchdog_min_fill:.1%} of {human_bytes(target_bytes)} "
        f"and practical-manifest underfilled=false.",
        flush=True,
    )
    if args.watchdog_estimate:
        command_quick_estimate(argparse.Namespace(
            core_pieces=args.core_pieces,
            node_limit=args.node_limit,
            target_size=args.target_size,
        ))
    pass_number = 0
    restarts = 0
    last_key = None
    stale_passes = 0
    while True:
        pass_number += 1
        print(f"\nWatchdog pass {pass_number}: running/resuming quick-generate...", flush=True)
        caught_exception = None
        try:
            exit_code = command_quick_generate(args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            caught_exception = exc
            exit_code = 99
            print(
                "Watchdog caught a resumable generator error. The committed checkpoint is retained; "
                "the watchdog will inspect current progress and then resume after the configured sleep.",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
        # Rebuild flags are intentionally one-shot. If the first pass did not
        # satisfy the watchdog, later passes must resume the checkpoint instead
        # of deleting the work and starting over.
        args.rebuild_core = False
        args.rebuild_practical = False
        status = _read_practical_status(Path(args.output), target_bytes, watchdog_min_fill)
        print(
            "Watchdog status: "
            f"size={human_bytes(status['measuredBytes'])}/{human_bytes(target_bytes)} "
            f"fill={status['fillRatio']:.1%}, "
            f"nodes={status['graphNodes']:,}, exported={status['exportedRecords']:,}, "
            f"underfilled={status['reportedUnderfilled']}",
            flush=True,
        )
        if status.get("manifestReadError"):
            print(f"Watchdog: manifest read was incomplete/corrupt this pass: {status['manifestReadError']}", flush=True)
        if status["reached"]:
            print("Watchdog accepted the build. Running final verify is safe now.", flush=True)
            return 0
        if exit_code not in (0, 2):
            if caught_exception is not None and bool(getattr(args, "watchdog_ignore_errors", True)):
                print(
                    f"Watchdog: quick-generate raised {caught_exception.__class__.__name__}; "
                    "treating it as resumable and continuing from the last committed checkpoint.",
                    flush=True,
                )
            else:
                print(f"Watchdog stopped because quick-generate returned fatal exit code {exit_code}.", file=sys.stderr)
                return int(exit_code)
        progress_key = _watchdog_progress_key(status)
        if last_key == progress_key:
            stale_passes += 1
        else:
            stale_passes = 0
            last_key = progress_key
        if args.watchdog_grow and stale_passes >= int(args.watchdog_stall_passes):
            _grow_watchdog_budget(args)
            stale_passes = 0
            last_key = None
        restarts += 1
        if int(args.watchdog_max_restarts) > 0 and restarts > int(args.watchdog_max_restarts):
            print(
                "Watchdog restart limit reached before the requested fill target. "
                "The checkpoint is preserved; rerun the same command to continue.",
                file=sys.stderr,
            )
            return 3
        sleep_seconds = max(0.0, float(args.watchdog_sleep))
        print(f"Watchdog: not accepted yet; sleeping {sleep_seconds:.1f}s, then resuming immediately.", flush=True)
        time.sleep(sleep_seconds)

def command_quick_probe(args):
    board, turn = parse_fen(args.fen)
    tablebase = PracticalTablebase(args.tables, max_cached_blocks=args.cache_blocks)
    result = tablebase.probe(board, turn)
    names = {LOSS: "loss", DRAW: "draw", WIN: "win", UNKNOWN: "unknown"}
    result = dict(result)
    result["wdl"] = names[result["wdl"]]
    if result.get("bestMove"):
        result["bestMoveUci"] = move_to_uci(unpack_move(result["bestMove"]))
    print(json.dumps(result, indent=2))


def command_quick_status(args):
    root = Path(args.work).resolve() / "practical" / "practical.sqlite3"
    if not root.exists():
        print("No practical checkpoint exists yet.")
        return
    import sqlite3
    connection = sqlite3.connect(root)
    rows = dict(connection.execute("SELECT key,value FROM meta"))
    counts = {
        "nodes": connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
        "expanded": connection.execute("SELECT COUNT(*) FROM nodes WHERE expanded=1").fetchone()[0],
        "proved": connection.execute("SELECT COUNT(*) FROM nodes WHERE state IN (-1,0,1)").fetchone()[0],
        "queued": connection.execute("SELECT COUNT(*) FROM retro_queue").fetchone()[0],
        "edges": connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
    }
    connection.close()
    print(json.dumps({"checkpoint": str(root), "counts": counts, "meta": {k: json.loads(v) for k,v in rows.items()}}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="gardner-tb", description="Resumable Gardner 5x5 WDL/DTM tablebase generator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    presets = subparsers.add_parser("presets", help="List curated material presets.")
    presets.set_defaults(func=command_presets)

    estimate = subparsers.add_parser("estimate", help="Estimate indexed state and disk requirements.")
    add_selection(estimate)
    estimate.add_argument("--details", action="store_true")
    estimate.set_defaults(func=command_estimate)

    plan = subparsers.add_parser("plan", help="Print the dependency-ordered material plan.")
    add_selection(plan)
    plan.add_argument("--targets-only", action="store_true", help="For presets/targets, print requested targets without dependency closure.")
    plan.set_defaults(func=command_plan)

    generate = subparsers.add_parser("generate", help="Generate tables, resuming checkpoints automatically.")
    add_selection(generate)
    generate.add_argument("--output", default="tables", help="Final lazy-load table directory.")
    generate.add_argument("--work", default="work", help="Checkpoint and temporary directory.")
    generate.add_argument("--block-size", type=int, default=1 << 18)
    generate.add_argument("--init-chunk", type=int, default=100_000)
    generate.add_argument("--retro-batch", type=int, default=50_000)
    generate.add_argument("--dependency-cache-mb", type=int, default=512, help="Soft RAM budget for full dependency tables during exact generation; 0 disables the cap.")
    generate.add_argument("--keep-work", action="store_true")
    generate.add_argument("--rebuild", action="store_true")
    generate.set_defaults(func=command_generate)

    watchdog_exact = subparsers.add_parser("watchdog-exact-generate", help="Generate exact material tables under a resumable watchdog; use --preset common-5 for curated five-piece endings.")
    add_selection(watchdog_exact)
    watchdog_exact.add_argument("--output", default="tables", help="Final lazy-load table directory.")
    watchdog_exact.add_argument("--work", default="work", help="Checkpoint and temporary directory.")
    watchdog_exact.add_argument("--block-size", type=int, default=1 << 18)
    watchdog_exact.add_argument("--init-chunk", type=int, default=100_000)
    watchdog_exact.add_argument("--retro-batch", type=int, default=50_000)
    watchdog_exact.add_argument("--dependency-cache-mb", type=int, default=512, help="Soft RAM budget for full dependency tables during exact generation; 0 disables the cap.")
    watchdog_exact.add_argument("--keep-work", action="store_true")
    watchdog_exact.add_argument("--rebuild", action="store_true", help="Dangerous: rebuilds every material in the selected dependency plan. Avoid for common-5 unless you really want to overwrite existing core tables.")
    watchdog_exact.add_argument("--watchdog-sleep", type=float, default=1.0, help="Seconds to rest before an automatic resume; default is 1s.")
    watchdog_exact.add_argument("--watchdog-max-restarts", type=int, default=0, help="0 means unlimited automatic resumes.")
    watchdog_exact.add_argument("--watchdog-stop-on-error", dest="watchdog_ignore_errors", action="store_false", help="Stop instead of auto-resuming when exact generation raises an exception.")
    watchdog_exact.set_defaults(func=command_watchdog_exact_generate, watchdog_ignore_errors=True)

    probe = subparsers.add_parser("probe", help="Probe a generated position.")
    probe.add_argument("--tables", default="tables")
    probe.add_argument("--fen", required=True)
    probe.add_argument("--moves", action="store_true", help="Also rank legal moves using child table probes.")
    probe.add_argument("--cache-blocks", type=int, default=16)
    probe.set_defaults(func=command_probe)

    verify = subparsers.add_parser("verify", help="Verify compressed block checksums.")
    verify.add_argument("--tables", default="tables")
    verify.set_defaults(func=command_verify)

    clean = subparsers.add_parser("clean", help="Remove checkpoint work after a completed or abandoned table.")
    clean.add_argument("--work", default="work")
    clean.add_argument("--signature")
    clean.add_argument("--all", action="store_true")
    clean.set_defaults(func=command_clean)

    quick_estimate = subparsers.add_parser("quick-estimate", help="Estimate the under-100-MiB practical tablebase profile.")
    quick_estimate.add_argument("--core-pieces", type=int, choices=(3, 4), default=4, help="Exhaustive core size. v2 defaults to 4 for materially better tactical coverage.")
    quick_estimate.add_argument("--node-limit", type=int, default=12_000_000)
    quick_estimate.add_argument("--target-size", default="96M")
    quick_estimate.set_defaults(func=command_quick_estimate)

    quick_generate = subparsers.add_parser("quick-generate", help="Generate the fast practical exact-core + sparse 5/6-piece database.")
    quick_generate.add_argument("--output", default="tables")
    quick_generate.add_argument("--work", default="work")
    quick_generate.add_argument("--hours", type=float, default=8.0, help="Approximate total wall-clock budget including the exact core.")
    quick_generate.add_argument("--target-size", default="96M", help="Hard final directory target, e.g. 96M.")
    quick_generate.add_argument("--core-pieces", type=int, choices=(3, 4), default=4, help="Exhaustive core size. v2 defaults to 4 for stronger real-game coverage.")
    quick_generate.add_argument("--node-limit", type=int, default=12_000_000)
    quick_generate.add_argument("--rollouts", type=int, default=50_000)
    quick_generate.add_argument("--rollout-plies", type=int, default=100)
    quick_generate.add_argument("--seed-file")
    quick_generate.add_argument("--random-seed", type=int, default=0x47544271)
    quick_generate.add_argument("--refine-passes", type=int, default=3)
    quick_generate.add_argument("--shell-seeds", type=int, default=12_000_000, help="Legal material-shell samples for broad practical 4-6 piece coverage.")
    quick_generate.add_argument("--coverage-mode", choices=("coverage", "exact"), default="coverage", help="coverage exports solved records plus UNKNOWN search-hint records; exact exports solved records only.")
    quick_generate.add_argument("--min-fill-ratio", type=float, default=0.85, help="Minimum acceptable final-size/target-size ratio before the command reports UNDERFILLED.")
    quick_generate.add_argument("--allow-underfilled", action="store_true", help="Return success even when final tables are below the requested fill ratio.")
    quick_generate.add_argument("--commit-every", type=int, default=100)
    quick_generate.add_argument("--practical-block-records", type=int, default=65_536)
    quick_generate.add_argument("--block-size", type=int, default=1 << 18)
    quick_generate.add_argument("--init-chunk", type=int, default=100_000)
    quick_generate.add_argument("--retro-batch", type=int, default=50_000)
    quick_generate.add_argument("--rebuild-core", action="store_true")
    quick_generate.add_argument("--rebuild-practical", action="store_true")
    quick_generate.set_defaults(func=command_quick_generate)


    watchdog_generate = subparsers.add_parser("watchdog-generate", help="Run quick-generate under a watchdog that rejects false COMPLETE/UNDERFILLED exits and resumes after 1s.")
    watchdog_generate.add_argument("--output", default="tables")
    watchdog_generate.add_argument("--work", default="work")
    watchdog_generate.add_argument("--hours", type=float, default=8.0, help="Approximate wall-clock budget per watchdog pass.")
    watchdog_generate.add_argument("--target-size", default="96M", help="Hard final directory target, e.g. 96M.")
    watchdog_generate.add_argument("--core-pieces", type=int, choices=(3, 4), default=4, help="Exhaustive core size. v2 defaults to 4 for stronger real-game coverage.")
    watchdog_generate.add_argument("--node-limit", type=int, default=12_000_000)
    watchdog_generate.add_argument("--rollouts", type=int, default=50_000)
    watchdog_generate.add_argument("--rollout-plies", type=int, default=100)
    watchdog_generate.add_argument("--seed-file")
    watchdog_generate.add_argument("--random-seed", type=int, default=0x47544271)
    watchdog_generate.add_argument("--refine-passes", type=int, default=3)
    watchdog_generate.add_argument("--shell-seeds", type=int, default=12_000_000, help="Legal material-shell samples for broad practical 4-6 piece coverage.")
    watchdog_generate.add_argument("--coverage-mode", choices=("coverage", "exact"), default="coverage", help="coverage exports solved records plus UNKNOWN search-hint records; exact exports solved records only.")
    watchdog_generate.add_argument("--min-fill-ratio", type=float, default=0.95, help="Under watchdog, quick-generate is treated as underfilled below this ratio.")
    watchdog_generate.add_argument("--allow-underfilled", action="store_true", help="Kept for compatibility; watchdog still waits for its own fill threshold unless --watchdog-min-fill-ratio is lowered.")
    watchdog_generate.add_argument("--commit-every", type=int, default=100)
    watchdog_generate.add_argument("--practical-block-records", type=int, default=65_536)
    watchdog_generate.add_argument("--block-size", type=int, default=1 << 18)
    watchdog_generate.add_argument("--init-chunk", type=int, default=100_000)
    watchdog_generate.add_argument("--retro-batch", type=int, default=50_000)
    watchdog_generate.add_argument("--rebuild-core", action="store_true")
    watchdog_generate.add_argument("--rebuild-practical", action="store_true")
    watchdog_generate.add_argument("--watchdog-sleep", type=float, default=1.0, help="Seconds to rest before an automatic resume; default is 1s.")
    watchdog_generate.add_argument("--watchdog-max-restarts", type=int, default=0, help="0 means unlimited automatic resumes.")
    watchdog_generate.add_argument("--watchdog-min-fill-ratio", type=float, default=0.95, help="Actual measured tables/target ratio required before COMPLETE is accepted.")
    watchdog_generate.add_argument("--watchdog-stall-passes", type=int, default=1, help="Grow budgets after this many no-growth resumes.")
    watchdog_generate.add_argument("--watchdog-grow-factor", type=float, default=1.25)
    watchdog_generate.add_argument("--watchdog-grow-min-nodes", type=int, default=1_000_000)
    watchdog_generate.add_argument("--watchdog-grow-min-rollouts", type=int, default=10_000)
    watchdog_generate.add_argument("--watchdog-grow-min-shell-seeds", type=int, default=1_000_000)
    watchdog_generate.add_argument("--no-watchdog-grow", dest="watchdog_grow", action="store_false", help="Disable automatic budget growth on stagnant resumes.")
    watchdog_generate.add_argument("--watchdog-estimate", action="store_true", help="Print quick-estimate before the first watched pass.")
    watchdog_generate.add_argument("--watchdog-stop-on-error", dest="watchdog_ignore_errors", action="store_false", help="Stop instead of auto-resuming when quick-generate raises an exception.")
    watchdog_generate.set_defaults(func=command_watchdog_generate, watchdog_grow=True, watchdog_ignore_errors=True)

    quick_probe = subparsers.add_parser("quick-probe", help="Probe exact core or a verified practical sparse record.")
    quick_probe.add_argument("--tables", default="tables")
    quick_probe.add_argument("--fen", required=True)
    quick_probe.add_argument("--cache-blocks", type=int, default=12)
    quick_probe.set_defaults(func=command_quick_probe)

    quick_status = subparsers.add_parser("quick-status", help="Show resumable practical graph/retrograde progress.")
    quick_status.add_argument("--work", default="work")
    quick_status.set_defaults(func=command_quick_status)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Committed batches are preserved; rerun the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
