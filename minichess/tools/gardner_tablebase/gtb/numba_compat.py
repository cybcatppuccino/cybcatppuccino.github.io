from __future__ import annotations

import os
import sys


def _identity_njit(*args, **kwargs):
    """Small no-op replacement for numba.njit.

    It supports both @njit and @njit(cache=True, inline="always") usage.
    """
    if args and len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(function):
        return function

    return decorator


def _load_njit():
    if os.environ.get("GARDNER_DISABLE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}:
        if os.environ.get("GARDNER_REQUIRE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError("GARDNER_DISABLE_NUMBA=1 conflicts with GARDNER_REQUIRE_NUMBA=1")
        return _identity_njit
    try:
        from numba import njit as loaded_njit
        return loaded_njit
    except Exception as exc:  # catches ImportError plus numba import-time SyntaxError/IndentationError/SystemError
        if os.environ.get("GARDNER_REQUIRE_NUMBA", "").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        print(
            "WARNING: numba could not be imported cleanly "
            f"({exc.__class__.__name__}: {exc}). Falling back to pure Python njit shim. "
            "Generation can continue, but it may run slower. To force this mode set GARDNER_DISABLE_NUMBA=1; "
            "to fail fast set GARDNER_REQUIRE_NUMBA=1.",
            file=sys.stderr,
            flush=True,
        )
        return _identity_njit


njit = _load_njit()
