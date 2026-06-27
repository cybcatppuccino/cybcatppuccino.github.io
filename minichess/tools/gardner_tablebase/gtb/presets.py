from __future__ import annotations

from dataclasses import dataclass

from .material import MaterialSpec, dependency_closure


@dataclass(frozen=True)
class PresetInfo:
    name: str
    title: str
    description: str
    targets: tuple[str, ...]

    def target_specs(self) -> list[MaterialSpec]:
        result: list[MaterialSpec] = []
        seen: set[str] = set()
        for item in self.targets:
            spec = MaterialSpec.parse(item)
            if spec.signature in seen:
                continue
            seen.add(spec.signature)
            result.append(spec)
        return result

    def closure(self) -> list[MaterialSpec]:
        return dependency_closure(self.target_specs())


# Curated exact five-piece targets for Gardner 5x5 endgames.
# Every target below is a true 5-piece material class: kings + (2 non-kings vs 1 non-king).
# The generator still expands the dependency closure automatically, because pawn promotion
# and capture transitions may require sibling five-piece classes and already-built <=4-piece tables.
COMMON_5_TARGETS: tuple[str, ...] = (
    # The examples requested by the user: balanced or slightly-superior but drawish fights.
    "KBNvKR",  # knight+bishop vs rook
    "KNPvKB",  # knight+pawn vs bishop
    "KBPvKN",  # bishop+pawn vs knight
    "KBRvKQ",  # bishop+rook vs queen
    "KRPvKR",  # rook+pawn vs rook
    "KPPvKP",  # two pawns vs pawn

    # Extra real-game practical shells with roughly balanced force.
    "KRNvKQ",  # rook+knight vs queen
    "KQPvKR",  # queen+pawn vs rook, often technical but not always trivial on 5x5
    "KRPvKB",  # rook+pawn vs bishop
    "KRPvKN",  # rook+pawn vs knight
    "KBPvKR",  # bishop+pawn vs rook: inferior-side drawing-resource shell
    "KNPvKR",  # knight+pawn vs rook: inferior-side drawing-resource shell
    "KNPvKP",  # knight+pawn vs pawn
    "KBPvKP",  # bishop+pawn vs pawn
    "KBNvKQ",  # two minors vs queen: defensive fortress/drawing-resource shell
)

COMMON_5_LITE_TARGETS: tuple[str, ...] = (
    "KBNvKR",
    "KBRvKQ",
    "KRPvKR",
    "KPPvKP",
    "KNPvKB",
    "KBPvKN",
)

NO_PAWN_5_TARGETS: tuple[str, ...] = (
    "KBNvKR",
    "KBRvKQ",
    "KRNvKQ",
    "KRBvKQ",
    "KBNvKQ",
    "KRBvKR",
    "KRNvKR",
)

PRESETS: dict[str, PresetInfo] = {
    "common-5": PresetInfo(
        name="common-5",
        title="Common practical exact five-piece endings",
        description=(
            "Curated 5-piece exact table targets where one side has two non-king pieces "
            "and the other side has one. It emphasizes balanced rook/queen/minor/pawn "
            "technical endings and includes the requested KBNvKR, KNPvKB, KBRvKQ, "
            "KRPvKR, KPPvKP and KBPvKN shells."
        ),
        targets=COMMON_5_TARGETS,
    ),
    "common-5-lite": PresetInfo(
        name="common-5-lite",
        title="Small exact five-piece starter set",
        description=(
            "A smaller starter target set containing the six requested/endgame-representative "
            "shells. The dependency closure is still exact and promotion-safe."
        ),
        targets=COMMON_5_LITE_TARGETS,
    ),
    "no-pawn-5": PresetInfo(
        name="no-pawn-5",
        title="No-pawn exact five-piece tactical endings",
        description=(
            "Pawn-free 5-piece targets. Useful for validating the exact five-piece pipeline "
            "without promotion sibling dependencies."
        ),
        targets=NO_PAWN_5_TARGETS,
    ),
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> PresetInfo:
    key = str(name).strip().lower()
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown material preset {name!r}. Available presets: {', '.join(preset_names())}") from exc


def preset_targets(name: str) -> list[MaterialSpec]:
    return get_preset(name).target_specs()


def preset_plan(name: str) -> list[MaterialSpec]:
    return get_preset(name).closure()
