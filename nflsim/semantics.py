"""Feature-schema semantics shared by producers (features/training) and consumers (runtime/lineup)."""

from __future__ import annotations

import json
from pathlib import Path

ISOLATED_MU_V1 = "isolated_mu_v1"
ISOLATED_MU_CONTEXT_V2 = "isolated_mu_context_v2"
ISOLATED_MU_CONTEXT_V3 = "isolated_mu_context_v3"
ISOLATED_SEMANTICS = (ISOLATED_MU_V1, ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3)
LEGACY_SEMANTICS = "legacy_conditioned"

ISOLATED_MU_DEFINITION = {
    "player_metrics": "raw signed ridge subject coefficient from the joint weighted fit (no intercept, position, PV or clipping)",
    "defense_metrics": "raw signed ridge opposing-defense coefficient from the joint weighted fit",
    "unseen_entity": "zero shrunk coefficient",
    "separate_inputs": "venue/offense-team coefficients, weather and five raw PV columns enter Torch as separate columns",
    "pv_in_runtime": "lineup PV changes the five PV input columns only; entity mus are never shifted",
    "interpretation": "ridge-conditional shrunk attribution relative to the fitted intercept and penalties; not causal and not an absolute rate",
}

ISOLATED_MU_CONTEXT_V2_DEFINITION = {
    **ISOLATED_MU_DEFINITION,
    "paired_defense_metrics": "for targets with both subject and defense outputs, the defense output is the defense-block coefficient of the SAME subject fit (no separate defense fit)",
    "defense_only_metrics": "targets without a subject output (int_rate) keep their own defense fit",
    "opponent_defense_context": "ridge_<target>_opponent_defense_effect = actual opponent's same-fit defense coefficient (ridge_<target>_defense_mu) for targets without a paired defense output",
    "venue_context": "ridge_<target>_venue_effect from every offensive subject fit (player and team); offense-team effect for player fits",
    "never_folded": "context coefficients are separate Torch inputs; entity mus are never mutated",
}

ISOLATED_MU_CONTEXT_V3_DEFINITION = {
    **ISOLATED_MU_CONTEXT_V2_DEFINITION,
    "torch_inputs": "model-specific original non-Ridge features, directly involved player Ridge mus, venue/offense/opponent-defense Ridge effects, player-value terms, weather, and game state",
    "availability_flags": "not used; unavailable optional Ridge effects are neutral-zero filled",
    "duplicate_defense_aliases": "legacy defensive columns are removed only when the same numeric effect is present as an explicit opponent-defense Ridge input",
}


def venue_column(target: str) -> str:
    return f"ridge_{target}_venue_effect"


def offense_column(target: str) -> str:
    return f"ridge_{target}_offense_effect"


def opponent_defense_column(target: str) -> str:
    return f"ridge_{target}_opponent_defense_effect"


def context_available_column(column: str) -> str:
    return f"{column}_available"


def defense_mu_column(target: str) -> str:
    return f"ridge_{target}_defense_mu"


def opponent_defense_source(column: str) -> str | None:
    """Own-team defense-mu column that supplies an opponent-defense context column."""
    availability_suffix = "_opponent_defense_effect_available"
    if column.startswith("ridge_") and column.endswith(availability_suffix):
        target = column[len("ridge_"):-len(availability_suffix)]
        return context_available_column(defense_mu_column(target))
    suffix = "_opponent_defense_effect"
    if column.startswith("ridge_") and column.endswith(suffix):
        target = column[len("ridge_"):-len(suffix)]
        return defense_mu_column(target)
    return None


def semantics_of(directory: Path) -> str:
    """Semantics recorded in a bundle's feature manifest; bundles predating the field are legacy."""
    path = Path(directory) / "feature_manifest.json"
    if not path.exists():
        return LEGACY_SEMANTICS
    return json.loads(path.read_text()).get("feature_semantics", LEGACY_SEMANTICS)


def is_isolated(directory: Path) -> bool:
    return semantics_of(directory) in ISOLATED_SEMANTICS


def covered_player_metrics() -> list[str]:
    from nflsim.ridge.metrics import TARGETS

    return sorted({target.player_output for target in TARGETS if target.scope == "player" and target.player_output})
