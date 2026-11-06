from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nflsim.engine.fast_inputs import CompiledProcessor
from nflsim.engine.model_architecture import HierarchicalLatentGameModel
from nflsim.engine.ngmm_model import NGMMModel
from nflsim.engine.simulation_runtime import CascadeModelWrapper, Game, PlayRow, SimulationRuntime
from nflsim.features.player_value import PV_DEFENSE_COLUMNS
from nflsim.semantics import (
    ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3, ISOLATED_SEMANTICS, LEGACY_SEMANTICS, context_available_column,
    covered_player_metrics, opponent_defense_source,
    semantics_of,
)


def opponent_context_map(context_map: dict, semantics: str) -> dict[str, str]:
    """V2 opponent-defense context column -> opposing team's own same-fit defense-mu column."""
    if semantics not in {ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3}:
        return {}
    columns = {column for values in context_map.values() for column in values}
    return {column: opponent_defense_source(column) for column in sorted(columns) if opponent_defense_source(column)}


def with_opponent_context(mapping: dict[str, str], attributes, opponent_attributes):
    """Copy of ``attributes`` whose opponent-defense contexts come from the actual opponent; own mus untouched."""
    result = attributes.copy()
    for column, source in mapping.items():
        if source not in opponent_attributes:
            raise ValueError(f"Opponent lacks {source} for {column}")
        value = float(opponent_attributes[source])
        if not np.isfinite(value):
            flag = opponent_attributes.get(context_available_column(source), np.nan)
            if not source.endswith("_available") and np.isfinite(float(flag)) and float(flag) == 0.0:
                value = 0.0
            else:
                raise ValueError(f"Nonfinite opponent {source} for {column}")
        result[column] = value
    return result


class RidgeRuntime(SimulationRuntime):
    # Compiled preprocessing for one-row plays of plain networks; outputs match the sklearn path bit for bit.
    fast_inputs = True
    # Set by nflsim.engine.batching to evaluate many simulations' model calls in one forward pass.
    batcher = None

    def __init__(self, data_dir: Path, strict_schema: bool = True):
        self.ridge_manifest = json.loads((Path(data_dir) / "training_manifest.json").read_text())
        self.feature_semantics = self.ridge_manifest.get("feature_semantics", LEGACY_SEMANTICS)
        if semantics_of(data_dir) != self.feature_semantics:
            raise ValueError(
                f"Feature schema {semantics_of(data_dir)} does not match models trained on {self.feature_semantics}")
        self.isolated_mu = self.feature_semantics in ISOLATED_SEMANTICS
        self.mu_player_metrics = set(covered_player_metrics()) if self.isolated_mu else set()
        self.context_map = self.ridge_manifest["context_map"]
        self.opponent_context = opponent_context_map(self.context_map, self.feature_semantics)
        self.rng = np.random.default_rng(0)
        super().__init__(data_dir, strict_schema=strict_schema)
        context = sorted({column for columns in self.context_map.values() for column in columns})
        self.required_team_feature_columns = sorted(set(self.required_team_feature_columns + context + list(self.opponent_context.values())))
        self._validate_context_contracts()

    def _validate_context_contracts(self) -> None:
        for model, columns in self.context_map.items():
            if model not in self.model_dict:
                continue
            network = self.model_dict[model]
            if isinstance(network, CascadeModelWrapper):
                if not set(columns) <= set(network.context_columns):
                    raise ValueError(f"{model}: context absent from cascade stage 2")
            else:
                expected = set(getattr(self.processor[model], "feature_names_in_", []))
                if not set(columns) <= expected:
                    raise ValueError(f"{model}: context absent from trained preprocessor")
            prior = self.processor.get(f"{model}_prior")
            if prior is not None and not set(columns) <= set(prior.feature_names_in_):
                raise ValueError(f"{model}: context absent from prior preprocessor")

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def structural_schema_issues(self):
        return [message for message in self.schema_issues if ": detected NaN inputs (" not in message]

    def checked_context(self, model: str, source) -> dict:
        source = dict(source)
        values = {}
        for column in self.context_map.get(model, []):
            if column not in source:
                raise ValueError(f"{model}: missing ridge context {column}")
            value = float(source[column])
            availability = context_available_column(column)
            if (getattr(self, "feature_semantics", LEGACY_SEMANTICS) in {ISOLATED_MU_CONTEXT_V2, ISOLATED_MU_CONTEXT_V3}
                    and column.startswith("ridge_") and not column.endswith("_available")
                    and (availability in source or getattr(self, "feature_semantics", LEGACY_SEMANTICS) == ISOLATED_MU_CONTEXT_V3)):
                if availability in source:
                    available = pd.to_numeric(pd.Series([source[availability]]), errors="coerce").iloc[0]
                else:
                    available = float(np.isfinite(value))
                if not np.isfinite(value) or not np.isfinite(available) or available <= 0:
                    value = 0.0
                    if availability in source:
                        source[availability] = 0.0
                else:
                    if availability in source:
                        source[availability] = 1.0
            if not np.isfinite(value):
                raise ValueError(f"{model}: nonfinite ridge context {column}")
            values[column] = value
        for column in self.context_map.get(model, []):
            if column.endswith("_available") and column in source:
                value = float(source[column])
                if not np.isfinite(value):
                    raise ValueError(f"{model}: nonfinite ridge context {column}")
                values[column] = value
        return values

    def compiled_processor(self, key):
        cache = self.__dict__.setdefault("_compiled_processors", {})
        processor = self.processor.get(key)
        if key not in cache or cache[key][0] is not processor:  # recompile if the processor is replaced
            supported = processor is not None and CompiledProcessor.supports(processor)
            cache[key] = (processor, CompiledProcessor(processor, self._post_transform_columns(processor)) if supported else None)
        return cache[key][1]

    def _fast_logits(self, model_key, row, latent_draw):
        """forward_model_logits for one play row of a plain network, without pandas or sklearn."""
        compiled = self.compiled_processor(model_key)
        if model_key in {"target", "rusher"}:
            missing = compiled.missing(row)
            if missing:
                message = f"{model_key}: missing required selector columns (fail-fast): {missing}"
                self._record_schema_issue(message)
                raise ValueError(message)
        missing = compiled.missing(row)
        if missing:
            self._record_schema_issue(f"{model_key}: auto-filled missing columns: {missing}")

        def transform(values):
            output, counts = compiled.transform(values)
            if counts:
                self._accumulate_nan_log(model_key, "processor_input", 1, len(counts), 1, counts)
            return output

        if getattr(self, "neutral_site", False) and "posteam_type" in row:
            inputs = (transform({**row, "posteam_type": "home"}) + transform({**row, "posteam_type": "away"})) * 0.5
        else:
            inputs = transform(row)
        inputs = inputs[None, :]
        if np.isnan(inputs).any():
            self._record_nan_array(model_key, "post_transform", inputs, compiled.output_columns)
        shift = None
        if latent_draw is not None and self.game_effects is not None and model_key in self.game_effects.models:
            shift = torch.as_tensor(np.asarray(latent_draw, dtype=np.float32))
        if self.batcher is not None and self.batcher.active:
            return self.batcher.request(model_key, inputs, shift)
        logits = self.network_logits(model_key, inputs)
        return logits if shift is None else logits + shift

    def network_logits(self, model_key, inputs):
        """Logits for transformed input rows: NaNs zeroed and values clipped as the sklearn path does."""
        inputs = np.clip(np.nan_to_num(inputs, nan=0.0), -self.MODEL_INPUT_CLIP, self.MODEL_INPUT_CLIP)
        with torch.no_grad():
            return self.model_dict[model_key](self.as_model_input(model_key, inputs))

    def _fast_path(self, model_key, single):
        model = getattr(self, "model_dict", {}).get(model_key)
        return (model is not None and self.fast_inputs and not getattr(self, "debug_print_outliers", False) and single
                and not isinstance(model, (CascadeModelWrapper, NGMMModel, HierarchicalLatentGameModel))
                and self.compiled_processor(model_key) is not None)

    def _play_context(self, model_key, role, base_values):
        """Checked Ridge context for a play by the ``role`` offense. It depends only on the two teams unless the
        play's own values carry a context column, so it is cached per game (keyed on the team_context object)."""
        if not self.context_map.get(model_key):
            return {}
        roles = getattr(self, "team_context", {})
        needed = self.__dict__.setdefault("_context_inputs", {})
        if model_key not in needed:
            columns = self.context_map[model_key]
            needed[model_key] = set(columns) | {context_available_column(column) for column in columns}
        cacheable = needed[model_key].isdisjoint(base_values)
        cache = getattr(self, "_context_cache", None)
        if cacheable and cache is not None and cache[0] is roles and (model_key, role) in cache[1]:
            return cache[1][(model_key, role)]
        source = dict(base_values)
        source.update(roles.get(role, {}))
        opponent = next((context for key, context in roles.items() if key != role), None)
        if opponent is not None:
            source.update({column: opponent[column] for column in PV_DEFENSE_COLUMNS if column in opponent})
            mapping = getattr(self, "opponent_context", {})
            if mapping:
                source = with_opponent_context(mapping, source, opponent)
        values = self.checked_context(model_key, source)
        if cacheable:
            if cache is None or cache[0] is not roles:
                cache = self._context_cache = (roles, {})
            cache[1][(model_key, role)] = values
        return values

    def forward_model_logits(self, model_key, play_frame, base_values=None, latent_draw=None):
        if isinstance(play_frame, PlayRow):
            first, fast = dict(play_frame), self._fast_path(model_key, True)
            if not fast:
                play_frame = play_frame.frame()
        else:
            first = dict(zip(play_frame.columns, play_frame.to_numpy(dtype=object)[0])) if len(play_frame) else {}
            fast = self._fast_path(model_key, len(play_frame) == 1)
        frame = None if fast else play_frame.copy()
        role = (base_values or {}).get("posteam_type", first.get("posteam_type") if first else None)
        values = self._play_context(model_key, role, base_values or {})
        if fast:
            return self._fast_logits(model_key, {**first, **values}, latent_draw)
        for column, value in values.items():
            frame[column] = value
        return super().forward_model_logits(model_key, frame, {**(base_values or {}), **values}, latent_draw)

    def transform_with_processor(self, processor_key, frame):
        model = processor_key.removesuffix("_prior") if hasattr(str, "removesuffix") else processor_key[:-6] if processor_key.endswith("_prior") else processor_key
        for column in self.context_map.get(model, []):
            if column not in frame or not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
                raise ValueError(f"{processor_key}: missing/nonfinite ridge context {column}")
        if getattr(self, "neutral_site", False) and "posteam_type" in frame:
            home_frame = frame.assign(posteam_type="home")
            away_frame = frame.assign(posteam_type="away")
            home_values = super().transform_with_processor(processor_key, home_frame)
            away_values = super().transform_with_processor(processor_key, away_frame)
            return (home_values + away_values) * 0.5
        return super().transform_with_processor(processor_key, frame)


class RidgeGame(Game):
    def __init__(self, team1, team2, dome, runtime):
        mapping = getattr(runtime, "opponent_context", {})
        if mapping:
            first, second = team1.attributes, team2.attributes
            team1.attributes = with_opponent_context(mapping, first, second)
            team2.attributes = with_opponent_context(mapping, second, first)
        runtime.team_context = {team.home: team.attributes.to_dict() for team in (team1, team2)}
        super().__init__(team1, team2, dome, runtime)

    def _resample_prior_embeddings(self):
        runtime = self.runtime
        for team in (self.team1, self.team2):
            for model in runtime.context_map:
                runtime.checked_context(model, team.attributes)
        cascade_models = {key: model for key, model in runtime.model_dict.items() if isinstance(model, CascadeModelWrapper)}
        original = runtime.model_dict
        runtime.model_dict = {key: model for key, model in original.items() if key not in cascade_models}
        try:
            super()._resample_prior_embeddings()
        finally:
            runtime.model_dict = original
        for key, model in cascade_models.items():
            for team in (self.team1, self.team2):
                opponent = self.team2 if team is self.team1 else self.team1
                values = {column: float((opponent.attributes if column in runtime.def_cols or column in PV_DEFENSE_COLUMNS else team.attributes).get(column, 0.0)) for column in model.context_columns}
                own = runtime.checked_context(key, team.attributes)
                own.update({column: float(opponent.attributes[column]) for column in PV_DEFENSE_COLUMNS if column in own})
                values.update(own)
                residual = model.sample_residual(values, runtime.rng)
                latent = np.array([residual], dtype=np.float32)
                team.model_latents[key] = latent
                if key == "target":
                    team.target_latent = latent
                    team.wr_embed = np.full(len(runtime.target_embed_columns), residual, dtype=np.float32)
                elif key == "rusher":
                    team.rusher_latent = latent
                    team.rb_embed = np.full(len(runtime.running_embed_columns), residual, dtype=np.float32)