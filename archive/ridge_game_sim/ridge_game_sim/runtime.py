from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from torch_game_simulation_runtime import CascadeModelWrapper, Game, SimulationRuntime


class RidgeRuntime(SimulationRuntime):
    def __init__(self, data_dir: Path, strict_schema: bool = True):
        self.ridge_manifest = json.loads((Path(data_dir) / "training_manifest.json").read_text())
        self.context_map = self.ridge_manifest["context_map"]
        self.rng = np.random.default_rng(0)
        super().__init__(data_dir, strict_schema=strict_schema)
        context = sorted({column for columns in self.context_map.values() for column in columns})
        self.required_team_feature_columns = sorted(set(self.required_team_feature_columns + context))
        for model, columns in self.context_map.items():
            if model not in self.model_dict:
                continue
            expected = set(getattr(self.processor[model], "feature_names_in_", []))
            if not set(columns) <= expected:
                raise ValueError(f"{model}: context absent from trained preprocessor")
            network = self.model_dict[model]
            if isinstance(network, CascadeModelWrapper):
                if not set(columns) <= set(network.context_columns):
                    raise ValueError(f"{model}: context absent from cascade stage 2")
            prior = self.processor.get(f"{model}_prior")
            if prior is not None and not set(columns) <= set(prior.feature_names_in_):
                raise ValueError(f"{model}: context absent from prior preprocessor")

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def structural_schema_issues(self):
        return [message for message in self.schema_issues if ": detected NaN inputs (" not in message]

    def checked_context(self, model: str, source) -> dict:
        values = {}
        for column in self.context_map.get(model, []):
            if column not in source:
                raise ValueError(f"{model}: missing ridge context {column}")
            value = float(source[column])
            if not np.isfinite(value):
                raise ValueError(f"{model}: nonfinite ridge context {column}")
            values[column] = value
        return values

    def forward_model_logits(self, model_key, play_frame, base_values=None, latent_draw=None):
        frame = play_frame.copy()
        source = dict(base_values or {})
        role = source.get("posteam_type", frame.iloc[0].get("posteam_type") if not frame.empty else None)
        source.update(getattr(self, "team_context", {}).get(role, {}))
        values = self.checked_context(model_key, source) if self.context_map.get(model_key) else {}
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
                values = {column: float((opponent.attributes if column in runtime.def_cols else team.attributes).get(column, 0.0)) for column in model.context_columns}
                values.update(runtime.checked_context(key, team.attributes))
                residual = model.sample_residual(values, runtime.rng)
                latent = np.array([residual], dtype=np.float32)
                team.model_latents[key] = latent
                if key == "target":
                    team.target_latent = latent
                    team.wr_embed = np.full(len(runtime.target_embed_columns), residual, dtype=np.float32)
                elif key == "rusher":
                    team.rusher_latent = latent
                    team.rb_embed = np.full(len(runtime.running_embed_columns), residual, dtype=np.float32)