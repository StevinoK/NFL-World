import numpy as np
import torch

from nflsim.engine.model_architecture import build_network
from nflsim.models.game_effects import Data, ModelRows, fit


def overconfident_rows(games=120, plays=40, seed=0):
    """A binary model whose logits are twice the true ones, with no game-level variance."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0, 1.2, games * plays)
    label = (rng.uniform(size=true.size) < 1 / (1 + np.exp(-true))).astype(np.int64)
    game = np.repeat([f"g{index}" for index in range(games)], plays)
    team = np.tile(np.repeat(["A", "B"], plays // 2), games)
    return {"m": ModelRows("m", "binary", 2, (2 * true)[:, None], label, None, game, team)}


def test_the_fitted_temperature_undoes_overconfidence():
    data = Data(overconfident_rows())
    effects = fit(data, 0, 0, False, iterations=80, learning_rate=0.08, log_every=0, temperature=True)
    assert abs(float(effects.log_temperature["m"].exp()) - 2.0) < 0.3
    assert abs(effects.export()["models"]["m"]["temperature"] - float(effects.log_temperature["m"].exp())) < 1e-12
    plain = fit(data, 0, 0, False, iterations=5, log_every=0)
    assert "temperature" not in plain.export()["models"]["m"]


def test_the_runtime_divides_logits_by_the_temperature():
    from types import SimpleNamespace

    from nflsim.engine.simulation_runtime import SimulationRuntime
    network = build_network("flat", {"input_dim": 3, "output_dim": 2}, {"hidden_dims": [4], "dropout": [0.0]}).eval()
    x = torch.randn(5, 3)
    before = network(x).detach()
    runtime = SimulationRuntime.__new__(SimulationRuntime)
    runtime.model_dict = {"m": network}
    runtime.game_effects = SimpleNamespace(temperatures={"m": 2.0})
    runtime.apply_temperatures()
    assert torch.allclose(network(x), before / 2, atol=1e-6)
