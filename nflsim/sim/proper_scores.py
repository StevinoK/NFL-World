"""Proper scores for a sample of simulated outcomes against what happened. All are lower-is-better.

The sample-based estimators are the fair ones (pairs of distinct draws), so a score doesn't depend on how many
simulations a run used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def crps(samples: np.ndarray, value: float) -> float:
    """Fair CRPS: E|X - y| - E|X - X'| / 2 over distinct sample pairs."""
    samples = np.sort(np.asarray(samples, float))
    n = len(samples)
    spread = np.sum((2 * np.arange(1, n + 1) - n - 1) * samples) / (n * (n - 1))
    return float(np.mean(np.abs(samples - value)) - spread)


def energy_score(draws: np.ndarray, actual: np.ndarray) -> float:
    """Fair energy score: E|X - y| - E|X - X'| / 2 over distinct sample pairs (rows of ``draws``)."""
    n = len(draws)
    first = np.linalg.norm(draws - actual, axis=1).mean()
    return float(first - pdist(draws).sum() / (n * (n - 1)))


def variogram_score(draws: np.ndarray, actual: np.ndarray, order: float = 0.5) -> float:
    left, right = np.triu_indices(draws.shape[1], 1)
    expected = (np.abs(draws[:, left] - draws[:, right]) ** order).mean(axis=0)
    return float(((np.abs(actual[left] - actual[right]) ** order - expected) ** 2).sum())
