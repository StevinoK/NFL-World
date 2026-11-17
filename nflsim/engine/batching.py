"""Run many simulations of one game together, batching their model calls.

Each simulation runs unchanged as a greenlet. When it needs a model's logits it hands the scheduler its input
row and pauses; once every simulation is paused or finished, the scheduler evaluates each model once on all
rows waiting for it and resumes the simulations in a fixed order.

Each simulation keeps its own random streams: it gets private ``numpy.random.RandomState`` and
``random.Random`` instances, and the module-level functions the engine calls (``np.random.choice``,
``random.random``, ...) are rebound to the running simulation's instances at every switch. Seeding an instance
gives the same stream as seeding the global generator, so each simulation draws exactly the numbers it would
draw running alone. Batched and single-row matrix products may differ in the last bits; in practice the
outputs have matched the sequential path exactly.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from greenlet import getcurrent, greenlet

# Runtime attributes that belong to one simulation rather than the whole chunk.
SIMULATION_FIELDS = ("rng", "team_context", "neutral_site", "nan_input_log", "_nan_issue_labels", "_context_cache")
# Every module-level function bound to the global generators is rebound to the running simulation's own; any
# one left out (say random.randrange) would draw from a shared, unseeded stream.
NUMPY_FUNCTIONS = tuple(name for name in dir(np.random)
                        if getattr(getattr(np.random, name), "__self__", None) is np.random.mtrand._rand)
# Module functions that reach the global generator without being bound to it, and the method each stands for.
NUMPY_ALIASES = {"seed": "seed", "ranf": "random_sample", "sample": "random_sample"}
PYTHON_FUNCTIONS = tuple(name for name in dir(random)
                         if getattr(getattr(random, name), "__self__", None) is random._inst)


def _bind(numpy_state, python_state):
    for name in NUMPY_FUNCTIONS:
        setattr(np.random, name, getattr(numpy_state, name))
    for name, method in NUMPY_ALIASES.items():
        setattr(np.random, name, getattr(numpy_state, method))
    for name in PYTHON_FUNCTIONS:
        setattr(random, name, getattr(python_state, name))


class Batcher:
    def __init__(self, runtime):
        self.runtime = runtime
        self.hub = None

    @property
    def active(self) -> bool:
        return self.hub is not None and getcurrent() is not self.hub

    def request(self, model_key, tensor, shift):
        return self.hub.switch((model_key, tensor, shift))

    def _save(self, state):
        state["torch"] = torch.get_rng_state()
        state["fields"] = {field: getattr(self.runtime, field, None) for field in SIMULATION_FIELDS}

    def _restore(self, state):
        _bind(state["numpy"], state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        for field, value in state.get("fields", {}).items():
            setattr(self.runtime, field, value)

    def _evaluate(self, model_key, items):
        logits = self.runtime.network_logits(model_key, np.vstack([inputs for _, inputs, _ in items]))
        return [logits[index:index + 1] if shift is None else logits[index:index + 1] + shift
                for index, (_, _, shift) in enumerate(items)]

    def run(self, jobs):
        """Run zero-argument callables as concurrent simulations; returns their results in order."""
        self.hub = getcurrent()
        self.runtime.batcher = self
        lets = [greenlet(job) for job in jobs]
        states = [{"numpy": np.random.RandomState(), "python": random.Random()} for _ in jobs]
        results = [None] * len(jobs)
        waiting = {}
        original = ({name: getattr(np.random, name) for name in (*NUMPY_FUNCTIONS, *NUMPY_ALIASES)},
                    {name: getattr(random, name) for name in PYTHON_FUNCTIONS}, torch.get_rng_state())

        def resume(index, value=None):
            self._restore(states[index])
            out = lets[index].switch(value) if value is not None else lets[index].switch()
            self._save(states[index])
            if lets[index].dead:
                results[index] = out
            else:
                waiting[index] = out

        try:
            for index in range(len(jobs)):
                resume(index)
            while waiting:
                # Evaluate the model the most simulations are waiting on; the rest keep queueing behind it,
                # which grows batches far beyond evaluating every waiting model each round.
                groups = {}
                for index in sorted(waiting):
                    groups.setdefault(waiting[index][0], []).append(index)
                model_key = max(sorted(groups), key=lambda key: len(groups[key]))
                indices = groups[model_key]
                items = [waiting.pop(index) for index in indices]
                for index, logits in zip(indices, self._evaluate(model_key, items)):
                    resume(index, logits)
        finally:
            self.hub = None
            self.runtime.batcher = None
            for name, function in original[0].items():
                setattr(np.random, name, function)
            for name, function in original[1].items():
                setattr(random, name, function)
            torch.set_rng_state(original[2])
        return results
