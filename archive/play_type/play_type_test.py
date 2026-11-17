"""Evaluate a held-out play-type classification result.

The evaluator deliberately consumes predictions made on an already isolated
test set. It does not refit a model or randomly resample rows, which keeps
the reported numbers tied to the evaluation protocol rather than introducing
train/test leakage.

Expected CSV columns are ``actual`` and either ``predicted`` or one
probability column for each class. The default class order matches the
play-type model in ``Torch Modeling.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


DEFAULT_CLASSES = ("pass", "rush", "punt", "kick")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate held-out multiclass play-type predictions.")
	parser.add_argument("predictions", type=Path, help="CSV containing held-out rows")
	parser.add_argument("--actual-column", default="actual", help="Ground-truth label column")
	parser.add_argument("--predicted-column", default="predicted", help="Predicted label column")
	parser.add_argument(
		"--probability-columns",
		nargs=4,
		metavar=("PASS", "RUSH", "PUNT", "KICK"),
		help="Probability columns in pass, rush, punt, kick order",
	)
	parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="Class order")
	parser.add_argument("--output-prefix", type=Path, default=Path("play_type_evaluation"))
	return parser.parse_args()


def evaluate_predictions(
	frame: pd.DataFrame,
	actual_column: str = "actual",
	predicted_column: str = "predicted",
	probability_columns: list[str] | None = None,
	classes: tuple[str, ...] = DEFAULT_CLASSES,
) -> tuple[dict[str, object], np.ndarray, pd.DataFrame]:
	"""Calculate multiclass metrics from one held-out prediction table."""
	if actual_column not in frame.columns:
		raise ValueError(f"Missing ground-truth column: {actual_column}")
	if len(classes) < 2 or len(set(classes)) != len(classes):
		raise ValueError("--classes must contain at least two unique labels")

	actual = frame[actual_column].astype(str)
	if probability_columns:
		if len(probability_columns) != len(classes):
			raise ValueError("One probability column is required for every class")
		missing = [column for column in probability_columns if column not in frame.columns]
		if missing:
			raise ValueError(f"Missing probability columns: {', '.join(missing)}")
		probabilities = frame[probability_columns].apply(pd.to_numeric, errors="raise").to_numpy()
		if not np.isfinite(probabilities).all() or (probabilities < 0).any():
			raise ValueError("Probability columns must contain finite, non-negative values")
		row_totals = probabilities.sum(axis=1)
		if (row_totals <= 0).any():
			raise ValueError("Every probability row must have a positive sum")
		probabilities = probabilities / row_totals[:, None]
		predicted = pd.Series(np.asarray(classes)[np.argmax(probabilities, axis=1)], index=frame.index)
	else:
		if predicted_column not in frame.columns:
			raise ValueError(f"Missing prediction column: {predicted_column}; provide --probability-columns instead")
		predicted = frame[predicted_column].astype(str)

	allowed = set(classes)
	unknown = sorted((set(actual) | set(predicted)) - allowed)
	if unknown:
		raise ValueError(f"Unknown labels {unknown}; expected only {list(classes)}")
	if len(actual) == 0:
		raise ValueError("The evaluation table is empty")

	labels = list(classes)
	matrix = confusion_matrix(actual, predicted, labels=labels)
	report = classification_report(actual, predicted, labels=labels, target_names=labels, output_dict=True, zero_division=0)
	metrics = {
		"n_observations": int(len(actual)),
		"classes": labels,
		"accuracy": float(accuracy_score(actual, predicted)),
		"balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
		"macro_precision": float(precision_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
		"macro_recall": float(recall_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
		"macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
		"weighted_precision": float(precision_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
		"weighted_recall": float(recall_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
		"weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
		"per_class": {
			label: {"precision": float(report[label]["precision"]), "recall": float(report[label]["recall"]), "f1": float(report[label]["f1-score"]), "support": int(report[label]["support"])}
			for label in labels
		},
	}
	return metrics, matrix, pd.DataFrame(matrix, index=labels, columns=labels)


def _write_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
	figure, axis = plt.subplots(figsize=(6, 5))
	image = axis.imshow(matrix, cmap="Blues")
	axis.set_xticks(range(len(classes)), classes)
	axis.set_yticks(range(len(classes)), classes)
	axis.set_xlabel("Predicted play type")
	axis.set_ylabel("Actual play type")
	axis.set_title("Play-type confusion matrix")
	for row in range(len(classes)):
		for column in range(len(classes)):
			axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
	figure.colorbar(image, ax=axis, label="Observations")
	figure.tight_layout()
	figure.savefig(output_path, dpi=160)
	plt.close(figure)


def main() -> None:
	arguments = _parse_args()
	frame = pd.read_csv(arguments.predictions)
	metrics, matrix, matrix_frame = evaluate_predictions(
		frame,
		actual_column=arguments.actual_column,
		predicted_column=arguments.predicted_column,
		probability_columns=arguments.probability_columns,
		classes=tuple(arguments.classes),
	)
	output_prefix = arguments.output_prefix
	output_prefix.parent.mkdir(parents=True, exist_ok=True)
	output_prefix.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
	matrix_frame.to_csv(output_prefix.with_name(output_prefix.name + "_confusion_matrix.csv"))
	_write_confusion_matrix(matrix, list(arguments.classes), output_prefix.with_name(output_prefix.name + "_confusion_matrix.png"))
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
