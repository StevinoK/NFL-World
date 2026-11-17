from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ReportEntry:
    model_name: str
    model_type: str
    metrics: Dict[str, float] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)


class ModelReportBuilder:
    def __init__(self, title: str = "ML Model Performance Report"):
        self.title = title
        self.entries: List[ReportEntry] = []

    @staticmethod
    def _fig_to_base64(fig) -> str:
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        if not np.any(mask):
            return float("nan")
        return float(np.sqrt(mean_squared_error(y_true_arr[mask], y_pred_arr[mask])))

    @staticmethod
    def _extract_loss_curves(loss_history: Any) -> tuple[np.ndarray, np.ndarray]:
        train_losses: List[float] = []
        valid_losses: List[float] = []

        if loss_history is None:
            return np.asarray(train_losses, dtype=float), np.asarray(valid_losses, dtype=float)

        if isinstance(loss_history, dict):
            train_values = loss_history.get("train_loss", [])
            valid_values = loss_history.get("valid_loss", [])
            train_losses = [float(v) for v in train_values if np.isfinite(v)]
            valid_losses = [float(v) for v in valid_values if np.isfinite(v)]
            return np.asarray(train_losses, dtype=float), np.asarray(valid_losses, dtype=float)

        try:
            iterable_history = list(loss_history)
        except TypeError:
            iterable_history = []

        for epoch_row in iterable_history:
            if not isinstance(epoch_row, dict):
                continue

            train_value = epoch_row.get("train_loss")
            valid_value = epoch_row.get("valid_loss")

            if train_value is not None and np.isfinite(train_value):
                train_losses.append(float(train_value))
            if valid_value is not None and np.isfinite(valid_value):
                valid_losses.append(float(valid_value))

        return np.asarray(train_losses, dtype=float), np.asarray(valid_losses, dtype=float)

    def _append_loss_plot(self, images: List[str], model_name: str, loss_history: Any) -> None:
        train_losses, valid_losses = self._extract_loss_curves(loss_history)
        if train_losses.size == 0 and valid_losses.size == 0:
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        if train_losses.size:
            epochs_train = np.arange(1, train_losses.size + 1)
            ax.plot(epochs_train, train_losses, label="Train Loss", linewidth=2)
        if valid_losses.size:
            epochs_valid = np.arange(1, valid_losses.size + 1)
            ax.plot(epochs_valid, valid_losses, label="Validation Loss", linewidth=2)

        ax.set_title(f"{model_name} Loss vs Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        images.append(self._fig_to_base64(fig))

    def add_binary_classification(
        self,
        model_name: str,
        y_true: Iterable,
        y_score: Iterable,
        threshold: float = 0.5,
        loss_history: Any = None,
    ) -> None:
        y_true_arr = np.asarray(y_true).astype(int)
        y_score_arr = np.asarray(y_score).astype(float)
        y_pred_arr = (y_score_arr >= threshold).astype(int)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
            "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
            "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
            "rmse_prob": self._safe_rmse(y_true_arr, y_score_arr),
        }

        unique_vals = np.unique(y_true_arr)
        images: List[str] = []

        if unique_vals.size > 1:
            metrics["auc"] = float(roc_auc_score(y_true_arr, y_score_arr))
            fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title(f"{model_name} ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            images.append(self._fig_to_base64(fig))

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(f"{model_name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.colorbar(im, ax=ax)
        images.append(self._fig_to_base64(fig))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(y_score_arr, bins=30, alpha=0.8)
        ax.set_title(f"{model_name} Score Distribution")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        images.append(self._fig_to_base64(fig))
        self._append_loss_plot(images, model_name, loss_history)

        self.entries.append(ReportEntry(model_name=model_name, model_type="binary", metrics=metrics, images=images))

    def add_multiclass(
        self,
        model_name: str,
        y_true: Iterable,
        probs: np.ndarray,
        bins: Optional[Iterable] = None,
        cdfs: Optional[Iterable] = None,
        loss_history: Any = None,
    ) -> None:
        y_true_arr = np.asarray(y_true).astype(int)
        probs_arr = np.asarray(probs).astype(float)
        original_probs = probs_arr.copy()

        # Defensive cleanup: replace non-finite probabilities and re-normalize rows.
        probs_arr = np.nan_to_num(probs_arr, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs_arr.sum(axis=1, keepdims=True)
        valid_rows = row_sums[:, 0] > 0
        repaired_rows = (~np.all(np.isfinite(original_probs), axis=1)) | (~valid_rows)
        if probs_arr.shape[1] > 0:
            probs_arr[~valid_rows] = 1.0 / probs_arr.shape[1]
        row_sums = probs_arr.sum(axis=1, keepdims=True)
        probs_arr = np.divide(probs_arr, row_sums, out=np.zeros_like(probs_arr), where=row_sums > 0)

        pred_class = np.argmax(probs_arr, axis=1)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true_arr, pred_class)),
            "rows_total": float(probs_arr.shape[0]),
            "rows_repaired": float(np.count_nonzero(repaired_rows)),
        }

        if bins is not None:
            bins_arr = np.asarray(list(bins)).astype(float)
            if bins_arr.shape[0] == probs_arr.shape[1]:
                pred_mean = (probs_arr * bins_arr).sum(axis=1)
                true_mean = bins_arr[y_true_arr]
                finite_mask = np.isfinite(true_mean) & np.isfinite(pred_mean)
                metrics["rows_dropped_nonfinite_ev"] = float(np.count_nonzero(~finite_mask))
                if np.any(finite_mask):
                    metrics["rmse"] = self._safe_rmse(true_mean[finite_mask], pred_mean[finite_mask])
                    metrics["mae"] = float(mean_absolute_error(true_mean[finite_mask], pred_mean[finite_mask]))

                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    ax.scatter(pred_mean[finite_mask], true_mean[finite_mask], s=8, alpha=0.4)
                    ax.set_title(f"{model_name} Predicted vs Actual")
                    ax.set_xlabel("Predicted Expected Value")
                    ax.set_ylabel("Actual Value")
                    scatter_image = self._fig_to_base64(fig)
                else:
                    scatter_image = ""
            else:
                scatter_image = ""
        else:
            scatter_image = ""

        if cdfs is None:
            cdfs_arr = []
            for idx, cls in enumerate(y_true_arr):
                cdfs_arr.append(float(np.sum(probs_arr[idx, :cls])))
            cdfs_arr = np.asarray(cdfs_arr)
        else:
            cdfs_arr = np.asarray(cdfs).astype(float)

        metrics["cdfs_nonfinite"] = float(np.count_nonzero(~np.isfinite(cdfs_arr)))
        cdfs_arr = np.nan_to_num(cdfs_arr, nan=0.0, posinf=1.0, neginf=0.0)

        images: List[str] = []
        if scatter_image:
            images.append(scatter_image)

        cm = confusion_matrix(y_true_arr, pred_class)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        if cm.shape[0] <= 20:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(f"{model_name} Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        fig.colorbar(im, ax=ax)
        images.append(self._fig_to_base64(fig))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(cdfs_arr, bins=30, density=True, alpha=0.85)
        ax.set_title(f"{model_name} CDF Calibration Histogram")
        ax.set_xlabel("CDF at Actual Outcome")
        ax.set_ylabel("Density")
        images.append(self._fig_to_base64(fig))

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(np.max(probs_arr, axis=1), bins=30, alpha=0.85)
        ax.set_title(f"{model_name} Confidence Distribution")
        ax.set_xlabel("Max Predicted Probability")
        ax.set_ylabel("Count")
        images.append(self._fig_to_base64(fig))
        self._append_loss_plot(images, model_name, loss_history)

        self.entries.append(ReportEntry(model_name=model_name, model_type="multiclass", metrics=metrics, images=images))

    def add_regression(self, model_name: str, y_true: Iterable, y_pred: Iterable) -> None:
        y_true_arr = np.asarray(y_true).astype(float)
        y_pred_arr = np.asarray(y_pred).astype(float)

        metrics = {
            "rmse": self._safe_rmse(y_true_arr, y_pred_arr),
            "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
            "r2": float(r2_score(y_true_arr, y_pred_arr)),
        }

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(y_pred_arr, y_true_arr, s=8, alpha=0.4)
        ax.set_title(f"{model_name} Predicted vs Actual")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        scatter_img = self._fig_to_base64(fig)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.hist(y_true_arr - y_pred_arr, bins=30, alpha=0.85)
        ax.set_title(f"{model_name} Residual Distribution")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        resid_img = self._fig_to_base64(fig)

        self.entries.append(
            ReportEntry(model_name=model_name, model_type="regression", metrics=metrics, images=[scatter_img, resid_img])
        )

    def write_html(self, output_path: Path | str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        sections = []
        for entry in self.entries:
            metrics_html = "".join(
                f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in entry.metrics.items()
            )
            images_html = "".join(
                f"<img src=\"{img}\" style=\"max-width: 520px; margin: 8px 10px 8px 0; border: 1px solid #ddd;\"/>"
                for img in entry.images
            )
            section = f"""
            <section style=\"margin-bottom: 28px;\">
              <h2 style=\"margin-bottom: 6px;\">{entry.model_name}</h2>
              <p style=\"color: #666; margin-top: 0;\">Type: {entry.model_type}</p>
              <table style=\"border-collapse: collapse; margin-bottom: 10px;\">{metrics_html}</table>
              <div>{images_html}</div>
            </section>
            """
            sections.append(section)

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset=\"utf-8\"/>
          <title>{self.title}</title>
          <style>
            body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #222; }}
            h1 {{ margin-bottom: 4px; }}
            td {{ border: 1px solid #ddd; padding: 6px 10px; }}
            tr td:first-child {{ font-weight: 600; background: #fafafa; }}
          </style>
        </head>
        <body>
          <h1>{self.title}</h1>
          <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
          {''.join(sections)}
        </body>
        </html>
        """

        output.write_text(html, encoding="utf-8")
