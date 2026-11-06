"""Model-input preprocessing without per-play pandas frames or sklearn calls.

Every bundle preprocessor is a ColumnTransformer of a StandardScaler over numeric columns and a
OneHotEncoder(handle_unknown="ignore") over categorical ones, with the remaining columns dropped. Its
transform is compiled once into arrays and applied to plain dict rows with the same float64 arithmetic,
so outputs match ``processor.transform`` bit for bit. Anything else is reported as unsupported and the
caller keeps the sklearn path.
"""

from __future__ import annotations

import numpy as np


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN is the only value unequal to itself
    except TypeError:  # pd.NA refuses to be truth-tested
        return True


class CompiledProcessor:
    def __init__(self, processor, output_columns):
        self.columns = [str(column) for column in processor.feature_names_in_]
        transformers = {name: (transformer, list(columns)) for name, transformer, columns in processor.transformers_
                        if transformer != "drop"}
        scaler, numeric = transformers["num"]
        encoder, categorical = transformers["cat"]
        self.numeric = [str(column) for column in numeric]
        self.mean = np.asarray(scaler.mean_, dtype=np.float64) if scaler.with_mean else np.zeros(len(numeric))
        self.scale = np.asarray(scaler.scale_, dtype=np.float64) if scaler.with_std else np.ones(len(numeric))
        self.categorical = [str(column) for column in categorical]
        self.other = [column for column in self.columns if column not in set(self.numeric) | set(self.categorical)]
        self.offsets, self.lookup = [], []
        offset = len(self.numeric)
        for categories in encoder.categories_:
            self.offsets.append(offset)
            self.lookup.append({value: index for index, value in enumerate(categories.tolist())})
            offset += len(categories)
        self.width = offset
        self.output_columns = output_columns

    @staticmethod
    def supports(processor) -> bool:
        from sklearn.compose import ColumnTransformer

        transformers = getattr(processor, "transformers_", None)
        if type(processor) is not ColumnTransformer or transformers is None or processor.remainder != "drop":
            return False
        kinds = {name: type(transformer).__name__ for name, transformer, _ in transformers if transformer != "drop"}
        if kinds != {"num": "StandardScaler", "cat": "OneHotEncoder"}:
            return False
        encoder = processor.named_transformers_["cat"]
        return (encoder.handle_unknown == "ignore" and encoder.drop is None and not encoder.sparse_output
                and getattr(encoder, "max_categories", None) is None and getattr(encoder, "min_frequency", None) is None)

    def missing(self, row: dict) -> list[str]:
        return [column for column in self.columns if column not in row]

    def transform(self, row: dict) -> tuple[np.ndarray, dict[str, int]]:
        """Transformed row, plus the input columns that were NaN/None as pandas ``isna`` counts them after
        reindexing (absent columns are filled with 0.0, not NaN)."""
        values = np.array([row.get(column, 0.0) for column in self.numeric], dtype=np.float64)
        missing = {self.numeric[index]: 1 for index in np.flatnonzero(np.isnan(values))} if np.isnan(values).any() else {}
        missing.update({column: 1 for column in (*self.categorical, *self.other) if _missing(row.get(column, 0.0))})
        if missing:  # report in input-column order, as the frame-based count does
            missing = {column: 1 for column in self.columns if column in missing}
        output = np.zeros(self.width, dtype=np.float64)
        if self.numeric:
            values -= self.mean
            values /= self.scale
            output[:len(self.numeric)] = values
        for column, offset, lookup in zip(self.categorical, self.offsets, self.lookup):
            position = lookup.get(row.get(column, 0.0))
            if position is not None:
                output[offset + position] = 1.0
        return output, missing
