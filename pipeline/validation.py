"""
Data validation stage — the quality gate in front of training.

Checks schema, statistics and semantics before training.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

from pipeline.config import (
    MAX_MISSING_FRACTION,
    MAX_POSITIVE_RATE,
    MIN_POSITIVE_RATE,
    MIN_ROWS,
    RAW_FEATURES,
    TARGET,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset fails a check that must not be ignored."""


# Value domains, from the dataset documentation. (PROVIDED)
DOMAINS: Dict[str, Any] = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
}
RANGES: Dict[str, tuple] = {
    "LIMIT_BAL": (10_000, 2_000_000),
    "AGE": (18, 100),
    **{c: (-2, 8) for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]},
}


# =============================================================================
# validate_schema — level 1
# =============================================================================
# "Is the data shaped the way the code expects?"
#
# Requirements:
#   - every column in RAW_FEATURES + [TARGET] must be present
#   - every one of those columns must be a numeric dtype
#   - return a LIST OF STRINGS describing what is wrong; empty list means clean
#
# Return errors rather than raising, so the caller can collect all three levels
# and report them together. A validator that stops at the first problem makes
# you fix issues one deploy at a time.
#
# Hint: pd.api.types.is_numeric_dtype(df[col])

def validate_schema(df: pd.DataFrame) -> List[str]:
    """Level 1 — are the expected columns present, with usable types?"""
    errors = []
    required = RAW_FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    for col in required:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"{col}: expected numeric dtype, got {df[col].dtype}")
    return errors


# =============================================================================
# validate_statistics — level 2
# =============================================================================
# "Is the shape of the distribution what training assumes?"
#
# Requirements:
#   - at least MIN_ROWS rows
#   - no column more than MAX_MISSING_FRACTION missing
#   - target positive rate inside [MIN_POSITIVE_RATE, MAX_POSITIVE_RATE]
#
# The target check is the one that earns its keep. If an upstream extract breaks
# and every label comes back 0, training still succeeds — you get a model with
# 100% accuracy that predicts "no default" for everyone. This check catches that
# before anyone celebrates.

def validate_statistics(df: pd.DataFrame) -> List[str]:
    """Level 2 — is the shape of the data what training assumes?"""
    errors = []
    if len(df) < MIN_ROWS:
        errors.append(f"too few rows: {len(df)}; minimum is {MIN_ROWS}")
    for col, fraction in df.isna().mean().items():
        if fraction > MAX_MISSING_FRACTION:
            errors.append(
                f"{col}: missing fraction {fraction:.4f} exceeds {MAX_MISSING_FRACTION}"
            )
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]):
        positive_rate = df[TARGET].mean()
        if pd.isna(positive_rate) or not MIN_POSITIVE_RATE <= positive_rate <= MAX_POSITIVE_RATE:
            errors.append(
                f"{TARGET}: positive rate {positive_rate} outside "
                f"[{MIN_POSITIVE_RATE}, {MAX_POSITIVE_RATE}]"
            )
    return errors


# =============================================================================
# validate_semantics — level 3
# =============================================================================
# "Do the values mean what the business says they mean?"
#
# Requirements:
#   - every column in DOMAINS may only contain the allowed values
#   - every column in RANGES must stay inside its [lo, hi]
#   - no PAY_AMT* column may contain a negative number
#
# Level 3 is where domain knowledge lives. A SEX of 7 or an AGE of 400 is
# perfectly valid as an integer and perfectly meaningless as a customer.

def validate_semantics(df: pd.DataFrame) -> List[str]:
    """Level 3 — do the values mean what the business says they mean?"""
    errors = []
    for col, allowed in DOMAINS.items():
        if col in df.columns:
            values = df[col].dropna()
            invalid = values[~values.isin(allowed)].unique().tolist()
            if invalid:
                errors.append(f"{col}: values {invalid} outside allowed domain {sorted(allowed)}")
    for col, (lo, hi) in RANGES.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna()
            if ((values < lo) | (values > hi)).any():
                errors.append(f"{col}: values outside [{lo}, {hi}]")
    for col in df.columns:
        if col.startswith("PAY_AMT") and pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] < 0).any():
                errors.append(f"{col}: contains negative payments")
    return errors


# =============================================================================
# validate_dataset
# =============================================================================
# Requirements:
#   - run all three levels and concatenate their errors
#   - build and return this report:
#         {"passed": bool, "n_rows": int, "n_columns": int,
#          "schema_errors": [...], "statistical_errors": [...],
#          "semantic_errors": [...], "n_errors": int}
#   - log every error at ERROR level
#   - if there are errors and raise_on_error is True, raise DataValidationError
#
# The report is logged to MLflow as an artifact by the training stage, so a
# model trained on data with known problems carries the evidence with it.

def validate_dataset(df: pd.DataFrame, raise_on_error: bool = True) -> Dict[str, Any]:
    """Run all three levels and return a report."""
    schema_errors = validate_schema(df)
    statistical_errors = validate_statistics(df)
    semantic_errors = validate_semantics(df)
    errors = schema_errors + statistical_errors + semantic_errors
    report = {
        "passed": not errors,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "schema_errors": schema_errors,
        "statistical_errors": statistical_errors,
        "semantic_errors": semantic_errors,
        "n_errors": len(errors),
    }
    for error in errors:
        logger.error(error)
    if errors and raise_on_error:
        raise DataValidationError("Data validation failed: " + "; ".join(errors))
    return report
