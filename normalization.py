"""
===============================================================================
Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
YuvaIntern Internship Assignment

Script:
    normalization.py

Purpose:
    Implement Min-Max feature normalization on numerical logistics attributes.

    Rescales:
        1. Distance_KM
        2. Delivery_Days
        3. Shipping_Cost

    Normalized values should fall within the [0, 1] range.

    The script preserves:
        - Original numerical features and their units
        - Original non-numerical features
        - Record count and Order_ID sequence
        - Missing-value positions
        - Statistical outliers

    The script generates:
        - Normalized logistics dataset
        - Normalization summary report
        - Before-and-after comparison report
        - Original-versus-normalized boxplots
        - Normalized feature distribution charts

Save To:
    C:\\Logistics_Preprocessing\\normalization.py

Input:
    data/logistics_data_cleaned.csv

Outputs:
    data/logistics_data_normalized.csv
    outputs/normalization_summary.csv
    outputs/normalization_comparison.csv
    outputs/original_vs_normalized_boxplots.png
    outputs/normalized_feature_distributions.png

Python:
    3.14.3
===============================================================================
"""

import sys
from pathlib import Path

# Configure Matplotlib's non-interactive backend before importing pyplot.
# This allows charts to be generated from the VS Code terminal without
# requiring a graphical display window.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# SCHEMA AND FEATURE DEFINITIONS
# =============================================================================

EXPECTED_COLUMNS = [
    "Order_ID",
    "Transport_Mode",
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
    "Customer_City",
]

NUMERICAL_COLUMNS = [
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
]

NON_NUMERICAL_COLUMNS = [
    "Order_ID",
    "Transport_Mode",
    "Customer_City",
]

NORMALIZED_COLUMN_MAP = {
    "Distance_KM": "Distance_KM_Normalized",
    "Delivery_Days": "Delivery_Days_Normalized",
    "Shipping_Cost": "Shipping_Cost_Normalized",
}

FEATURE_UNITS = {
    "Distance_KM": "KM",
    "Delivery_Days": "Days",
    "Shipping_Cost": "INR",
}

NORMALIZATION_METHOD = "Min-Max Scaling [0, 1]"

FLOAT_TOLERANCE = 1e-8

# Invalid numerical entries are not silently imputed by this script.
# The script converts invalid values and infinities to NaN, records them,
# and marks the affected feature as PARTIAL_WITH_INVALID_VALUES.
#
# Existing genuine missing values remain missing as well.
# Normalization uses the valid finite observations for the feature's
# minimum and maximum.
#
# If a feature has no valid finite observations, it cannot be normalized.
#
# If every valid observation has the same value, the normalized valid
# observations are assigned 0.0 to avoid division by zero.


# =============================================================================
# 1. PROJECT PATH CONFIGURATION
# =============================================================================

def get_project_paths() -> tuple:
    """
    Resolve project directories and file paths dynamically.

    The project root is determined using the location of this Python file.

    Returns:
        A tuple containing:
            project_root
            input_csv
            output_norm_csv
            summary_csv
            comparison_csv
            boxplot_png
            distribution_png
    """

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    # Ensure the required output directories exist.
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    input_csv = data_dir / "logistics_data_cleaned.csv"

    output_norm_csv = data_dir / "logistics_data_normalized.csv"

    summary_csv = outputs_dir / "normalization_summary.csv"

    comparison_csv = outputs_dir / "normalization_comparison.csv"

    boxplot_png = outputs_dir / "original_vs_normalized_boxplots.png"

    distribution_png = (
        outputs_dir / "normalized_feature_distributions.png"
    )

    return (
        project_root,
        input_csv,
        output_norm_csv,
        summary_csv,
        comparison_csv,
        boxplot_png,
        distribution_png,
    )


# =============================================================================
# 2. INPUT DATA LOADING AND VALIDATION
# =============================================================================

def load_and_validate_cleaned_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the cleaned dataset and validate its basic structure.

    Checks:
        1. Input file exists.
        2. Input path points to a regular file.
        3. CSV can be read.
        4. Dataset contains data rows.
        5. Required columns are present.

    The input file is never overwritten by this function.
    """

    print("=" * 78)
    print("STEP 1: PROJECT PATHS AND INPUT VALIDATION")
    print("=" * 78)

    print(f"Target input dataset: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned input dataset does not exist: {csv_path}\n"
            "Please run data_cleaning.py successfully first."
        )

    if not csv_path.is_file():
        raise ValueError(
            f"Expected a regular CSV file but found another path: {csv_path}"
        )

    try:
        df = pd.read_csv(csv_path)

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The input CSV is empty: {csv_path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"The input CSV has a parsing problem: {exc}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode the input CSV: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The cleaned dataset contains no data records."
        )

    # Remove accidental leading/trailing spaces from column names only.
    # This does not modify the source CSV on disk.
    df.columns = [str(col).strip() for col in df.columns]

    if len(df.columns) != len(set(df.columns)):
        raise ValueError(
            "Duplicate column names were detected after trimming whitespace."
        )

    actual_columns = list(df.columns)

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Actual columns: {actual_columns}"
        )

    print("\n[OK] Source input validated successfully.")

    print(f"Total records : {len(df)}")
    print(f"Total columns : {len(df.columns)}")

    print("-" * 78)
    print("Input column data types and missing counts:")

    for col in df.columns:
        missing_count = int(df[col].isna().sum())

        print(
            f"  {col:<22}"
            f"Type: {str(df[col].dtype):<12}"
            f"Missing: {missing_count}"
        )

    print()

    return df


# =============================================================================
# 3. NUMERICAL DATA VALIDATION
# =============================================================================

def validate_numerical_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Validate numerical columns and identify invalid values.

    For each numerical feature:
        - Count genuine missing values before conversion.
        - Convert values using pd.to_numeric().
        - Detect non-missing values that fail numeric conversion.
        - Detect positive and negative infinity.
        - Convert invalid numerical values and infinities to NaN.
        - Preserve valid finite numerical values.

    This function does not impute missing values.
    """

    print("=" * 78)
    print("STEP 2: NUMERICAL DATA VALIDATION")
    print("=" * 78)

    df_work = df.copy()

    audit_results = {}

    for col in NUMERICAL_COLUMNS:

        raw_series = df_work[col]

        total_records = len(raw_series)

        # Genuine missing values before numeric conversion.
        original_missing_mask = raw_series.isna()

        original_missing_count = int(
            original_missing_mask.sum()
        )

        # Convert numeric-looking values to numbers.
        coerced_series = pd.to_numeric(
            raw_series,
            errors="coerce",
        )

        # Values that were originally present but could not be converted.
        invalid_conversion_mask = (
            raw_series.notna()
            & coerced_series.isna()
        )

        invalid_conversion_count = int(
            invalid_conversion_mask.sum()
        )

        # Detect positive or negative infinity.
        infinity_mask = pd.Series(
            np.isinf(
                coerced_series.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            ),
            index=coerced_series.index,
        )

        infinity_count = int(infinity_mask.sum())

        # Invalid entries include conversion failures and infinities.
        total_invalid_count = (
            invalid_conversion_count + infinity_count
        )

        # Convert infinity to NaN so it cannot distort the calculations.
        cleaned_series = coerced_series.copy()

        if infinity_count > 0:
            cleaned_series.loc[infinity_mask] = np.nan

        # Count available finite observations after sanitization.
        valid_finite_mask = cleaned_series.notna()

        valid_observation_count = int(
            valid_finite_mask.sum()
        )

        final_missing_count = int(
            cleaned_series.isna().sum()
        )

        # Update the working copy, not the original DataFrame.
        df_work[col] = cleaned_series

        audit_results[col] = {
            "total_records": total_records,
            "original_missing_count": original_missing_count,
            "invalid_conversion_count": invalid_conversion_count,
            "infinity_count": infinity_count,
            "total_invalid_count": total_invalid_count,
            "valid_observation_count": valid_observation_count,
            "final_missing_count": final_missing_count,
        }

        if total_invalid_count == 0:
            status_label = "[OK]"
        else:
            status_label = "[WARNING]"

        print(f"{status_label} {col}")

        print(
            f"  Original missing values       : "
            f"{original_missing_count}"
        )

        print(
            f"  Invalid numeric conversions   : "
            f"{invalid_conversion_count}"
        )

        print(
            f"  Infinite numerical values     : "
            f"{infinity_count}"
        )

        print(
            f"  Valid finite observations     : "
            f"{valid_observation_count}/{total_records}"
        )

        print(
            f"  Missing after sanitization    : "
            f"{final_missing_count}"
        )

        print()

    return df_work, audit_results


# =============================================================================
# 4. CALCULATE ORIGINAL FEATURE STATISTICS
# =============================================================================

def calculate_feature_statistics(
    df_work: pd.DataFrame,
    audit_results: dict,
) -> dict:
    """
    Calculate descriptive statistics for valid finite observations.

    Statistics include:
        - Total records
        - Valid observations
        - Original missing values
        - Invalid numerical entries
        - Final unavailable values
        - Minimum
        - Maximum
        - Mean
        - Median
        - Sample standard deviation
        - Range

    If no valid values exist, the feature is marked as not normalizable.
    """

    print("=" * 78)
    print("STEP 3: ORIGINAL FEATURE STATISTICS")
    print("=" * 78)

    stats_dict = {}

    for col in NUMERICAL_COLUMNS:

        valid_series = df_work[col].dropna()

        audit = audit_results[col]

        valid_count = len(valid_series)

        unit = FEATURE_UNITS[col]

        if valid_count == 0:

            print(
                f"[WARNING] {col} contains no valid numerical observations."
            )

            stats_dict[col] = {
                "total_records": audit["total_records"],
                "valid_count": 0,
                "original_missing_count": (
                    audit["original_missing_count"]
                ),
                "invalid_conversion_count": (
                    audit["invalid_conversion_count"]
                ),
                "infinity_count": audit["infinity_count"],
                "total_invalid_count": audit["total_invalid_count"],
                "final_missing_count": audit["final_missing_count"],
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "range": np.nan,
                "zero_range": False,
                "can_normalize": False,
                "reason": "NO_VALID_OBSERVATIONS",
            }

            continue

        min_val = float(valid_series.min())

        max_val = float(valid_series.max())

        mean_val = float(valid_series.mean())

        median_val = float(valid_series.median())

        if valid_count > 1:
            std_val = float(valid_series.std(ddof=1))
        else:
            std_val = 0.0

        range_val = float(max_val - min_val)

        zero_range = bool(range_val == 0.0)

        stats_dict[col] = {
            "total_records": audit["total_records"],
            "valid_count": valid_count,
            "original_missing_count": (
                audit["original_missing_count"]
            ),
            "invalid_conversion_count": (
                audit["invalid_conversion_count"]
            ),
            "infinity_count": audit["infinity_count"],
            "total_invalid_count": audit["total_invalid_count"],
            "final_missing_count": audit["final_missing_count"],
            "min": min_val,
            "max": max_val,
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "range": range_val,
            "zero_range": zero_range,
            "can_normalize": True,
            "reason": (
                "ZERO_RANGE_CONSTANT"
                if zero_range
                else "OK"
            ),
        }

        print(f"Feature: {col} ({unit})")

        print(f"  Valid observations : {valid_count}")

        print(f"  Minimum            : {min_val:.6f} {unit}")

        print(f"  Maximum            : {max_val:.6f} {unit}")

        print(f"  Range              : {range_val:.6f} {unit}")

        print(f"  Mean               : {mean_val:.6f} {unit}")

        print(f"  Median             : {median_val:.6f} {unit}")

        print(f"  Standard deviation : {std_val:.6f} {unit}")

        print(f"  Constant feature   : {zero_range}")

        print("-" * 78)

    print()

    return stats_dict


# =============================================================================
# 5. APPLY MIN-MAX NORMALIZATION
# =============================================================================

def apply_min_max_normalization(
    df_work: pd.DataFrame,
    stats_dict: dict,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply Min-Max normalization to the three numerical features.

    Formula:

        X_normalized = (X - X_min) / (X_max - X_min)

    Important behavior:
        - Original numerical columns are preserved.
        - New normalized columns are added.
        - Missing values remain missing.
        - Outliers are preserved.
        - Constant features are assigned 0.0 for valid observations.
        - Invalid entries are reported and are not silently imputed.
        - A feature with invalid entries is marked PARTIAL_WITH_INVALID_VALUES.
        - A feature with genuine missing values is marked
          PARTIAL_WITH_MISSING_VALUES when no invalid values exist.
        - A feature with no valid observations is marked FAILED_NO_VALID_DATA.
    """

    print("=" * 78)
    print("STEP 4: MIN-MAX NORMALIZATION")
    print("=" * 78)

    print("Formula:")
    print("  X_normalized = (X - X_min) / (X_max - X_min)")
    print("Expected normalized range: [0, 1]")

    print("-" * 78)

    df_norm = df_work.copy()

    norm_results = {}

    for col in NUMERICAL_COLUMNS:

        norm_col = NORMALIZED_COLUMN_MAP[col]

        col_stats = stats_dict[col]

        valid_original = df_norm[col].notna()

        if not col_stats["can_normalize"]:

            print(
                f"[FAILED] {col}: no valid numerical observations."
            )

            df_norm[norm_col] = np.nan

            norm_results[col] = {
                "norm_col_name": norm_col,
                "status": "FAILED_NO_VALID_DATA",
                "zero_range": False,
                "norm_min": np.nan,
                "norm_max": np.nan,
                "norm_mean": np.nan,
                "norm_median": np.nan,
                "norm_std": np.nan,
                "norm_missing": len(df_norm),
            }

            continue

        min_val = col_stats["min"]

        max_val = col_stats["max"]

        range_val = col_stats["range"]

        # Handle constant features without dividing by zero.
        if col_stats["zero_range"]:

            normalized_series = pd.Series(
                np.nan,
                index=df_norm.index,
                dtype=float,
            )

            normalized_series.loc[valid_original] = 0.0

            df_norm[norm_col] = normalized_series

            norm_min = 0.0

            norm_max = 0.0

            norm_mean = 0.0

            norm_median = 0.0

            norm_std = 0.0

            norm_missing = int(
                normalized_series.isna().sum()
            )

            status = "CONSTANT_VALUE_ASSIGNED_ZERO"

            print(
                f"[SPECIAL CASE] {col} has a zero range."
            )

            print(
                "  Valid observations are assigned 0.0."
            )

        else:

            # Apply the standard Min-Max transformation.
            normalized_series = (
                df_norm[col] - min_val
            ) / range_val

            df_norm[norm_col] = normalized_series

            valid_norm = normalized_series.dropna()

            if len(valid_norm) > 0:

                norm_min = float(valid_norm.min())

                norm_max = float(valid_norm.max())

                norm_mean = float(valid_norm.mean())

                norm_median = float(valid_norm.median())

                if len(valid_norm) > 1:
                    norm_std = float(valid_norm.std(ddof=1))
                else:
                    norm_std = 0.0

            else:

                norm_min = np.nan

                norm_max = np.nan

                norm_mean = np.nan

                norm_median = np.nan

                norm_std = np.nan

            norm_missing = int(
                normalized_series.isna().sum()
            )

            # Status is based on the state of the source feature.
            if col_stats["total_invalid_count"] > 0:

                status = "PARTIAL_WITH_INVALID_VALUES"

            elif col_stats["original_missing_count"] > 0:

                status = "PARTIAL_WITH_MISSING_VALUES"

            else:

                status = "SUCCESS"

        norm_results[col] = {
            "norm_col_name": norm_col,
            "status": status,
            "zero_range": col_stats["zero_range"],
            "norm_min": norm_min,
            "norm_max": norm_max,
            "norm_mean": norm_mean,
            "norm_median": norm_median,
            "norm_std": norm_std,
            "norm_missing": norm_missing,
        }

        print(f"\nFeature: {col}")

        print(f"  Normalized column : {norm_col}")

        print(
            f"  Original minimum  : {min_val:.6f}"
        )

        print(
            f"  Original maximum  : {max_val:.6f}"
        )

        print(
            f"  Normalized minimum: {norm_min:.8f}"
            if pd.notna(norm_min)
            else "  Normalized minimum: N/A"
        )

        print(
            f"  Normalized maximum: {norm_max:.8f}"
            if pd.notna(norm_max)
            else "  Normalized maximum: N/A"
        )

        print(f"  Missing values    : {norm_missing}")

        print(f"  Normalization status: {status}")

    print()

    return df_norm, norm_results


# =============================================================================
# 6. VALIDATE NORMALIZED FEATURES
# =============================================================================

def validate_normalized_features(
    df_source: pd.DataFrame,
    df_norm: pd.DataFrame,
    norm_results: dict,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    """
    Verify normalized feature integrity.

    Checks:
        1. Normalized values remain within [0, 1], within tolerance.
        2. Non-constant successfully normalized features reach 0 and 1.
        3. Constant features contain 0.0 for valid values.
        4. Missing-value positions are preserved.
        5. Normalized columns contain only finite values or NaN.
        6. Row count and Order_ID sequence are preserved.
    """

    print("=" * 78)
    print("STEP 5: NORMALIZED FEATURE VALIDATION")
    print("=" * 78)

    if len(df_source) != len(df_norm):

        raise ValueError(
            "Row count changed during normalization."
        )

    if not df_source["Order_ID"].equals(df_norm["Order_ID"]):

        raise ValueError(
            "Order_ID sequence changed during normalization."
        )

    for col in NUMERICAL_COLUMNS:

        norm_col = NORMALIZED_COLUMN_MAP[col]

        source_missing_mask = df_source[col].isna()

        normalized_missing_mask = df_norm[norm_col].isna()

        # Verify that missing values were not replaced by zero.
        if not source_missing_mask.equals(normalized_missing_mask):

            raise ValueError(
                f"Missing-value positions changed for {col}."
            )

        info = norm_results[col]

        status = info["status"]

        if status == "FAILED_NO_VALID_DATA":

            if df_norm[norm_col].notna().any():

                raise ValueError(
                    f"{norm_col} contains values despite having no "
                    "valid source observations."
                )

            print(
                f"[VERIFIED] {norm_col}: No valid source data; "
                "normalization correctly skipped."
            )

            continue

        normalized_values = df_norm[norm_col].dropna()

        if len(normalized_values) == 0:

            raise ValueError(
                f"{norm_col} contains no normalized values."
            )

        if not np.isfinite(
            normalized_values.to_numpy(dtype=float)
        ).all():

            raise ValueError(
                f"{norm_col} contains infinite or non-finite values."
            )

        # Confirm that values remain within [0, 1].
        out_of_bounds = normalized_values[
            (normalized_values < -tolerance)
            | (normalized_values > 1.0 + tolerance)
        ]

        if len(out_of_bounds) > 0:

            raise ValueError(
                f"{norm_col} contains values outside [0, 1]."
            )

        # A constant feature should contain only zeroes.
        if info["zero_range"]:

            if not np.allclose(
                normalized_values.to_numpy(dtype=float),
                0.0,
                atol=tolerance,
            ):

                raise ValueError(
                    f"Constant feature {norm_col} does not contain "
                    "only zeroes."
                )

        else:

            # Standard Min-Max scaling should reach both boundaries.
            actual_min = float(normalized_values.min())

            actual_max = float(normalized_values.max())

            if abs(actual_min) > tolerance:

                raise ValueError(
                    f"{norm_col} minimum is not approximately zero."
                )

            if abs(actual_max - 1.0) > tolerance:

                raise ValueError(
                    f"{norm_col} maximum is not approximately one."
                )

        print(
            f"[VERIFIED] {norm_col:<30}"
            f"Min={normalized_values.min():.8f} | "
            f"Max={normalized_values.max():.8f} | "
            f"Status={status}"
        )

        print(
            f"           Missing positions preserved: "
            f"{int(normalized_missing_mask.sum())}"
        )

    print("\n[OK] Normalized feature validation completed.\n")


# =============================================================================
# 7. BUILD AUDIT AND COMPARISON REPORTS
# =============================================================================

def build_summary_reports(
    stats_dict: dict,
    norm_results: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct the normalization summary and comparison reports.

    Summary report:
        Provides detailed numerical auditing and normalization status.

    Comparison report:
        Compares original and normalized descriptive statistics.
    """

    summary_rows = []

    comparison_rows = []

    for col in NUMERICAL_COLUMNS:

        original = stats_dict[col]

        normalized = norm_results[col]

        summary_rows.append(
            {
                "Column_Name": col,
                "Total_Records": original["total_records"],
                "Valid_Observations": original["valid_count"],
                "Original_Missing_Values": (
                    original["original_missing_count"]
                ),
                "Invalid_Numeric_Conversions": (
                    original["invalid_conversion_count"]
                ),
                "Infinite_Values": original["infinity_count"],
                "Total_Invalid_Entries": (
                    original["total_invalid_count"]
                ),
                "Missing_After_Sanitization": (
                    original["final_missing_count"]
                ),
                "Original_Minimum": original["min"],
                "Original_Maximum": original["max"],
                "Original_Mean": original["mean"],
                "Original_Median": original["median"],
                "Original_Standard_Deviation": original["std"],
                "Original_Range": original["range"],
                "Normalization_Method": NORMALIZATION_METHOD,
                "Zero_Range": original["zero_range"],
                "Normalized_Column_Name": (
                    normalized["norm_col_name"]
                ),
                "Normalized_Minimum": normalized["norm_min"],
                "Normalized_Maximum": normalized["norm_max"],
                "Normalized_Mean": normalized["norm_mean"],
                "Normalized_Median": normalized["norm_median"],
                "Normalized_Standard_Deviation": normalized["norm_std"],
                "Normalized_Missing_Values": normalized["norm_missing"],
                "Normalization_Status": normalized["status"],
            }
        )

        comparison_rows.append(
            {
                "Column_Name": col,
                "Original_Minimum": original["min"],
                "Original_Maximum": original["max"],
                "Original_Mean": original["mean"],
                "Original_Median": original["median"],
                "Original_Standard_Deviation": original["std"],
                "Original_Range": original["range"],
                "Normalized_Column_Name": (
                    normalized["norm_col_name"]
                ),
                "Normalized_Minimum": normalized["norm_min"],
                "Normalized_Maximum": normalized["norm_max"],
                "Normalized_Mean": normalized["norm_mean"],
                "Normalized_Median": normalized["norm_median"],
                "Normalized_Standard_Deviation": normalized["norm_std"],
                "Original_Missing_Values": (
                    original["original_missing_count"]
                ),
                "Invalid_Numeric_Entries": (
                    original["total_invalid_count"]
                ),
                "Normalized_Missing_Values": normalized["norm_missing"],
                "Normalization_Status": normalized["status"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    comparison_df = pd.DataFrame(comparison_rows)

    return summary_df, comparison_df


# =============================================================================
# 8. GENERATE VISUALIZATIONS
# =============================================================================

def generate_visualizations(
    df_norm: pd.DataFrame,
    stats_dict: dict,
    boxplot_path: Path,
    distribution_path: Path,
) -> None:
    """
    Generate two high-resolution visualizations.

    Visualization 1:
        Original-versus-normalized boxplots.

        Each feature uses separate left and right vertical axes because
        the original measurements and normalized values use different units.

    Visualization 2:
        Histograms of normalized numerical features, including mean and
        median reference lines.
    """

    print("=" * 78)
    print("STEP 6: VISUALIZATION GENERATION")
    print("=" * 78)

    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # -------------------------------------------------------------------------
    # Visualization 1: Original versus normalized boxplots
    # -------------------------------------------------------------------------

    fig1, axes1 = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        constrained_layout=True,
    )

    fig1.suptitle(
        "Logistics Analysis: Original vs. Min-Max Normalized Features",
        fontsize=15,
        fontweight="bold",
    )

    normalized_colors = [
        "#2b6cb0",
        "#2c7a7b",
        "#805ad5",
    ]

    for index, col in enumerate(NUMERICAL_COLUMNS):

        ax_left = axes1[index]

        ax_right = ax_left.twinx()

        norm_col = NORMALIZED_COLUMN_MAP[col]

        unit = FEATURE_UNITS[col]

        original_data = df_norm[col].dropna()

        normalized_data = df_norm[norm_col].dropna()

        if len(original_data) > 0:

            ax_left.boxplot(
                original_data,
                positions=[1],
                widths=0.45,
                patch_artist=True,
                boxprops={
                    "facecolor": "#cbd5e0",
                    "color": "#2d3748",
                },
                medianprops={
                    "color": "#c53030",
                    "linewidth": 2,
                },
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "#e53e3e",
                    "markersize": 5,
                    "alpha": 0.8,
                },
            )

        if len(normalized_data) > 0:

            ax_right.boxplot(
                normalized_data,
                positions=[2],
                widths=0.45,
                patch_artist=True,
                boxprops={
                    "facecolor": normalized_colors[index],
                    "color": "#1a202c",
                },
                medianprops={
                    "color": "#ffd700",
                    "linewidth": 2,
                },
                flierprops={
                    "marker": "s",
                    "markerfacecolor": "#dd6b20",
                    "markersize": 5,
                    "alpha": 0.8,
                },
            )

        ax_left.set_ylabel(
            f"Original scale ({unit})",
            fontsize=10,
        )

        ax_right.set_ylabel(
            "Normalized scale [0, 1]",
            fontsize=10,
        )

        ax_right.set_ylim(-0.08, 1.12)

        ax_left.set_xlim(0.3, 2.7)

        ax_left.set_xticks([1, 2])

        ax_left.set_xticklabels(
            ["Original", "Normalized"],
            fontsize=9,
        )

        ax_left.set_title(
            f"{col}\nOriginal and normalized distributions",
            fontsize=11,
            fontweight="bold",
        )

        ax_left.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

        ax_right.grid(False)

        # Make the different vertical scales explicit.
        ax_left.text(
            0.02,
            0.02,
            "Left axis: original\nRight axis: normalized",
            transform=ax_left.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "#cbd5e0",
            },
        )

    fig1.savefig(
        boxplot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig1)

    print(
        f"[SAVED] Original-versus-normalized boxplots:\n"
        f"        {boxplot_path}"
    )

    # -------------------------------------------------------------------------
    # Visualization 2: Normalized feature distributions
    # -------------------------------------------------------------------------

    fig2, axes2 = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        constrained_layout=True,
    )

    fig2.suptitle(
        "Distributions of Min-Max Normalized Logistics Features",
        fontsize=15,
        fontweight="bold",
    )

    histogram_colors = [
        "#3182ce",
        "#319795",
        "#805ad5",
    ]

    for index, col in enumerate(NUMERICAL_COLUMNS):

        ax = axes2[index]

        norm_col = NORMALIZED_COLUMN_MAP[col]

        normalized_series = df_norm[norm_col].dropna()

        if len(normalized_series) > 0:

            ax.hist(
                normalized_series,
                bins=15,
                range=(0.0, 1.0),
                color=histogram_colors[index],
                edgecolor="white",
                alpha=0.8,
            )

            mean_value = float(normalized_series.mean())

            median_value = float(normalized_series.median())

            ax.axvline(
                mean_value,
                color="#e53e3e",
                linestyle="--",
                linewidth=1.8,
                label=f"Mean: {mean_value:.3f}",
            )

            ax.axvline(
                median_value,
                color="#d69e2e",
                linestyle="-",
                linewidth=1.8,
                label=f"Median: {median_value:.3f}",
            )

            ax.set_xlim(-0.05, 1.05)

            ax.set_xlabel(
                "Normalized value",
                fontsize=10,
            )

            ax.set_ylabel(
                "Number of records",
                fontsize=10,
            )

            ax.set_title(
                norm_col,
                fontsize=11,
                fontweight="bold",
            )

            ax.legend(fontsize=9)

            ax.grid(
                True,
                linestyle="--",
                alpha=0.4,
            )

        else:

            ax.text(
                0.5,
                0.5,
                "No valid data available",
                ha="center",
                va="center",
            )

            ax.set_title(norm_col)

    fig2.savefig(
        distribution_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig2)

    print(
        f"[SAVED] Normalized feature distributions:\n"
        f"        {distribution_path}"
    )

    print()


# =============================================================================
# 9. OUTPUT EXPORT AND VERIFICATION
# =============================================================================

def export_and_verify_outputs(
    df_source: pd.DataFrame,
    df_norm: pd.DataFrame,
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    paths: tuple,
) -> None:
    """
    Export the normalized dataset and reports, then verify the saved outputs.

    Verifications include:
        - Required files exist and are non-empty.
        - Normalized CSV can be reloaded.
        - Original row count is preserved.
        - Original Order_ID sequence is preserved.
        - All required original columns remain.
        - All normalized columns exist.
        - Non-numerical source columns are unchanged.
        - Original numerical columns are unchanged after sanitization.
        - Missing-value positions are preserved.
        - Valid normalized values remain within [0, 1].
        - Report schemas and feature counts are correct.
    """

    print("=" * 78)
    print("STEP 7: OUTPUT EXPORT AND VERIFICATION")
    print("=" * 78)

    (
        project_root,
        input_csv,
        output_norm_csv,
        summary_csv,
        comparison_csv,
        boxplot_png,
        distribution_png,
    ) = paths

    # Export all CSV files.
    try:

        df_norm.to_csv(
            output_norm_csv,
            index=False,
        )

        summary_df.to_csv(
            summary_csv,
            index=False,
        )

        comparison_df.to_csv(
            comparison_csv,
            index=False,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Failed to export one or more CSV outputs: {exc}"
        ) from exc

    print(f"[EXPORTED] Normalized dataset: {output_norm_csv}")

    print(f"[EXPORTED] Summary report: {summary_csv}")

    print(f"[EXPORTED] Comparison report: {comparison_csv}")

    print("-" * 78)

    print("Checking output files...")

    required_files = [
        ("Normalized dataset", output_norm_csv),
        ("Normalization summary", summary_csv),
        ("Normalization comparison", comparison_csv),
        ("Original-versus-normalized boxplots", boxplot_png),
        ("Normalized feature distributions", distribution_png),
    ]

    for label, filepath in required_files:

        if not filepath.exists():

            raise FileNotFoundError(
                f"Expected output file was not generated: {filepath}"
            )

        if not filepath.is_file():

            raise ValueError(
                f"Expected an output file but found another path: {filepath}"
            )

        file_size = filepath.stat().st_size

        if file_size <= 0:

            raise ValueError(
                f"Output file is empty: {filepath}"
            )

        print(
            f"[VERIFIED] {label:<40} "
            f"{file_size:,} bytes"
        )

    # -------------------------------------------------------------------------
    # Reload and verify the normalized CSV.
    # -------------------------------------------------------------------------

    reloaded_df = pd.read_csv(output_norm_csv)

    if len(reloaded_df) != len(df_source):

        raise ValueError(
            f"Row count mismatch: expected {len(df_source)}, "
            f"found {len(reloaded_df)}."
        )

    if not reloaded_df["Order_ID"].equals(
        df_source["Order_ID"].reset_index(drop=True)
    ):

        raise ValueError(
            "Order_ID sequence changed in the exported normalized dataset."
        )

    # Verify all expected original columns.
    for col in EXPECTED_COLUMNS:

        if col not in reloaded_df.columns:

            raise ValueError(
                f"Original column is missing from exported CSV: {col}"
            )

    # Verify all normalized columns.
    for norm_col in NORMALIZED_COLUMN_MAP.values():

        if norm_col not in reloaded_df.columns:

            raise ValueError(
                f"Normalized column is missing: {norm_col}"
            )

    # Verify that non-numerical columns were preserved.
    for col in NON_NUMERICAL_COLUMNS:

        original_values = (
            df_source[col]
            .reset_index(drop=True)
            .astype("string")
            .fillna("<MISSING>")
        )

        exported_values = (
            reloaded_df[col]
            .astype("string")
            .fillna("<MISSING>")
        )

        if not original_values.equals(exported_values):

            raise ValueError(
                f"Non-numerical source column changed: {col}"
            )

    # Verify original numerical values and missing-value positions.
    for col in NUMERICAL_COLUMNS:

        source_values = (
            df_norm[col]
            .reset_index(drop=True)
            .to_numpy(dtype=float, na_value=np.nan)
        )

        exported_values = reloaded_df[col].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if not np.allclose(
            source_values,
            exported_values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):

            raise ValueError(
                f"Original numerical feature changed during export: {col}"
            )

        norm_col = NORMALIZED_COLUMN_MAP[col]

        normalized_source = (
            df_norm[norm_col]
            .reset_index(drop=True)
            .to_numpy(dtype=float, na_value=np.nan)
        )

        normalized_exported = reloaded_df[norm_col].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if not np.allclose(
            normalized_source,
            normalized_exported,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):

            raise ValueError(
                f"Normalized values changed during export: {norm_col}"
            )

        # Missing positions must remain identical.
        source_missing = np.isnan(source_values)

        exported_normalized_missing = np.isnan(normalized_exported)

        if not np.array_equal(
            source_missing,
            exported_normalized_missing,
        ):

            raise ValueError(
                f"Missing-value positions changed for {norm_col}."
            )

        # Verify normalized bounds.
        normalized_values = reloaded_df[norm_col].dropna()

        if len(normalized_values) > 0:

            if (
                (normalized_values < -FLOAT_TOLERANCE).any()
                or
                (normalized_values > 1.0 + FLOAT_TOLERANCE).any()
            ):

                raise ValueError(
                    f"Values outside [0, 1] found in {norm_col}."
                )

    # -------------------------------------------------------------------------
    # Verify report structures and expected feature counts.
    # -------------------------------------------------------------------------

    expected_summary_columns = [
        "Column_Name",
        "Total_Records",
        "Valid_Observations",
        "Original_Missing_Values",
        "Invalid_Numeric_Conversions",
        "Infinite_Values",
        "Normalization_Method",
        "Normalized_Column_Name",
        "Normalized_Minimum",
        "Normalized_Maximum",
        "Normalization_Status",
    ]

    expected_comparison_columns = [
        "Column_Name",
        "Original_Minimum",
        "Original_Maximum",
        "Original_Mean",
        "Original_Median",
        "Normalized_Column_Name",
        "Normalized_Minimum",
        "Normalized_Maximum",
        "Normalization_Status",
    ]

    for col in expected_summary_columns:

        if col not in summary_df.columns:

            raise ValueError(
                f"Summary report is missing column: {col}"
            )

    for col in expected_comparison_columns:

        if col not in comparison_df.columns:

            raise ValueError(
                f"Comparison report is missing column: {col}"
            )

    if len(summary_df) != len(NUMERICAL_COLUMNS):

        raise ValueError(
            "Summary report does not contain one row per numerical feature."
        )

    if len(comparison_df) != len(NUMERICAL_COLUMNS):

        raise ValueError(
            "Comparison report does not contain one row per numerical feature."
        )

    if summary_df["Column_Name"].tolist() != NUMERICAL_COLUMNS:

        raise ValueError(
            "Summary report contains unexpected feature rows or ordering."
        )

    if comparison_df["Column_Name"].tolist() != NUMERICAL_COLUMNS:

        raise ValueError(
            "Comparison report contains unexpected feature rows or ordering."
        )

    # Reload the reports to confirm that they can be read.
    reloaded_summary = pd.read_csv(summary_csv)

    reloaded_comparison = pd.read_csv(comparison_csv)

    if len(reloaded_summary) != len(NUMERICAL_COLUMNS):

        raise ValueError(
            "Exported summary report has an unexpected number of rows."
        )

    if len(reloaded_comparison) != len(NUMERICAL_COLUMNS):

        raise ValueError(
            "Exported comparison report has an unexpected number of rows."
        )

    print("-" * 78)

    print("[OK] Dataset row count preserved.")

    print("[OK] Order_ID sequence preserved.")

    print("[OK] Original columns preserved.")

    print("[OK] Original numerical values preserved.")

    print("[OK] Non-numerical values preserved.")

    print("[OK] Missing-value positions preserved.")

    print("[OK] Normalized values validated.")

    print("[OK] Both reports have the expected structure.")

    print("[SUCCESS] All output verification checks passed.")

    print()


# =============================================================================
# 10. MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def main() -> int:
    """
    Execute the complete normalization pipeline.

    Pipeline:
        1. Configure project paths.
        2. Load the cleaned dataset.
        3. Validate numerical data.
        4. Calculate original statistics.
        5. Apply Min-Max normalization.
        6. Validate normalized values.
        7. Build reports.
        8. Generate visualizations.
        9. Export and verify outputs.
        10. Print the final execution summary.
    """

    print("\n" + "=" * 78)

    print(
        "LOGISTICS DATA COLLECTION, CLEANING, AND PREPROCESSING"
    )

    print("YuvaIntern Internship Assignment - Week 2")

    print("NORMALIZATION MODULE")

    print("=" * 78)

    try:

        # Step 1: Configure project paths.
        paths = get_project_paths()

        (
            project_root,
            input_csv,
            output_norm_csv,
            summary_csv,
            comparison_csv,
            boxplot_png,
            distribution_png,
        ) = paths

        print(f"Project root: {project_root}")

        print()

        # Step 2: Load and validate the cleaned dataset.
        df_source = load_and_validate_cleaned_data(
            input_csv
        )

        # Keep a separate copy for source-integrity checks.
        original_order_ids = (
            df_source["Order_ID"]
            .reset_index(drop=True)
            .copy()
        )

        # Step 3: Validate numerical data.
        df_work, audit_results = validate_numerical_data(
            df_source
        )

        # Step 4: Calculate original statistics.
        stats_dict = calculate_feature_statistics(
            df_work,
            audit_results,
        )

        # Step 5: Apply Min-Max normalization.
        df_norm, norm_results = apply_min_max_normalization(
            df_work,
            stats_dict,
        )

        # Step 6: Verify normalized values and integrity.
        validate_normalized_features(
            df_source=df_work,
            df_norm=df_norm,
            norm_results=norm_results,
        )

        # Explicitly confirm original Order_ID sequence is unchanged.
        if not df_norm["Order_ID"].reset_index(drop=True).equals(
            original_order_ids
        ):

            raise ValueError(
                "Order_ID sequence changed during normalization."
            )

        # Step 7: Build summary and comparison reports.
        summary_df, comparison_df = build_summary_reports(
            stats_dict,
            norm_results,
        )

        # Step 8: Generate visualizations.
        generate_visualizations(
            df_norm=df_norm,
            stats_dict=stats_dict,
            boxplot_path=boxplot_png,
            distribution_path=distribution_png,
        )

        # Step 9: Export and verify all output files.
        export_and_verify_outputs(
            df_source=df_work,
            df_norm=df_norm,
            summary_df=summary_df,
            comparison_df=comparison_df,
            paths=paths,
        )

        # ---------------------------------------------------------------------
        # EXECUTIVE NORMALIZATION SUMMARY
        # ---------------------------------------------------------------------

        total_features = len(NUMERICAL_COLUMNS)

        successful_features = sum(
            1
            for result in norm_results.values()
            if result["status"] == "SUCCESS"
        )

        constant_features = sum(
            1
            for result in norm_results.values()
            if result["zero_range"]
        )

        partial_features = sum(
            1
            for result in norm_results.values()
            if result["status"].startswith("PARTIAL")
        )

        failed_features = sum(
            1
            for result in norm_results.values()
            if result["status"] == "FAILED_NO_VALID_DATA"
        )

        print("=" * 78)

        print("EXECUTIVE NORMALIZATION SUMMARY")

        print("=" * 78)

        print(f"Project root                  : {project_root}")

        print(f"Input dataset                 : {input_csv}")

        print(f"Normalized dataset            : {output_norm_csv}")

        print(f"Original record count         : {len(df_source)}")

        print(f"Normalized record count       : {len(df_norm)}")

        print(f"Numerical features processed  : {total_features}")

        print(f"Fully successful features     : {successful_features}")

        print(f"Partially processed features  : {partial_features}")

        print(f"Constant features             : {constant_features}")

        print(f"Features without valid data   : {failed_features}")

        print()

        print("Output files:")

        print(f"  1. {output_norm_csv}")

        print(f"  2. {summary_csv}")

        print(f"  3. {comparison_csv}")

        print(f"  4. {boxplot_png}")

        print(f"  5. {distribution_png}")

        print()

        print("Original cleaned dataset: UNMODIFIED")

        print("Statistical outliers    : PRESERVED")

        if failed_features > 0:

            print(
                "\n[WARNING] At least one feature could not be normalized."
            )

            print(
                "Review normalization_summary.csv before using the data."
            )

            print("=" * 78)

            return 1

        if partial_features > 0:

            print(
                "\n[WARNING] The pipeline completed, but some features "
                "contain missing or invalid numerical entries."
            )

            print(
                "Check normalization_summary.csv and "
                "normalization_comparison.csv."
            )

            print(
                "These features are not marked as fully successful."
            )

        else:

            print(
                "\n[SUCCESS] All numerical features were normalized "
                "successfully."
            )

        print()

        print(
            "[SUCCESS] NORMALIZATION PIPELINE COMPLETED."
        )

        print(
            "All required output files were saved and verified."
        )

        print(
            "The original cleaned dataset remains unchanged."
        )

        print(
            "Outliers were preserved for analysis."
        )

        print("=" * 78)

        print(
            "Next: Review the terminal output before proceeding "
            "to the final report module."
        )

        print("=" * 78)

        return 0

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTED] Script execution cancelled by the user."
        )

        return 130

    except Exception as exc:

        print(
            f"\n[ERROR] Normalization pipeline failed: {exc}"
        )

        print(
            "Review the error above before running the script again."
        )

        return 1


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    exit_code = main()

    sys.exit(exit_code)