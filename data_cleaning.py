"""
Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis
YuvaIntern Internship Assignment

Script:
    data_cleaning.py

Purpose:
    Clean, sanitize, and validate the logistics dataset.

    The program:
    1. Loads and validates the original logistics dataset.
    2. Audits missing values and blank categorical strings.
    3. Removes exact duplicate rows from a working copy.
    4. Validates numerical data and identifies invalid entries.
    5. Sanitizes categorical columns.
    6. Imputes missing numerical values using the median.
    7. Preserves statistical outliers for separate analysis.
    8. Audits Order_ID integrity.
    9. Generates detailed before-and-after cleaning reports.
    10. Saves the cleaned dataset without overwriting the source CSV.

Input:
    data/logistics_data.csv

Outputs:
    data/logistics_data_cleaned.csv
    outputs/data_cleaning_summary.csv
    outputs/missing_values_before_after.csv

Python Version:
    3.14.3
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Mandatory columns defining the logistics dataset schema.
EXPECTED_COLUMNS = [
    "Order_ID",
    "Transport_Mode",
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
    "Customer_City"
]

# Numerical attributes that require numeric validation
# and median imputation.
NUMERICAL_COLUMNS = [
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost"
]

# Categorical attributes that require whitespace sanitization.
CATEGORICAL_COLUMNS = [
    "Transport_Mode",
    "Customer_City"
]


# ============================================================
# STEP 1: PROJECT PATH CONFIGURATION
# ============================================================

def get_project_paths() -> tuple[Path, Path, Path, Path, Path]:
    """
    Discover project paths relative to this Python script.

    This approach does not depend on the current terminal
    working directory.

    Returns:
        A tuple containing:
        - Project root directory
        - Original dataset path
        - Cleaned dataset path
        - Cleaning summary path
        - Missing-value comparison report path
    """

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    # Create required directories if they do not exist.
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = data_dir / "logistics_data.csv"

    cleaned_csv_path = (
        data_dir / "logistics_data_cleaned.csv"
    )

    summary_csv_path = (
        outputs_dir / "data_cleaning_summary.csv"
    )

    missing_comparison_csv_path = (
        outputs_dir / "missing_values_before_after.csv"
    )

    return (
        project_root,
        raw_csv_path,
        cleaned_csv_path,
        summary_csv_path,
        missing_comparison_csv_path
    )


# ============================================================
# STEP 2: LOAD AND VALIDATE THE ORIGINAL DATASET
# ============================================================

def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load the original logistics dataset and validate its schema.

    The original CSV is only read. It is never modified or
    overwritten by this function.

    Args:
        csv_path: Path to the original CSV file.

    Returns:
        A DataFrame containing the original dataset.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the dataset is empty or has missing
                    mandatory columns.
        RuntimeError: If the CSV cannot be read.
    """

    print("=" * 78)
    print("STEP 1: DATASET INGESTION AND SOURCE VALIDATION")
    print("=" * 78)

    print(f"Original dataset: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Original dataset was not found:\n{csv_path}\n"
            "Please run create_dataset.py first."
        )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"The dataset path is not a file:\n{csv_path}"
        )

    try:
        df = pd.read_csv(csv_path)

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The CSV file contains no readable data: {csv_path}"
        ) from exc

    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Encoding error while reading the CSV file: {exc}"
        ) from exc

    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(
            f"Failed to read the original CSV file: {exc}"
        ) from exc

    if df.empty:
        raise ValueError(
            "The loaded dataset contains zero records."
        )

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset does not contain all mandatory columns.\n"
            f"Missing columns: {missing_columns}\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Actual columns: {actual_columns}"
        )

    total_rows, total_columns = df.shape

    exact_duplicate_count = int(
        df.duplicated().sum()
    )

    actual_missing_count = int(
        df.isnull().sum().sum()
    )

    print("\n[OK] Original dataset loaded successfully.")

    print(
        f"Original dimensions: "
        f"{total_rows} rows x {total_columns} columns"
    )

    print(
        f"Exact duplicate rows: {exact_duplicate_count}"
    )

    print(
        f"Actual missing cells: {actual_missing_count}"
    )

    print("\nOriginal column information:")

    for column in df.columns:
        print(
            f"  {column:<20} "
            f"Data type: {str(df[column].dtype):<12} "
            f"Non-missing: {df[column].count()}/{total_rows}"
        )

    print("\nOriginal missing values by column:")

    print(
        df.isnull().sum().to_string()
    )

    print()

    return df


# ============================================================
# STEP 3: AUDIT THE ORIGINAL DATASET
# ============================================================

def audit_raw_data(
    df: pd.DataFrame
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Count actual missing values and blank categorical strings
    before any cleaning or conversion.

    Actual missing values and blank strings are tracked
    separately to avoid misleading audit reports.

    Args:
        df: Original dataset.

    Returns:
        A tuple containing:
        - Actual missing-value counts by column.
        - Blank categorical-string counts by column.
    """

    print("=" * 78)
    print("STEP 2: ORIGINAL DATA QUALITY AUDIT")
    print("=" * 78)

    raw_missing_counts: dict[str, int] = {}
    raw_blank_string_counts: dict[str, int] = {}

    for column in EXPECTED_COLUMNS:

        # Count genuine missing values in the original CSV.
        raw_missing_counts[column] = int(
            df[column].isnull().sum()
        )

        # Count whitespace-only strings separately.
        if column in CATEGORICAL_COLUMNS:

            blank_mask = (
                df[column].notna()
                & df[column].astype(str).str.strip().eq("")
            )

            raw_blank_string_counts[column] = int(
                blank_mask.sum()
            )

        else:
            raw_blank_string_counts[column] = 0

    print("\nOriginal data-quality audit:")

    for column in EXPECTED_COLUMNS:

        print(
            f"  {column:<20} "
            f"Missing: {raw_missing_counts[column]:<5} "
            f"Blank strings: "
            f"{raw_blank_string_counts[column]}"
        )

    total_missing = sum(
        raw_missing_counts.values()
    )

    total_blank_strings = sum(
        raw_blank_string_counts.values()
    )

    print("-" * 78)

    print(
        f"Total original missing values: {total_missing}"
    )

    print(
        f"Total original blank categorical strings: "
        f"{total_blank_strings}"
    )

    print()

    return (
        raw_missing_counts,
        raw_blank_string_counts
    )


# ============================================================
# STEP 4: REMOVE EXACT DUPLICATE ROWS
# ============================================================

def remove_exact_duplicates(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from a working copy.

    Rows are considered duplicates only when all column values
    match according to pandas' duplicate-row comparison.

    Records are not removed merely because Order_ID values match.

    Args:
        df: Original dataset.

    Returns:
        A tuple containing:
        - Deduplicated working DataFrame.
        - Number of exact duplicate rows removed.
    """

    print("=" * 78)
    print("STEP 3: EXACT DUPLICATE REMOVAL")
    print("=" * 78)

    initial_row_count = len(df)

    duplicate_mask = df.duplicated(
        keep="first"
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count > 0:

        print(
            f"Found {duplicate_count} exact duplicate row(s)."
        )

        duplicate_samples = df.loc[
            duplicate_mask
        ]

        for _, row in duplicate_samples.iterrows():

            print(
                "  Removing duplicate record: "
                f"Order_ID={row['Order_ID']}, "
                f"Mode={row['Transport_Mode']}, "
                f"Cost={row['Shipping_Cost']}"
            )

        df_clean = (
            df.drop_duplicates(keep="first")
            .copy()
            .reset_index(drop=True)
        )

    else:

        print("No exact duplicate rows found.")

        df_clean = (
            df.copy()
            .reset_index(drop=True)
        )

    final_row_count = len(df_clean)

    print(
        f"\nRows before deduplication: {initial_row_count}"
    )

    print(
        f"Rows after deduplication: {final_row_count}"
    )

    print(
        f"Duplicate rows removed: {duplicate_count}"
    )

    print()

    return df_clean, duplicate_count


# ============================================================
# STEP 5: VALIDATE NUMERICAL AND CATEGORICAL DATA
# ============================================================

def validate_and_sanitize_data(
    df: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, int]
]:
    """
    Validate numerical and categorical attributes.

    Numerical processing:
        - Identifies invalid non-missing numeric entries.
        - Converts valid values into numerical types.
        - Converts invalid entries to NaN.
        - Records invalid conversion counts accurately.

    Categorical processing:
        - Preserves genuine missing values.
        - Removes leading and trailing whitespace.
        - Converts whitespace-only values to NaN.
        - Does not invent replacement categories.

    Args:
        df: Deduplicated working dataset.

    Returns:
        A tuple containing:
        - Sanitized DataFrame.
        - Invalid numerical conversion counts.
        - Blank categorical-string counts in the working data.
    """

    print("=" * 78)
    print("STEP 4: DATA TYPE VALIDATION AND SANITIZATION")
    print("=" * 78)

    df_clean = df.copy()

    invalid_numeric_counts: dict[str, int] = {}

    working_blank_string_counts: dict[str, int] = {}

    # --------------------------------------------------------
    # A. NUMERICAL VALIDATION
    # --------------------------------------------------------

    print("\nNumerical column validation:")

    for column in NUMERICAL_COLUMNS:

        original_series = df_clean[column]

        # Identify values that were not missing before conversion.
        originally_non_missing = (
            original_series.notna()
        )

        # Safely convert numerical values.
        converted_series = pd.to_numeric(
            original_series,
            errors="coerce"
        )

        # Invalid entries were present originally but became NaN.
        invalid_mask = (
            originally_non_missing
            & converted_series.isna()
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        invalid_numeric_counts[column] = invalid_count

        df_clean[column] = converted_series

        if invalid_count > 0:

            print(
                f"  [ALERT] {column}: "
                f"{invalid_count} invalid numerical value(s) "
                "converted to missing."
            )

            invalid_examples = (
                original_series.loc[invalid_mask]
                .head(5)
                .tolist()
            )

            print(
                f"          Example invalid values: "
                f"{invalid_examples}"
            )

        else:

            print(
                f"  [OK] {column}: "
                "No invalid numerical entries detected."
            )

    # --------------------------------------------------------
    # B. CATEGORICAL SANITIZATION
    # --------------------------------------------------------

    print("\nCategorical column sanitization:")

    for column in CATEGORICAL_COLUMNS:

        original_series = df_clean[column]

        # Count blank strings before modifying this column.
        blank_mask = (
            original_series.notna()
            & original_series.astype(str).str.strip().eq("")
        )

        blank_count = int(
            blank_mask.sum()
        )

        working_blank_string_counts[column] = blank_count

        # Preserve genuine missing values rather than converting
        # them into the literal strings "nan" or "None".
        def sanitize_string(value):

            if pd.isna(value):
                return np.nan

            cleaned_value = str(value).strip()

            if cleaned_value == "":
                return np.nan

            return cleaned_value

        df_clean[column] = original_series.apply(
            sanitize_string
        )

        if blank_count > 0:

            print(
                f"  [ALERT] {column}: "
                f"{blank_count} whitespace-only value(s) "
                "converted to missing."
            )

        else:

            print(
                f"  [OK] {column}: "
                "No whitespace-only values detected."
            )

    print()

    return (
        df_clean,
        invalid_numeric_counts,
        working_blank_string_counts
    )


# ============================================================
# STEP 6: MEDIAN IMPUTATION
# ============================================================

def impute_numerical_medians(
    df: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    dict[str, int]
]:
    """
    Fill missing numerical values using unrounded medians.

    Medians are calculated from valid numerical values in the
    deduplicated, sanitized working dataset.

    Statistical outliers are preserved.

    If a column contains no valid numerical values, its missing
    values remain unresolved and no artificial median is created.

    Args:
        df: Sanitized working DataFrame.

    Returns:
        A tuple containing:
        - Imputed DataFrame.
        - Median values used for each numerical column.
        - Number of values imputed in each numerical column.
    """

    print("=" * 78)
    print("STEP 5: NUMERICAL MEDIAN IMPUTATION")
    print("=" * 78)

    print(
        "Strategy: Median imputation using unrounded values."
    )

    print(
        "Statistical outliers will not be removed."
    )

    print("-" * 78)

    df_clean = df.copy()

    computed_medians: dict[str, float] = {}

    imputed_counts: dict[str, int] = {}

    for column in NUMERICAL_COLUMNS:

        series = df_clean[column]

        missing_count = int(
            series.isnull().sum()
        )

        valid_count = int(
            series.count()
        )

        # Handle columns without valid numerical values.
        if valid_count == 0:

            print(
                f"\n[WARNING] {column} contains no valid "
                "numerical values."
            )

            print(
                "           Median cannot be calculated. "
                "Missing values will remain unresolved."
            )

            computed_medians[column] = np.nan

            imputed_counts[column] = 0

            continue

        # Calculate the exact median without rounding.
        median_value = float(
            series.median()
        )

        computed_medians[column] = median_value

        if missing_count > 0:

            # Record affected Order_ID values for the audit trail.
            missing_indices = (
                df_clean.index[
                    df_clean[column].isnull()
                ].tolist()
            )

            affected_order_ids = (
                df_clean.loc[
                    missing_indices,
                    "Order_ID"
                ].tolist()
            )

            # Fill missing values with the unrounded median.
            df_clean[column] = (
                df_clean[column].fillna(median_value)
            )

            imputed_counts[column] = missing_count

            if column == "Distance_KM":
                unit = "KM"

            elif column == "Delivery_Days":
                unit = "Days"

            else:
                unit = "INR"

            print(
                f"\n  {column}: "
                f"{missing_count} missing value(s) imputed."
            )

            print(
                f"    Median used: {median_value:.10g} {unit}"
            )

            print(
                f"    Affected Order_ID values: "
                f"{affected_order_ids}"
            )

        else:

            imputed_counts[column] = 0

            print(
                f"\n  {column}: No missing values to impute."
            )

            print(
                f"    Calculated median: {median_value:.10g}"
            )

    print(
        "\n[OUTLIER PRESERVATION] Unusually high numerical "
        "values are retained."
    )

    print(
        "They will be investigated separately in "
        "outlier_detection.py."
    )

    print()

    return (
        df_clean,
        computed_medians,
        imputed_counts
    )


# ============================================================
# STEP 7: ORDER_ID INTEGRITY AUDIT
# ============================================================

def audit_order_ids(
    df: pd.DataFrame
) -> dict[str, int]:
    """
    Audit missing and duplicate Order_ID values.

    Records with duplicate Order_ID values are retained.
    Only exact duplicate rows are removed by the cleaning pipeline.

    Args:
        df: Working dataset after numerical imputation.

    Returns:
        A dictionary containing Order_ID integrity statistics.
    """

    print("=" * 78)
    print("STEP 6: ORDER_ID INTEGRITY AUDIT")
    print("=" * 78)

    missing_ids = int(
        df["Order_ID"].isnull().sum()
    )

    duplicate_ids = int(
        df["Order_ID"].duplicated().sum()
    )

    total_records = len(df)

    unique_ids = int(
        df["Order_ID"].nunique(dropna=True)
    )

    print(
        f"Total working records: {total_records}"
    )

    print(
        f"Unique non-missing Order_ID values: {unique_ids}"
    )

    print(
        f"Missing Order_ID values: {missing_ids}"
    )

    print(
        f"Duplicate Order_ID values after exact-row removal: "
        f"{duplicate_ids}"
    )

    if missing_ids == 0 and duplicate_ids == 0:

        print(
            "[STATUS] Every record has a non-missing, "
            "unique Order_ID."
        )

    else:

        print(
            "[ATTENTION] Order_ID integrity issues remain."
        )

        print(
            "Records are retained for investigation; "
            "they are not deleted solely because IDs match."
        )

    print()

    return {
        "missing_order_ids": missing_ids,
        "duplicate_order_ids": duplicate_ids,
        "unique_order_ids": unique_ids
    }


# ============================================================
# STEP 8: BUILD THE MISSING-VALUE COMPARISON REPORT
# ============================================================

def build_missing_comparison_report(
    raw_missing_counts: dict[str, int],
    raw_blank_string_counts: dict[str, int],
    working_blank_string_counts: dict[str, int],
    invalid_numeric_counts: dict[str, int],
    imputed_counts: dict[str, int],
    df_cleaned: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a column-level missing-value audit report.

    The report distinguishes original missing values from blank
    strings, invalid numerical conversions, imputation, and
    missing values remaining after cleaning.

    Blank-string counts in the original dataset are reported
    separately from those found in the deduplicated working data.

    Args:
        raw_missing_counts: Actual original missing counts.
        raw_blank_string_counts: Original blank-string counts.
        working_blank_string_counts: Blank counts after
                                     duplicate removal.
        invalid_numeric_counts: Invalid numerical conversion counts.
        imputed_counts: Number of numerical values imputed.
        df_cleaned: Final cleaned working DataFrame.

    Returns:
        A DataFrame containing the column-level audit.
    """

    records = []

    for column in EXPECTED_COLUMNS:

        original_missing = raw_missing_counts.get(
            column,
            0
        )

        original_blanks = raw_blank_string_counts.get(
            column,
            0
        )

        working_blanks = working_blank_string_counts.get(
            column,
            0
        )

        invalid_numeric = invalid_numeric_counts.get(
            column,
            0
        )

        values_imputed = imputed_counts.get(
            column,
            0
        )

        remaining_missing = int(
            df_cleaned[column].isnull().sum()
        )

        if remaining_missing == 0:
            status = "No Missing Values Remaining"
        else:
            status = "Missing Values Remain"

        records.append({
            "Column_Name": column,
            "Original_Missing_Values": original_missing,
            "Original_Blank_Strings": original_blanks,
            "Working_Blank_Strings_Before_Sanitization": (
                working_blanks
            ),
            "Invalid_Numeric_Entries": invalid_numeric,
            "Values_Imputed": values_imputed,
            "Remaining_Missing_Values": remaining_missing,
            "Cleaning_Status": status
        })

    return pd.DataFrame(records)


# ============================================================
# STEP 9: BUILD THE EXECUTIVE CLEANING SUMMARY
# ============================================================

def build_cleaning_summary(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    raw_missing_counts: dict[str, int],
    raw_blank_string_counts: dict[str, int],
    working_blank_string_counts: dict[str, int],
    invalid_numeric_counts: dict[str, int],
    imputed_counts: dict[str, int],
    computed_medians: dict[str, float],
    duplicates_removed: int,
    order_id_audit: dict[str, int]
) -> pd.DataFrame:
    """
    Construct an executive summary of the cleaning process.

    The execution status is set to PENDING until all output files
    have been successfully written.

    Args:
        df_raw: Original dataset.
        df_cleaned: Final cleaned working dataset.
        raw_missing_counts: Original missing counts.
        raw_blank_string_counts: Original blank-string counts.
        working_blank_string_counts: Blank strings after
                                     duplicate removal.
        invalid_numeric_counts: Invalid numerical conversion counts.
        imputed_counts: Number of numerical values imputed.
        computed_medians: Median used for each numerical column.
        duplicates_removed: Number of exact duplicate rows removed.
        order_id_audit: Order_ID integrity statistics.

    Returns:
        A DataFrame with summary metrics.
    """

    total_raw_missing = sum(
        raw_missing_counts.values()
    )

    total_original_blanks = sum(
        raw_blank_string_counts.values()
    )

    total_working_blanks = sum(
        working_blank_string_counts.values()
    )

    total_invalid_numeric = sum(
        invalid_numeric_counts.values()
    )

    total_imputed = sum(
        imputed_counts.values()
    )

    total_remaining_missing = int(
        df_cleaned.isnull().sum().sum()
    )

    duplicate_ids_before = int(
        df_raw["Order_ID"].duplicated().sum()
    )

    duplicate_ids_after = int(
        df_cleaned["Order_ID"].duplicated().sum()
    )

    summary_records = [
        {
            "Metric": "Original row count",
            "Value": len(df_raw)
        },
        {
            "Metric": "Final cleaned row count",
            "Value": len(df_cleaned)
        },
        {
            "Metric": "Original column count",
            "Value": df_raw.shape[1]
        },
        {
            "Metric": "Final column count",
            "Value": df_cleaned.shape[1]
        },
        {
            "Metric": "Exact duplicate rows removed",
            "Value": duplicates_removed
        },
        {
            "Metric": "Original missing-value count",
            "Value": total_raw_missing
        },
        {
            "Metric": "Original blank categorical strings",
            "Value": total_original_blanks
        },
        {
            "Metric": "Blank categorical strings in deduplicated data",
            "Value": total_working_blanks
        },
        {
            "Metric": "Invalid numeric entries converted to missing",
            "Value": total_invalid_numeric
        },
        {
            "Metric": "Numerical values imputed",
            "Value": total_imputed
        },
        {
            "Metric": "Remaining missing values",
            "Value": total_remaining_missing
        },
        {
            "Metric": "Missing Order_ID values after cleaning",
            "Value": order_id_audit["missing_order_ids"]
        },
        {
            "Metric": "Duplicate Order_ID count before cleaning",
            "Value": duplicate_ids_before
        },
        {
            "Metric": "Duplicate Order_ID count after cleaning",
            "Value": duplicate_ids_after
        },
        {
            "Metric": "Unique non-missing Order_ID values after cleaning",
            "Value": order_id_audit["unique_order_ids"]
        }
    ]

    # Include the calculated median for each numerical attribute.
    for column in NUMERICAL_COLUMNS:

        median_value = computed_medians.get(
            column,
            np.nan
        )

        summary_records.append({
            "Metric": f"Median used for {column}",
            "Value": median_value
        })

    # This status will be changed only after all files are saved.
    summary_records.append({
        "Metric": "Final execution status",
        "Value": "PENDING"
    })

    return pd.DataFrame(summary_records)


# ============================================================
# STEP 10: EXPORT ALL REQUIRED FILES
# ============================================================

def save_and_export_all(
    df_raw: pd.DataFrame,
    df_cleaned: pd.DataFrame,
    raw_missing_counts: dict[str, int],
    raw_blank_string_counts: dict[str, int],
    working_blank_string_counts: dict[str, int],
    invalid_numeric_counts: dict[str, int],
    imputed_counts: dict[str, int],
    computed_medians: dict[str, float],
    duplicates_removed: int,
    order_id_audit: dict[str, int],
    cleaned_csv_path: Path,
    summary_csv_path: Path,
    missing_comparison_csv_path: Path
) -> None:
    """
    Export the cleaned dataset and two audit reports.

    The program reports success only after all three output
    files have been saved successfully.

    The original dataset is never overwritten.

    Raises:
        RuntimeError: If an output file cannot be written.
    """

    print("=" * 78)
    print("STEP 7: EXPORT CLEANED DATA AND AUDIT REPORTS")
    print("=" * 78)

    # --------------------------------------------------------
    # A. Prepare both reports before writing output files.
    # --------------------------------------------------------

    comparison_df = build_missing_comparison_report(
        raw_missing_counts=raw_missing_counts,
        raw_blank_string_counts=raw_blank_string_counts,
        working_blank_string_counts=working_blank_string_counts,
        invalid_numeric_counts=invalid_numeric_counts,
        imputed_counts=imputed_counts,
        df_cleaned=df_cleaned
    )

    summary_df = build_cleaning_summary(
        df_raw=df_raw,
        df_cleaned=df_cleaned,
        raw_missing_counts=raw_missing_counts,
        raw_blank_string_counts=raw_blank_string_counts,
        working_blank_string_counts=working_blank_string_counts,
        invalid_numeric_counts=invalid_numeric_counts,
        imputed_counts=imputed_counts,
        computed_medians=computed_medians,
        duplicates_removed=duplicates_removed,
        order_id_audit=order_id_audit
    )

    # --------------------------------------------------------
    # B. Write the cleaned dataset.
    # --------------------------------------------------------

    try:

        df_cleaned.to_csv(
            cleaned_csv_path,
            index=False
        )

        if not cleaned_csv_path.is_file():
            raise OSError(
                "Cleaned dataset was not created."
            )

        print(
            f"[SAVED] Cleaned dataset:\n"
            f"        {cleaned_csv_path}"
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to save the cleaned dataset. "
            f"Details: {exc}"
        ) from exc

    # --------------------------------------------------------
    # C. Write the missing-value comparison report.
    # --------------------------------------------------------

    try:

        comparison_df.to_csv(
            missing_comparison_csv_path,
            index=False
        )

        if not missing_comparison_csv_path.is_file():
            raise OSError(
                "Missing-value comparison report was not created."
            )

        print(
            f"[SAVED] Missing-value comparison report:\n"
            f"        {missing_comparison_csv_path}"
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to save the missing-value comparison report. "
            f"Details: {exc}"
        ) from exc

    # --------------------------------------------------------
    # D. Write the summary initially with PENDING status.
    # --------------------------------------------------------

    try:

        summary_df.to_csv(
            summary_csv_path,
            index=False
        )

        if not summary_csv_path.is_file():
            raise OSError(
                "Cleaning summary report was not created."
            )

        print(
            f"[SAVED] Cleaning summary report:\n"
            f"        {summary_csv_path}"
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed to save the cleaning summary report. "
            f"Details: {exc}"
        ) from exc

    # --------------------------------------------------------
    # E. Verify all three files exist and are readable.
    # --------------------------------------------------------

    output_paths = [
        cleaned_csv_path,
        missing_comparison_csv_path,
        summary_csv_path
    ]

    for output_path in output_paths:

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):

            raise RuntimeError(
                "Output verification failed. "
                f"Missing or empty file: {output_path}"
            )

    try:

        # Read the files back to verify their CSV structure.
        cleaned_check = pd.read_csv(
            cleaned_csv_path
        )

        comparison_check = pd.read_csv(
            missing_comparison_csv_path
        )

        summary_check = pd.read_csv(
            summary_csv_path
        )

        if cleaned_check.empty:
            raise ValueError(
                "The saved cleaned dataset has no records."
            )

        if "Column_Name" not in comparison_check.columns:
            raise ValueError(
                "Missing-value report has an invalid schema."
            )

        if not {"Metric", "Value"}.issubset(
            summary_check.columns
        ):
            raise ValueError(
                "Cleaning summary has an invalid schema."
            )

    except Exception as exc:

        raise RuntimeError(
            "Output-file verification failed. "
            f"Details: {exc}"
        ) from exc

    # --------------------------------------------------------
    # F. Mark the summary successful only after the other
    #    files have been written and verified.
    # --------------------------------------------------------

    try:

        status_mask = (
            summary_df["Metric"]
            == "Final execution status"
        )

        summary_df.loc[
            status_mask,
            "Value"
        ] = "SUCCESSFUL"

        summary_df.to_csv(
            summary_csv_path,
            index=False
        )

        # Verify the final status was actually saved.
        saved_summary = pd.read_csv(
            summary_csv_path
        )

        saved_status = saved_summary.loc[
            saved_summary["Metric"]
            == "Final execution status",
            "Value"
        ]

        if (
            saved_status.empty
            or saved_status.iloc[0] != "SUCCESSFUL"
        ):

            raise RuntimeError(
                "The final success status could not be verified."
            )

    except Exception as exc:

        raise RuntimeError(
            "Failed to finalize the cleaning summary status. "
            f"Details: {exc}"
        ) from exc

    print(
        "\n[OK] All required output files were saved and verified."
    )

    print(
        "[OK] Original data/logistics_data.csv was not overwritten."
    )

    print()


# ============================================================
# STEP 11: MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Execute the complete logistics data-cleaning pipeline.

    Processing order:
        1. Configure project paths.
        2. Load the original dataset.
        3. Audit original missing values and blank strings.
        4. Remove exact duplicate rows.
        5. Validate and sanitize data types.
        6. Impute missing numerical values using medians.
        7. Audit Order_ID integrity.
        8. Generate and save the cleaned dataset.
        9. Export and verify all audit reports.
        10. Display the final execution summary.
    """

    try:

        # ----------------------------------------------------
        # Configure project directories and output paths.
        # ----------------------------------------------------

        (
            project_root,
            raw_csv_path,
            cleaned_csv_path,
            summary_csv_path,
            missing_comparison_csv_path
        ) = get_project_paths()

        print("\n" + "=" * 78)

        print(
            "LOGISTICS DATA COLLECTION, CLEANING, "
            "AND PREPROCESSING"
        )

        print(
            "YuvaIntern Internship Assignment - Week 2"
        )

        print("=" * 78)

        print(
            f"Project root: {project_root}"
        )

        # ----------------------------------------------------
        # Step 1: Load the source dataset.
        # ----------------------------------------------------

        df_raw = load_raw_dataset(
            raw_csv_path
        )

        # ----------------------------------------------------
        # Step 2: Audit the original dataset before modification.
        # ----------------------------------------------------

        raw_missing_counts, raw_blank_string_counts = (
            audit_raw_data(df_raw.copy())
        )

        # ----------------------------------------------------
        # Step 3: Remove exact duplicate rows.
        # ----------------------------------------------------

        df_deduplicated, duplicates_removed = (
            remove_exact_duplicates(df_raw)
        )

        # ----------------------------------------------------
        # Step 4: Validate and sanitize the working dataset.
        # ----------------------------------------------------

        (
            df_sanitized,
            invalid_numeric_counts,
            working_blank_string_counts
        ) = validate_and_sanitize_data(
            df_deduplicated
        )

        # ----------------------------------------------------
        # Step 5: Impute missing numerical values.
        # ----------------------------------------------------

        (
            df_imputed,
            computed_medians,
            imputed_counts
        ) = impute_numerical_medians(
            df_sanitized
        )

        # ----------------------------------------------------
        # Step 6: Audit Order_ID integrity.
        # ----------------------------------------------------

        order_id_audit = audit_order_ids(
            df_imputed
        )

        # ----------------------------------------------------
        # Step 7: Export and verify all required output files.
        # ----------------------------------------------------

        save_and_export_all(
            df_raw=df_raw,
            df_cleaned=df_imputed,
            raw_missing_counts=raw_missing_counts,
            raw_blank_string_counts=raw_blank_string_counts,
            working_blank_string_counts=working_blank_string_counts,
            invalid_numeric_counts=invalid_numeric_counts,
            imputed_counts=imputed_counts,
            computed_medians=computed_medians,
            duplicates_removed=duplicates_removed,
            order_id_audit=order_id_audit,
            cleaned_csv_path=cleaned_csv_path,
            summary_csv_path=summary_csv_path,
            missing_comparison_csv_path=missing_comparison_csv_path
        )

        # ----------------------------------------------------
        # Final terminal summary.
        # Reached only after successful output verification.
        # ----------------------------------------------------

        total_imputed = sum(
            imputed_counts.values()
        )

        total_remaining_missing = int(
            df_imputed.isnull().sum().sum()
        )

        total_invalid_numeric = sum(
            invalid_numeric_counts.values()
        )

        total_original_missing = sum(
            raw_missing_counts.values()
        )

        print("=" * 78)

        print(
            "EXECUTIVE DATA CLEANING SUMMARY"
        )

        print("=" * 78)

        print(
            f"Original row count             : {len(df_raw)}"
        )

        print(
            f"Final cleaned row count         : {len(df_imputed)}"
        )

        print(
            f"Original column count           : {df_raw.shape[1]}"
        )

        print(
            f"Final column count              : {df_imputed.shape[1]}"
        )

        print(
            f"Exact duplicate rows removed    : {duplicates_removed}"
        )

        print(
            f"Original missing values         : {total_original_missing}"
        )

        print(
            f"Invalid numeric entries         : {total_invalid_numeric}"
        )

        print(
            f"Numerical values imputed        : {total_imputed}"
        )

        print(
            f"Remaining missing values        : {total_remaining_missing}"
        )

        print(
            f"Cleaned dataset                 : {cleaned_csv_path}"
        )

        print(
            f"Cleaning summary                : {summary_csv_path}"
        )

        print(
            f"Missing-value comparison        : "
            f"{missing_comparison_csv_path}"
        )

        print("=" * 78)

        print(
            "[SUCCESS] DATA CLEANING PIPELINE COMPLETED SUCCESSFULLY."
        )

        print(
            "All three output files were saved and verified."
        )

        print(
            "The original dataset remains unchanged."
        )

        print(
            "Statistical outliers have been preserved for "
            "outlier_detection.py."
        )

        print("=" * 78)

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTED] Cleaning process cancelled by user."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            f"\n[ERROR] DATA CLEANING PIPELINE FAILED:\n{exc}"
        )

        print(
            "The final success message was not issued."
        )

        sys.exit(1)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()