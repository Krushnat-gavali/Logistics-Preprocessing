"""
Project: Logistics Data Collection, Cleaning, and Preprocessing
Assignment: YuvaIntern Internship - Week 2

File: outlier_detection.py

Purpose:
1. Load and validate the cleaned logistics dataset.
2. Detect potential numerical outliers using the IQR method.
3. Calculate statistical boundaries for numerical columns.
4. Identify affected records using Order_ID.
5. Generate detailed CSV reports.
6. Create boxplots for visual analysis.
7. Preserve all records in the cleaned dataset.
8. Verify that output files were saved successfully.

Important:
This script identifies potential outliers but does not automatically
remove or modify them.
"""

import sys
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script works without requiring
# a chart window to open.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "Logistics Data Collection, Cleaning, and Preprocessing"
ASSIGNMENT_NAME = "YuvaIntern Internship Assignment - Week 2"

NUMERICAL_COLUMNS = [
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
]

REQUIRED_COLUMNS = [
    "Order_ID",
    "Transport_Mode",
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
    "Customer_City",
]

# Output filenames
OUTLIER_SUMMARY_FILENAME = "outlier_detection_summary.csv"
OUTLIER_RECORDS_FILENAME = "detected_outlier_records.csv"
OUTLIER_CHART_FILENAME = "numerical_outlier_boxplots.png"


# ============================================================
# SECTION 1: PROJECT PATHS
# ============================================================

def get_project_paths():
    """
    Locate the project root and prepare required input/output paths.

    Expected structure:

    Logistics_Preprocessing/
        data/
            logistics_data.csv
            logistics_data_cleaned.csv
        outputs/
        outlier_detection.py
    """

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    cleaned_data_path = data_dir / "logistics_data_cleaned.csv"

    summary_path = outputs_dir / OUTLIER_SUMMARY_FILENAME
    outlier_records_path = outputs_dir / OUTLIER_RECORDS_FILENAME
    boxplot_path = outputs_dir / OUTLIER_CHART_FILENAME

    # Create the output directory if it does not already exist.
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return (
        project_root,
        data_dir,
        outputs_dir,
        cleaned_data_path,
        summary_path,
        outlier_records_path,
        boxplot_path,
    )


# ============================================================
# SECTION 2: LOAD AND VALIDATE DATASET
# ============================================================

def load_cleaned_dataset(cleaned_data_path):
    """
    Load the cleaned dataset and validate its structure.

    The original source dataset is not used as the analysis input.
    This function reads logistics_data_cleaned.csv.
    """

    print("=" * 78)
    print("STEP 1: CLEANED DATASET INGESTION AND VALIDATION")
    print("=" * 78)

    print(f"Input dataset: {cleaned_data_path}")

    if not cleaned_data_path.exists():
        raise FileNotFoundError(
            "\nThe cleaned dataset was not found.\n"
            f"Expected location: {cleaned_data_path}\n"
            "Please run data_cleaning.py successfully before continuing."
        )

    if not cleaned_data_path.is_file():
        raise FileNotFoundError(
            f"The input path is not a file: {cleaned_data_path}"
        )

    try:
        df = pd.read_csv(cleaned_data_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "The cleaned dataset exists but contains no readable data."
        ) from error

    except UnicodeDecodeError as error:
        raise ValueError(
            "The cleaned dataset could not be decoded. "
            "Please check its text encoding."
        ) from error

    except pd.errors.ParserError as error:
        raise ValueError(
            "The cleaned dataset contains a CSV parsing problem."
        ) from error

    if df.empty:
        raise ValueError(
            "The cleaned dataset contains column headers but no records."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The cleaned dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    missing_numerical_columns = [
        column
        for column in NUMERICAL_COLUMNS
        if column not in df.columns
    ]

    if missing_numerical_columns:
        raise ValueError(
            "The following numerical columns are missing: "
            + ", ".join(missing_numerical_columns)
        )

    print("\n[OK] Cleaned dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nMissing values in required columns:")

    for column in REQUIRED_COLUMNS:
        missing_count = int(df[column].isna().sum())

        print(f"  {column:<20}: {missing_count}")

    return df


# ============================================================
# SECTION 3: NUMERICAL DATA VALIDATION
# ============================================================

def validate_numerical_data(df):
    """
    Convert required numerical columns to numeric values and
    validate them before calculating statistical boundaries.

    This function works on a copy and does not overwrite the
    cleaned CSV file.
    """

    print("\n" + "=" * 78)
    print("STEP 2: NUMERICAL DATA VALIDATION")
    print("=" * 78)

    working_df = df.copy()

    invalid_counts = {}
    missing_counts = {}

    for column in NUMERICAL_COLUMNS:

        original_values = working_df[column].copy()

        converted_values = pd.to_numeric(
            original_values,
            errors="coerce",
        )

        # Count values that were present originally but could not
        # be converted into numerical values.
        invalid_mask = (
            original_values.notna()
            & converted_values.isna()
        )

        invalid_count = int(invalid_mask.sum())

        invalid_counts[column] = invalid_count

        working_df[column] = converted_values

        missing_count = int(working_df[column].isna().sum())

        missing_counts[column] = missing_count

        print(
            f"\n{column}:"
            f"\n  Invalid numerical entries: {invalid_count}"
            f"\n  Missing values: {missing_count}"
        )

        if invalid_count > 0:
            print(
                "  [WARNING] Invalid numerical entries were detected. "
                "They will be reported as unavailable for outlier analysis."
            )

        if missing_count > 0:
            print(
                "  [WARNING] Missing numerical values remain in this "
                "input dataset."
            )

    return working_df, invalid_counts, missing_counts


# ============================================================
# SECTION 4: CALCULATE IQR BOUNDARIES
# ============================================================

def calculate_iqr_boundaries(df):
    """
    Calculate Q1, Q3, IQR, and outlier boundaries for each numerical
    column using the standard 1.5 * IQR rule.

    Formula:

        IQR = Q3 - Q1

        Lower Boundary = Q1 - 1.5 * IQR

        Upper Boundary = Q3 + 1.5 * IQR

    Values below the lower boundary or above the upper boundary
    are classified as potential outliers.
    """

    print("\n" + "=" * 78)
    print("STEP 3: IQR-BASED OUTLIER BOUNDARY CALCULATION")
    print("=" * 78)

    print(
        "\nMethod: Interquartile Range (IQR)"
        "\nMultiplier: 1.5"
    )

    boundary_results = {}

    for column in NUMERICAL_COLUMNS:

        series = df[column].dropna()

        print("\n" + "-" * 78)
        print(f"Column: {column}")
        print("-" * 78)

        if series.empty:
            print(
                "[WARNING] No valid numerical observations are available. "
                "Outlier boundaries cannot be calculated."
            )

            boundary_results[column] = {
                "Valid_Observations": 0,
                "Q1": np.nan,
                "Median": np.nan,
                "Q3": np.nan,
                "IQR": np.nan,
                "Lower_Boundary": np.nan,
                "Upper_Boundary": np.nan,
                "Potential_Outlier_Count": 0,
                "Outlier_Percentage": 0.0,
                "Minimum": np.nan,
                "Maximum": np.nan,
                "Mean": np.nan,
                "Standard_Deviation": np.nan,
            }

            continue

        q1 = float(series.quantile(0.25))
        median = float(series.median())
        q3 = float(series.quantile(0.75))

        iqr = q3 - q1

        lower_boundary = q1 - (1.5 * iqr)
        upper_boundary = q3 + (1.5 * iqr)

        outlier_mask = (
            (df[column] < lower_boundary)
            | (df[column] > upper_boundary)
        ).fillna(False)

        outlier_count = int(outlier_mask.sum())

        valid_count = int(series.count())

        outlier_percentage = (
            (outlier_count / valid_count) * 100
            if valid_count > 0
            else 0.0
        )

        boundary_results[column] = {
            "Valid_Observations": valid_count,
            "Q1": q1,
            "Median": median,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Boundary": lower_boundary,
            "Upper_Boundary": upper_boundary,
            "Potential_Outlier_Count": outlier_count,
            "Outlier_Percentage": outlier_percentage,
            "Minimum": float(series.min()),
            "Maximum": float(series.max()),
            "Mean": float(series.mean()),
            "Standard_Deviation": (
                float(series.std())
                if valid_count > 1
                else np.nan
            ),
        }

        print(f"Valid observations : {valid_count}")
        print(f"Q1                 : {q1:.6f}")
        print(f"Median             : {median:.6f}")
        print(f"Q3                 : {q3:.6f}")
        print(f"IQR                : {iqr:.6f}")
        print(f"Lower boundary     : {lower_boundary:.6f}")
        print(f"Upper boundary     : {upper_boundary:.6f}")
        print(f"Minimum value      : {series.min():.6f}")
        print(f"Maximum value      : {series.max():.6f}")
        print(f"Potential outliers : {outlier_count}")
        print(f"Outlier percentage : {outlier_percentage:.2f}%")

    return boundary_results


# ============================================================
# SECTION 5: IDENTIFY POTENTIAL OUTLIER RECORDS
# ============================================================

def identify_outlier_records(df, boundary_results):
    """
    Identify individual records that fall outside the IQR boundaries.

    All detected records are reported with their Order_ID and the
    numerical value that triggered the outlier flag.

    No records are deleted.
    """

    print("\n" + "=" * 78)
    print("STEP 4: IDENTIFY POTENTIAL OUTLIER RECORDS")
    print("=" * 78)

    outlier_records = []

    for column in NUMERICAL_COLUMNS:

        results = boundary_results[column]

        lower_boundary = results["Lower_Boundary"]
        upper_boundary = results["Upper_Boundary"]

        if pd.isna(lower_boundary) or pd.isna(upper_boundary):
            print(
                f"\n[SKIPPED] {column}: boundaries could not be calculated."
            )
            continue

        valid_mask = df[column].notna()

        outlier_mask = valid_mask & (
            (df[column] < lower_boundary)
            | (df[column] > upper_boundary)
        )

        column_outliers = df.loc[outlier_mask]

        print(f"\nColumn: {column}")
        print(f"Potential outlier records: {len(column_outliers)}")

        if column_outliers.empty:
            print("  No potential outliers detected.")
            continue

        for row_index, row in column_outliers.iterrows():

            value = float(row[column])

            if value < lower_boundary:
                outlier_type = "Below Lower Boundary"
            else:
                outlier_type = "Above Upper Boundary"

            record = {
                "Order_ID": row["Order_ID"],
                "Transport_Mode": row["Transport_Mode"],
                "Customer_City": row["Customer_City"],
                "Numerical_Column": column,
                "Observed_Value": value,
                "Lower_Boundary": lower_boundary,
                "Upper_Boundary": upper_boundary,
                "Outlier_Type": outlier_type,
                "Dataset_Row_Index": int(row_index),
            }

            outlier_records.append(record)

            print(
                f"  Order_ID: {row['Order_ID']}"
                f" | Value: {value}"
                f" | Type: {outlier_type}"
            )

    outlier_records_df = pd.DataFrame(
        outlier_records,
        columns=[
            "Order_ID",
            "Transport_Mode",
            "Customer_City",
            "Numerical_Column",
            "Observed_Value",
            "Lower_Boundary",
            "Upper_Boundary",
            "Outlier_Type",
            "Dataset_Row_Index",
        ],
    )

    print("\n" + "-" * 78)
    print(
        "Total outlier flags across numerical columns: "
        f"{len(outlier_records_df)}"
    )

    if not outlier_records_df.empty:
        unique_order_count = outlier_records_df["Order_ID"].nunique()

        print(
            "Unique orders containing at least one outlier flag: "
            f"{unique_order_count}"
        )

    print(
        "\nNote: One order can be flagged in more than one numerical "
        "column. Therefore, total outlier flags and unique affected "
        "orders may differ."
    )

    return outlier_records_df


# ============================================================
# SECTION 6: BUILD OUTLIER SUMMARY REPORT
# ============================================================

def build_outlier_summary(
    boundary_results,
    invalid_counts,
    missing_counts,
    total_rows,
):
    """
    Build a structured report containing statistical boundaries,
    outlier counts, data-quality information, and analysis status.
    """

    summary_records = []

    for column in NUMERICAL_COLUMNS:

        results = boundary_results[column]

        summary_records.append({
            "Column_Name": column,
            "Total_Dataset_Rows": total_rows,
            "Valid_Observations": results["Valid_Observations"],
            "Missing_Values": missing_counts[column],
            "Invalid_Numerical_Entries": invalid_counts[column],
            "Minimum": results["Minimum"],
            "Q1": results["Q1"],
            "Median": results["Median"],
            "Q3": results["Q3"],
            "Maximum": results["Maximum"],
            "Mean": results["Mean"],
            "Standard_Deviation": results["Standard_Deviation"],
            "IQR": results["IQR"],
            "Lower_Boundary": results["Lower_Boundary"],
            "Upper_Boundary": results["Upper_Boundary"],
            "Potential_Outlier_Count": results["Potential_Outlier_Count"],
            "Outlier_Percentage": results["Outlier_Percentage"],
            "Detection_Method": "Interquartile Range (IQR)",
            "IQR_Multiplier": 1.5,
            "Records_Removed": 0,
            "Analysis_Status": "PENDING",
        })

    return pd.DataFrame(summary_records)


# ============================================================
# SECTION 7: CREATE BOXPLOT VISUALIZATION
# ============================================================

def create_outlier_boxplots(df, boxplot_path):
    """
    Create boxplots for all three numerical columns.

    Boxplots help visualize distributions, spread, medians, and
    potential outliers.

    The source dataset is not modified.
    """

    print("\n" + "=" * 78)
    print("STEP 5: OUTLIER VISUALIZATION")
    print("=" * 78)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(17, 6),
    )

    figure.suptitle(
        "Logistics Numerical Data - Outlier Detection Using Boxplots",
        fontsize=14,
        fontweight="bold",
    )

    chart_titles = {
        "Distance_KM": "Distance (KM)",
        "Delivery_Days": "Delivery Duration (Days)",
        "Shipping_Cost": "Shipping Cost (INR)",
    }

    for axis, column in zip(axes, NUMERICAL_COLUMNS):

        values = df[column].dropna()

        if values.empty:
            axis.text(
                0.5,
                0.5,
                "No valid data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axis.transAxes,
            )

            axis.set_title(chart_titles[column])
            axis.set_ylabel("Value")
            continue

        axis.boxplot(
            values,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
        )

        axis.set_title(chart_titles[column])
        axis.set_ylabel("Value")
        axis.set_xticks([1])
        axis.set_xticklabels([column])
        axis.grid(axis="y", linestyle="--", alpha=0.5)

    figure.tight_layout(rect=[0, 0, 1, 0.92])

    try:
        figure.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    if not boxplot_path.exists():
        raise OSError(
            f"The boxplot image was not created: {boxplot_path}"
        )

    if boxplot_path.stat().st_size == 0:
        raise OSError(
            f"The boxplot image is empty: {boxplot_path}"
        )

    print(f"[SAVED] Boxplot visualization: {boxplot_path}")

    return boxplot_path


# ============================================================
# SECTION 8: EXPORT AND VERIFY REPORTS
# ============================================================

def save_and_verify_outputs(
    summary_df,
    outlier_records_df,
    summary_path,
    outlier_records_path,
):
    """
    Save the summary and individual outlier records as CSV files.

    Verify that the generated files exist, contain data in the
    expected format, and can be read back successfully.
    """

    print("\n" + "=" * 78)
    print("STEP 6: SAVE AND VERIFY OUTLIER REPORTS")
    print("=" * 78)

    # Save the summary with PENDING status first.
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save the individual outlier records.
    outlier_records_df.to_csv(
        outlier_records_path,
        index=False,
        encoding="utf-8-sig",
    )

    required_files = [
        summary_path,
        outlier_records_path,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise OSError(
                f"Required output file was not created: {file_path}"
            )

        if file_path.stat().st_size == 0:
            raise OSError(
                f"Required output file is empty: {file_path}"
            )

    # Read files back to verify their structure.
    verified_summary = pd.read_csv(summary_path)

    verified_records = pd.read_csv(outlier_records_path)

    if len(verified_summary) != len(NUMERICAL_COLUMNS):
        raise ValueError(
            "The saved summary report does not contain the expected "
            "number of numerical-column records."
        )

    expected_summary_columns = [
        "Column_Name",
        "Potential_Outlier_Count",
        "Lower_Boundary",
        "Upper_Boundary",
        "Analysis_Status",
    ]

    missing_summary_columns = [
        column
        for column in expected_summary_columns
        if column not in verified_summary.columns
    ]

    if missing_summary_columns:
        raise ValueError(
            "The saved summary report is missing required columns: "
            + ", ".join(missing_summary_columns)
        )

    if len(verified_records) != len(outlier_records_df):
        raise ValueError(
            "The saved outlier-record count does not match the "
            "number of records identified during analysis."
        )

    # Only mark the reports successful after initial file verification.
    summary_df["Analysis_Status"] = "SUCCESSFUL"

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Verify the final status was written successfully.
    final_summary = pd.read_csv(summary_path)

    if not (final_summary["Analysis_Status"] == "SUCCESSFUL").all():
        raise ValueError(
            "The final summary report could not be verified as successful."
        )

    if not summary_path.exists() or summary_path.stat().st_size == 0:
        raise OSError(
            "The final summary report is missing or empty."
        )

    print(f"[SAVED] Outlier summary: {summary_path}")
    print(f"[SAVED] Detected outlier records: {outlier_records_path}")

    print("\n[OK] Both CSV reports were saved and verified.")

    return final_summary, verified_records


# ============================================================
# SECTION 9: PRINT EXECUTIVE SUMMARY
# ============================================================

def print_executive_summary(
    df,
    boundary_results,
    outlier_records_df,
    summary_path,
    outlier_records_path,
    boxplot_path,
):
    """
    Display a concise final report after successful file exports.
    """

    print("\n" + "=" * 78)
    print("EXECUTIVE OUTLIER DETECTION SUMMARY")
    print("=" * 78)

    print(f"Dataset rows analyzed       : {len(df)}")
    print(f"Numerical columns analyzed  : {len(NUMERICAL_COLUMNS)}")

    total_flags = len(outlier_records_df)

    unique_affected_orders = (
        outlier_records_df["Order_ID"].nunique()
        if not outlier_records_df.empty
        else 0
    )

    print(f"Total outlier flags         : {total_flags}")
    print(f"Unique affected orders      : {unique_affected_orders}")

    print("\nOutlier results by numerical column:")

    for column in NUMERICAL_COLUMNS:

        results = boundary_results[column]

        print(
            f"  {column:<20}: "
            f"{results['Potential_Outlier_Count']} potential outlier(s)"
        )

    print("\nGenerated output files:")

    print(f"  Summary report     : {summary_path}")
    print(f"  Outlier records    : {outlier_records_path}")
    print(f"  Boxplot chart      : {boxplot_path}")

    print("\nData preservation:")
    print("  Records automatically removed: 0")
    print("  Cleaned dataset overwritten   : No")

    print("=" * 78)
    print("[SUCCESS] OUTLIER DETECTION COMPLETED SUCCESSFULLY.")
    print("Potential outliers have been reported, not automatically removed.")
    print("The cleaned dataset remains unchanged.")
    print("=" * 78)


# ============================================================
# SECTION 10: MAIN EXECUTION PIPELINE
# ============================================================

def main():
    """
    Execute the full outlier detection workflow in the correct order.
    """

    print("\n" + "=" * 78)
    print(PROJECT_NAME.upper())
    print(ASSIGNMENT_NAME)
    print("=" * 78)

    (
        project_root,
        data_dir,
        outputs_dir,
        cleaned_data_path,
        summary_path,
        outlier_records_path,
        boxplot_path,
    ) = get_project_paths()

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {outputs_dir}")

    # Step 1: Load and validate the cleaned dataset.
    df = load_cleaned_dataset(cleaned_data_path)

    # Step 2: Validate numerical columns.
    working_df, invalid_counts, missing_counts = validate_numerical_data(df)

    # Step 3: Calculate IQR boundaries and outlier statistics.
    boundary_results = calculate_iqr_boundaries(working_df)

    # Step 4: Identify individual potential outlier records.
    outlier_records_df = identify_outlier_records(
        working_df,
        boundary_results,
    )

    # Step 5: Build the summary report.
    summary_df = build_outlier_summary(
        boundary_results,
        invalid_counts,
        missing_counts,
        len(working_df),
    )

    # Step 6: Create boxplot visualizations.
    create_outlier_boxplots(
        working_df,
        boxplot_path,
    )

    # Step 7: Save and verify both CSV reports.
    save_and_verify_outputs(
        summary_df,
        outlier_records_df,
        summary_path,
        outlier_records_path,
    )

    # Step 8: Print the final executive summary.
    print_executive_summary(
        working_df,
        boundary_results,
        outlier_records_df,
        summary_path,
        outlier_records_path,
        boxplot_path,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Execution was stopped by the user.")
        sys.exit(1)

    except Exception as error:
        print("\n" + "=" * 78)
        print("[ERROR] OUTLIER DETECTION PIPELINE FAILED.")
        print("=" * 78)
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        print(
            "\nPlease review the error message and check the input dataset, "
            "file paths, and required Python libraries."
        )
        sys.exit(1)