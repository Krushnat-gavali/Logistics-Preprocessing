"""
Project Name : Data Collection, Cleaning, and Preprocessing
               for Logistics Analysis

Week         : Week 2
Module       : Exploratory Data Analysis (EDA)
Language     : Python
Libraries    : Pandas, NumPy, Matplotlib

Description:
This module performs exploratory data analysis on a simulated
logistics dataset.

The program:
1. Locates the project directories.
2. Loads and validates the logistics dataset.
3. Examines the dataset structure and data types.
4. Identifies missing values and duplicate records.
5. Generates descriptive statistics for numerical columns.
6. Examines categorical features.
7. Detects numerical outliers using the IQR method.
8. Generates data visualization charts.
9. Exports analysis results to CSV files.
10. Prints a summary of the analysis.

Important:
The dataset is simulated for educational purposes. It does not
represent actual records from a logistics company.

The original dataset is read-only in this module. This script
does not remove duplicates, fill missing values, or modify the
source CSV file. Those operations belong in the data-cleaning
module.
"""

# ============================================================
# SECTION 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so charts can be generated
# without opening separate graphical windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# SECTION 2: PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "Logistics Data Preprocessing Project"

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

CATEGORICAL_COLUMNS = [
    "Transport_Mode",
    "Customer_City",
]

IDENTIFIER_COLUMN = "Order_ID"

# IQR multiplier commonly used for exploratory outlier detection.
IQR_MULTIPLIER = 1.5

# Set the chart size and resolution.
FIGURE_DPI = 150
FIGURE_SIZE = (10, 6)

# Ensure Pandas displays useful information in the terminal.
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 40)


# ============================================================
# SECTION 3: DEFINE OUTPUT REPORT COLUMNS
# ============================================================

MISSING_VALUES_COLUMNS = [
    "Column",
    "Data_Type",
    "Total_Rows",
    "Missing_Count",
    "Missing_Percentage",
    "Non_Missing_Count",
]

NUMERICAL_SUMMARY_COLUMNS = [
    "Column",
    "Data_Type",
    "Total_Rows",
    "Valid_Count",
    "Missing_Count",
    "Missing_Percentage",
    "Mean",
    "Median",
    "Standard_Deviation",
    "Variance",
    "Minimum",
    "First_Quartile_Q1",
    "Third_Quartile_Q3",
    "Maximum",
    "IQR",
    "Skewness",
]

CATEGORICAL_SUMMARY_COLUMNS = [
    "Column",
    "Category",
    "Frequency",
    "Percentage",
    "Total_Rows",
    "Non_Missing_Count",
]

OUTLIER_SUMMARY_COLUMNS = [
    "Column",
    "Valid_Count",
    "Q1",
    "Q3",
    "IQR",
    "Lower_Bound",
    "Upper_Bound",
    "Outlier_Count",
    "Outlier_Percentage",
    "Outlier_Values",
]


# ============================================================
# SECTION 4: LOCATE PROJECT DIRECTORIES
# ============================================================

def get_project_paths():
    """
    Locate the project root, input dataset, and output directory.

    Returns
    -------
    dict
        Dictionary containing the important project paths.
    """

    project_root = Path(__file__).resolve().parent

    data_dir = project_root / "data"

    outputs_dir = project_root / "outputs"

    dataset_path = data_dir / "logistics_data.csv"

    # Create the output directory if it does not exist.
    outputs_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "project_root": project_root,
        "data_dir": data_dir,
        "outputs_dir": outputs_dir,
        "dataset_path": dataset_path,
    }

    print("\n" + "=" * 75)
    print("PROJECT DIRECTORY CONFIGURATION")
    print("=" * 75)

    print(f"Project root   : {project_root}")
    print(f"Data directory : {data_dir}")
    print(f"Output folder  : {outputs_dir}")
    print(f"Dataset path   : {dataset_path}")

    return paths


# ============================================================
# SECTION 5: LOAD AND VALIDATE THE DATASET
# ============================================================

def load_and_validate_dataset(dataset_path):
    """
    Load the logistics dataset and perform basic validation.

    Parameters
    ----------
    dataset_path : pathlib.Path
        Location of the source CSV file.

    Returns
    -------
    pandas.DataFrame
        Validated DataFrame.

    Raises
    ------
    SystemExit
        If the dataset cannot be loaded or required columns
        are missing.
    """

    print("\n" + "=" * 75)
    print("STEP 1: LOAD AND VALIDATE DATASET")
    print("=" * 75)

    if not dataset_path.exists():
        print("\nERROR: Dataset file was not found.")
        print(f"Expected location: {dataset_path}")

        print("\nPlease check that:")
        print("1. The data folder exists.")
        print("2. logistics_data.csv is inside the data folder.")
        print("3. The dataset creation script has been executed.")

        raise SystemExit(1)

    if not dataset_path.is_file():
        print("\nERROR: The specified dataset path is not a file.")
        print(f"Path: {dataset_path}")

        raise SystemExit(1)

    try:
        df = pd.read_csv(dataset_path)

    except pd.errors.EmptyDataError:
        print("\nERROR: The CSV file is empty.")
        print(f"File: {dataset_path}")

        raise SystemExit(1)

    except UnicodeDecodeError:
        print("\nERROR: The CSV file could not be decoded.")
        print("Check the file encoding before running the analysis.")

        raise SystemExit(1)

    except PermissionError:
        print("\nERROR: Permission denied while reading the dataset.")
        print("Close applications that may be locking the file.")

        raise SystemExit(1)

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print("\nERROR: Failed to read the dataset.")
        print(f"Details: {error}")

        raise SystemExit(1)

    except Exception as error:
        print("\nERROR: An unexpected problem occurred while loading data.")
        print(f"Error type: {type(error).__name__}")
        print(f"Details: {error}")

        raise SystemExit(1)

    # Check whether the CSV contains any rows and columns.
    if df.empty:
        print("\nERROR: The dataset contains no data rows.")
        print("Please check the contents of logistics_data.csv.")

        raise SystemExit(1)

    if len(df.columns) == 0:
        print("\nERROR: The dataset does not contain any columns.")

        raise SystemExit(1)

    # Check required columns without changing the original column names.
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Required columns are missing.")

        for column in missing_columns:
            print(f"  - {column}")

        print("\nColumns actually found:")
        print(list(df.columns))

        print("\nPlease check the dataset structure.")

        raise SystemExit(1)

    print("\nDataset loaded successfully.")

    print(f"Number of rows    : {len(df):,}")
    print(f"Number of columns : {len(df.columns):,}")
    print(f"Memory usage      : {df.memory_usage(deep=True).sum():,} bytes")

    print("\nDataset column names:")
    print(list(df.columns))

    print("\nDataset data types:")
    print(df.dtypes.to_string())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nLast five records:")
    print(df.tail().to_string(index=False))

    print("\nDataset information:")
    df.info()

    return df


# ============================================================
# SECTION 6: ANALYZE DATA QUALITY
# ============================================================

def analyze_data_quality(df, outputs_dir):
    """
    Examine missing values, duplicate rows, identifier quality,
    and basic dataset completeness.

    The source DataFrame is not modified.

    Parameters
    ----------
    df : pandas.DataFrame
        Source logistics dataset.

    outputs_dir : pathlib.Path
        Directory for exported analysis reports.

    Returns
    -------
    dict
        Summary of data-quality checks.
    """

    print("\n" + "=" * 75)
    print("STEP 2: DATA QUALITY ANALYSIS")
    print("=" * 75)

    total_rows = len(df)

    total_columns = len(df.columns)

    total_cells = total_rows * total_columns

    missing_cells = int(df.isna().sum().sum())

    missing_percentage = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0.0
    )

    print(f"\nTotal rows              : {total_rows:,}")
    print(f"Total columns           : {total_columns:,}")
    print(f"Total cells             : {total_cells:,}")
    print(f"Missing cells           : {missing_cells:,}")
    print(f"Overall missing percent : {missing_percentage:.2f}%")

    # --------------------------------------------------------
    # 6.1 Missing-value analysis
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("6.1 MISSING-VALUE ANALYSIS")
    print("-" * 65)

    missing_rows = []

    for column in df.columns:

        missing_count = int(df[column].isna().sum())

        non_missing_count = int(df[column].notna().sum())

        column_missing_percentage = (
            (missing_count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        missing_rows.append({
            "Column": column,
            "Data_Type": str(df[column].dtype),
            "Total_Rows": total_rows,
            "Missing_Count": missing_count,
            "Missing_Percentage": column_missing_percentage,
            "Non_Missing_Count": non_missing_count,
        })

        print(
            f"{column:<20} "
            f"Missing: {missing_count:>4} "
            f"({column_missing_percentage:>6.2f}%)"
        )

    missing_summary = pd.DataFrame(
        missing_rows,
        columns=MISSING_VALUES_COLUMNS,
    )

    missing_summary_path = (
        outputs_dir / "missing_values_summary.csv"
    )

    missing_summary.to_csv(
        missing_summary_path,
        index=False,
    )

    print(f"\nMissing-value report saved:")
    print(missing_summary_path)

    # --------------------------------------------------------
    # 6.2 Duplicate-row analysis
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("6.2 DUPLICATE-ROW ANALYSIS")
    print("-" * 65)

    duplicate_mask = df.duplicated()

    duplicate_count = int(duplicate_mask.sum())

    duplicate_percentage = (
        (duplicate_count / total_rows) * 100
        if total_rows > 0
        else 0.0
    )

    print(f"Duplicate rows       : {duplicate_count:,}")
    print(f"Duplicate percentage : {duplicate_percentage:.2f}%")
    print(f"Unique rows          : {total_rows - duplicate_count:,}")

    if duplicate_count > 0:

        print("\nSample duplicate records:")

        duplicate_examples = df.loc[duplicate_mask].head(5)

        print(duplicate_examples.to_string(index=False))

    else:

        print("\nNo exact duplicate rows were detected.")

    # --------------------------------------------------------
    # 6.3 Identifier analysis
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("6.3 IDENTIFIER ANALYSIS")
    print("-" * 65)

    identifier_missing_count = 0

    identifier_duplicate_count = 0

    identifier_unique_count = 0

    if IDENTIFIER_COLUMN in df.columns:

        identifier_series = df[IDENTIFIER_COLUMN]

        identifier_missing_count = int(
            identifier_series.isna().sum()
        )

        identifier_duplicate_count = int(
            identifier_series.dropna().duplicated().sum()
        )

        identifier_unique_count = int(
            identifier_series.nunique(dropna=True)
        )

        print(
            f"Missing {IDENTIFIER_COLUMN} values : "
            f"{identifier_missing_count}"
        )

        print(
            f"Repeated non-missing identifiers   : "
            f"{identifier_duplicate_count}"
        )

        print(
            f"Unique non-missing identifiers     : "
            f"{identifier_unique_count}"
        )

        if identifier_missing_count > 0:
            print(
                "WARNING: Some records do not have an Order_ID."
            )

        if identifier_duplicate_count > 0:
            print(
                "WARNING: Repeated Order_ID values were detected."
            )

    else:

        print(
            f"Identifier column {IDENTIFIER_COLUMN} is unavailable."
        )

    # --------------------------------------------------------
    # 6.4 Overall data-quality assessment
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("6.4 DATA-QUALITY ASSESSMENT")
    print("-" * 65)

    if missing_cells == 0:
        print("Missing-value check: PASSED")
    else:
        print("Missing-value check: ATTENTION REQUIRED")

    if duplicate_count == 0:
        print("Duplicate-row check: PASSED")
    else:
        print("Duplicate-row check: ATTENTION REQUIRED")

    if identifier_missing_count == 0:
        print("Identifier completeness: PASSED")
    else:
        print("Identifier completeness: ATTENTION REQUIRED")

    quality_summary = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": duplicate_percentage,
        "identifier_missing_count": identifier_missing_count,
        "identifier_duplicate_count": identifier_duplicate_count,
        "identifier_unique_count": identifier_unique_count,
        "missing_summary_path": missing_summary_path,
    }

    return quality_summary


# ============================================================
# SECTION 7: ANALYZE NUMERICAL FEATURES
# ============================================================

def analyze_numerical_features(df, outputs_dir):
    """
    Calculate descriptive statistics for numerical columns.

    Statistics include:
    - Valid observations
    - Missing observations
    - Mean
    - Median
    - Standard deviation
    - Variance
    - Minimum
    - First quartile
    - Third quartile
    - Maximum
    - Interquartile range
    - Skewness

    If a numerical column has no valid observations, its
    statistics remain NaN instead of incorrectly becoming zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Logistics dataset.

    outputs_dir : pathlib.Path
        Directory for exported reports.

    Returns
    -------
    pandas.DataFrame
        Numerical summary report.
    """

    print("\n" + "=" * 75)
    print("STEP 3: NUMERICAL FEATURE ANALYSIS")
    print("=" * 75)

    numerical_rows = []

    for column in NUMERICAL_COLUMNS:

        print("\n" + "-" * 65)
        print(f"Analyzing numerical column: {column}")
        print("-" * 65)

        if column not in df.columns:

            print(f"WARNING: Column {column} was not found.")

            continue

        original_series = df[column]

        # Convert values to numeric for analysis.
        # Invalid text values become NaN in this analysis copy.
        numeric_series = pd.to_numeric(
            original_series,
            errors="coerce",
        )

        valid_series = numeric_series.dropna()

        total_rows = len(df)

        valid_count = int(valid_series.count())

        missing_count = total_rows - valid_count

        missing_percentage = (
            (missing_count / total_rows) * 100
            if total_rows > 0
            else 0.0
        )

        if valid_count == 0:

            print(
                "WARNING: This column contains no valid numerical "
                "observations."
            )

            statistics = {
                "Mean": np.nan,
                "Median": np.nan,
                "Standard_Deviation": np.nan,
                "Variance": np.nan,
                "Minimum": np.nan,
                "First_Quartile_Q1": np.nan,
                "Third_Quartile_Q3": np.nan,
                "Maximum": np.nan,
                "IQR": np.nan,
                "Skewness": np.nan,
            }

        else:

            q1 = float(valid_series.quantile(0.25))

            q3 = float(valid_series.quantile(0.75))

            iqr_value = q3 - q1

            statistics = {
                "Mean": float(valid_series.mean()),
                "Median": float(valid_series.median()),
                "Standard_Deviation": float(
                    valid_series.std()
                ),
                "Variance": float(valid_series.var()),
                "Minimum": float(valid_series.min()),
                "First_Quartile_Q1": q1,
                "Third_Quartile_Q3": q3,
                "Maximum": float(valid_series.max()),
                "IQR": float(iqr_value),
                "Skewness": float(valid_series.skew()),
            }

        numerical_rows.append({
            "Column": column,
            "Data_Type": str(original_series.dtype),
            "Total_Rows": total_rows,
            "Valid_Count": valid_count,
            "Missing_Count": missing_count,
            "Missing_Percentage": missing_percentage,
            **statistics,
        })

        print(f"Total observations : {total_rows:,}")
        print(f"Valid observations : {valid_count:,}")
        print(f"Missing values     : {missing_count:,}")

        if valid_count > 0:

            print(f"Mean               : {statistics['Mean']:.4f}")
            print(f"Median             : {statistics['Median']:.4f}")
            print(
                "Standard deviation : "
                f"{statistics['Standard_Deviation']:.4f}"
            )
            print(f"Minimum            : {statistics['Minimum']:.4f}")
            print(f"Maximum            : {statistics['Maximum']:.4f}")
            print(
                "First quartile Q1  : "
                f"{statistics['First_Quartile_Q1']:.4f}"
            )
            print(
                "Third quartile Q3  : "
                f"{statistics['Third_Quartile_Q3']:.4f}"
            )
            print(f"IQR                : {statistics['IQR']:.4f}")

        else:

            print("Descriptive statistics are unavailable.")

    numerical_summary = pd.DataFrame(
        numerical_rows,
        columns=NUMERICAL_SUMMARY_COLUMNS,
    )

    summary_path = outputs_dir / "numerical_summary.csv"

    numerical_summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nNumerical summary report saved:")
    print(summary_path)

    if not numerical_summary.empty:

        print("\nNumerical summary preview:")

        print(
            numerical_summary[
                [
                    "Column",
                    "Valid_Count",
                    "Missing_Count",
                    "Mean",
                    "Median",
                    "Minimum",
                    "Maximum",
                ]
            ].to_string(index=False)
        )

    return numerical_summary


# ============================================================
# SECTION 8: ANALYZE CATEGORICAL FEATURES
# ============================================================

def analyze_categorical_features(df, outputs_dir):
    """
    Analyze categorical columns using frequency distributions.

    Missing values are included as a separate category in the
    report so that missing categories are not silently ignored.

    Percentages are calculated using the total number of rows.
    Therefore, the percentages for each categorical column add
    up to approximately 100%, including missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Logistics dataset.

    outputs_dir : pathlib.Path
        Directory for exported reports.

    Returns
    -------
    pandas.DataFrame
        Categorical frequency summary.
    """

    print("\n" + "=" * 75)
    print("STEP 4: CATEGORICAL FEATURE ANALYSIS")
    print("=" * 75)

    categorical_rows = []

    total_rows = len(df)

    for column in CATEGORICAL_COLUMNS:

        print("\n" + "-" * 65)
        print(f"Analyzing categorical column: {column}")
        print("-" * 65)

        if column not in df.columns:

            print(f"WARNING: Column {column} was not found.")

            continue

        series = df[column]

        non_missing_count = int(series.notna().sum())

        # dropna=False includes missing values in the frequency table.
        frequency_series = series.value_counts(
            dropna=False,
            sort=True,
        )

        if frequency_series.empty:

            print("No categories are available for analysis.")

            continue

        print(f"Total rows       : {total_rows:,}")
        print(f"Non-missing rows : {non_missing_count:,}")
        print(
            f"Missing rows     : "
            f"{total_rows - non_missing_count:,}"
        )
        print(f"Distinct values  : {series.nunique(dropna=True):,}")

        print("\nCategory frequencies:")

        for category, count in frequency_series.items():

            # Use a clear text label for missing categories.
            if pd.isna(category):
                category_label = "<MISSING>"
            else:
                category_label = str(category)

            # Calculate percentage directly from the frequency.
            percentage = (
                (int(count) / total_rows) * 100
                if total_rows > 0
                else 0.0
            )

            print(
                f"  {category_label:<25} "
                f"Count: {int(count):>5} "
                f"Percentage: {percentage:>7.2f}%"
            )

            categorical_rows.append({
                "Column": column,
                "Category": category_label,
                "Frequency": int(count),
                "Percentage": percentage,
                "Total_Rows": total_rows,
                "Non_Missing_Count": non_missing_count,
            })

    categorical_summary = pd.DataFrame(
        categorical_rows,
        columns=CATEGORICAL_SUMMARY_COLUMNS,
    )

    summary_path = outputs_dir / "categorical_summary.csv"

    categorical_summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nCategorical summary report saved:")
    print(summary_path)

    if not categorical_summary.empty:

        print("\nCategorical summary preview:")

        print(
            categorical_summary.head(15).to_string(index=False)
        )

    return categorical_summary


# ============================================================
# SECTION 9: DETECT OUTLIERS USING THE IQR METHOD
# ============================================================

def detect_outliers_iqr(df, outputs_dir):
    """
    Detect numerical outliers using the Interquartile Range method.

    IQR = Q3 - Q1

    Lower bound = Q1 - 1.5 * IQR

    Upper bound = Q3 + 1.5 * IQR

    Values below the lower bound or above the upper bound are
    classified as potential outliers.

    Important:
    Outlier detection identifies unusual values. It does not
    automatically mean that the observations are incorrect.
    Potential outliers should be reviewed before removal.

    Parameters
    ----------
    df : pandas.DataFrame
        Logistics dataset.

    outputs_dir : pathlib.Path
        Directory for exported reports.

    Returns
    -------
    pandas.DataFrame
        Outlier analysis summary.
    """

    print("\n" + "=" * 75)
    print("STEP 5: OUTLIER DETECTION USING THE IQR METHOD")
    print("=" * 75)

    outlier_rows = []

    for column in NUMERICAL_COLUMNS:

        print("\n" + "-" * 65)
        print(f"Outlier analysis for: {column}")
        print("-" * 65)

        if column not in df.columns:

            print(f"WARNING: Column {column} was not found.")

            continue

        # Convert the analysis copy to numeric.
        # The original DataFrame remains unchanged.
        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid_series = numeric_series.dropna()

        valid_count = int(valid_series.count())

        if valid_count == 0:

            print(
                "No valid numerical observations are available. "
                "Outlier bounds cannot be calculated."
            )

            outlier_rows.append({
                "Column": column,
                "Valid_Count": 0,
                "Q1": np.nan,
                "Q3": np.nan,
                "IQR": np.nan,
                "Lower_Bound": np.nan,
                "Upper_Bound": np.nan,
                "Outlier_Count": 0,
                "Outlier_Percentage": np.nan,
                "Outlier_Values": "",
            })

            continue

        # Calculate the first and third quartiles.
        q1 = float(valid_series.quantile(0.25))

        q3 = float(valid_series.quantile(0.75))

        # Calculate the interquartile range.
        iqr_value = q3 - q1

        # Calculate the lower and upper outlier boundaries.
        lower_bound = q1 - (
            IQR_MULTIPLIER * iqr_value
        )

        upper_bound = q3 + (
            IQR_MULTIPLIER * iqr_value
        )

        # Identify potential outliers.
        outlier_mask = (
            (valid_series < lower_bound)
            | (valid_series > upper_bound)
        )

        outliers = valid_series.loc[outlier_mask]

        outlier_count = int(outliers.count())

        outlier_percentage = (
            (outlier_count / valid_count) * 100
            if valid_count > 0
            else 0.0
        )

        # Store the outlier values as a readable string.
        outlier_values = ", ".join(
            str(value)
            for value in outliers.tolist()
        )

        outlier_rows.append({
            "Column": column,
            "Valid_Count": valid_count,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr_value,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": outlier_percentage,
            "Outlier_Values": outlier_values,
        })

        print(f"Valid observations : {valid_count:,}")
        print(f"Q1                 : {q1:.4f}")
        print(f"Q3                 : {q3:.4f}")
        print(f"IQR                : {iqr_value:.4f}")
        print(f"Lower bound        : {lower_bound:.4f}")
        print(f"Upper bound        : {upper_bound:.4f}")
        print(f"Outlier count      : {outlier_count:,}")
        print(f"Outlier percentage : {outlier_percentage:.2f}%")

        if outlier_count > 0:

            print("\nPotential outlier values:")

            for value in outliers.tolist():
                print(f"  {value}")

        else:

            print("\nNo potential outliers were detected.")

    # Explicit column names ensure that the report retains
    # its headers even if no outliers are detected.
    outlier_summary = pd.DataFrame(
        outlier_rows,
        columns=OUTLIER_SUMMARY_COLUMNS,
    )

    summary_path = outputs_dir / "outlier_summary.csv"

    outlier_summary.to_csv(
        summary_path,
        index=False,
    )

    print("\nOutlier summary report saved:")
    print(summary_path)

    if not outlier_summary.empty:

        print("\nOutlier summary preview:")

        print(
            outlier_summary[
                [
                    "Column",
                    "Valid_Count",
                    "Lower_Bound",
                    "Upper_Bound",
                    "Outlier_Count",
                    "Outlier_Percentage",
                ]
            ].to_string(index=False)
        )

    return outlier_summary


# ============================================================
# SECTION 10: CREATE AND SAVE VISUALIZATIONS
# ============================================================

def generate_visualizations(df, outputs_dir):
    """
    Generate exploratory data analysis charts.

    Charts generated:
    1. Distance distribution histogram.
    2. Delivery-days distribution histogram.
    3. Shipping-cost distribution histogram.
    4. Numerical boxplots.
    5. Transport-mode frequency chart.

    All charts are saved as PNG files in the outputs directory.

    Parameters
    ----------
    df : pandas.DataFrame
        Logistics dataset.

    outputs_dir : pathlib.Path
        Directory for saving chart images.

    Returns
    -------
    list
        Paths of successfully generated chart files.
    """

    print("\n" + "=" * 75)
    print("STEP 6: GENERATE DATA VISUALIZATIONS")
    print("=" * 75)

    outputs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_charts = []

    # --------------------------------------------------------
    # 10.1 Generate numerical distribution histograms
    # --------------------------------------------------------

    histogram_configuration = [
        (
            "Distance_KM",
            "Distance Distribution",
            "Distance (KM)",
            "distance_distribution.png",
        ),
        (
            "Delivery_Days",
            "Delivery Time Distribution",
            "Delivery Days",
            "delivery_days_distribution.png",
        ),
        (
            "Shipping_Cost",
            "Shipping Cost Distribution",
            "Shipping Cost (INR)",
            "shipping_cost_distribution.png",
        ),
    ]

    for (
        column,
        chart_title,
        x_label,
        filename,
    ) in histogram_configuration:

        print(f"\nCreating chart: {filename}")

        if column not in df.columns:

            print(
                f"WARNING: Cannot create chart because "
                f"{column} is missing."
            )

            continue

        # Convert values to numeric for visualization.
        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if numeric_series.empty:

            print(
                f"WARNING: No valid numerical data is available "
                f"for {column}."
            )

            continue

        figure, axis = plt.subplots(
            figsize=FIGURE_SIZE,
        )

        try:

            axis.hist(
                numeric_series,
                bins="auto",
                edgecolor="black",
                alpha=0.75,
            )

            mean_value = float(numeric_series.mean())

            median_value = float(numeric_series.median())

            axis.axvline(
                mean_value,
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_value:.2f}",
            )

            axis.axvline(
                median_value,
                linestyle=":",
                linewidth=2,
                label=f"Median: {median_value:.2f}",
            )

            axis.set_title(
                chart_title,
                fontsize=14,
            )

            axis.set_xlabel(
                x_label,
                fontsize=11,
            )

            axis.set_ylabel(
                "Frequency",
                fontsize=11,
            )

            axis.grid(
                axis="y",
                alpha=0.25,
            )

            axis.legend()

            figure.tight_layout()

            chart_path = outputs_dir / filename

            figure.savefig(
                chart_path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            generated_charts.append(chart_path)

            print(f"Chart saved successfully: {chart_path}")

        except Exception as error:

            print(f"WARNING: Failed to create {filename}.")
            print(f"Details: {error}")

        finally:

            plt.close(figure)

    # --------------------------------------------------------
    # 10.2 Generate numerical boxplots
    # --------------------------------------------------------

    print("\nCreating chart: numerical_boxplots.png")

    available_boxplot_columns = []

    boxplot_data = []

    for column in NUMERICAL_COLUMNS:

        if column not in df.columns:
            continue

        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if numeric_series.empty:
            continue

        available_boxplot_columns.append(column)

        boxplot_data.append(
            numeric_series.to_numpy()
        )

    if not boxplot_data:

        print(
            "WARNING: No numerical data is available "
            "for generating boxplots."
        )

    else:

        figure, axis = plt.subplots(
            figsize=(11, 6),
        )

        try:

            # Explicit positions and labels prevent the
            # set_xticklabels-without-ticks warning.
            positions = list(
                range(1, len(available_boxplot_columns) + 1)
            )

            axis.boxplot(
                boxplot_data,
                positions=positions,
                widths=0.5,
                patch_artist=True,
            )

            axis.set_xticks(
                positions,
                labels=available_boxplot_columns,
            )

            axis.set_title(
                "Boxplots of Numerical Logistics Features",
                fontsize=14,
            )

            axis.set_xlabel(
                "Numerical Features",
                fontsize=11,
            )

            axis.set_ylabel(
                "Observed Values",
                fontsize=11,
            )

            axis.grid(
                axis="y",
                alpha=0.25,
            )

            figure.tight_layout()

            chart_path = (
                outputs_dir / "numerical_boxplots.png"
            )

            figure.savefig(
                chart_path,
                dpi=FIGURE_DPI,
                bbox_inches="tight",
            )

            generated_charts.append(chart_path)

            print(f"Chart saved successfully: {chart_path}")

        except Exception as error:

            print(
                "WARNING: Failed to generate numerical boxplots."
            )

            print(f"Details: {error}")

        finally:

            plt.close(figure)

    # --------------------------------------------------------
    # 10.3 Generate transport-mode distribution chart
    # --------------------------------------------------------

    print("\nCreating chart: transport_mode_distribution.png")

    transport_column = "Transport_Mode"

    if transport_column not in df.columns:

        print(
            "WARNING: Transport_Mode column is not available."
        )

    else:

        transport_series = df[transport_column]

        transport_counts = transport_series.value_counts(
            dropna=False,
            sort=True,
        )

        if transport_counts.empty:

            print(
                "WARNING: No transport-mode data is available."
            )

        else:

            # Replace missing values with a readable label.
            category_labels = [
                "<MISSING>" if pd.isna(category)
                else str(category)
                for category in transport_counts.index
            ]

            figure, axis = plt.subplots(
                figsize=FIGURE_SIZE,
            )

            try:

                positions = list(
                    range(len(category_labels))
                )

                bars = axis.bar(
                    positions,
                    transport_counts.to_numpy(),
                    edgecolor="black",
                    alpha=0.8,
                )

                axis.set_xticks(
                    positions,
                    labels=category_labels,
                )

                axis.set_title(
                    "Distribution of Transport Modes",
                    fontsize=14,
                )

                axis.set_xlabel(
                    "Transport Mode",
                    fontsize=11,
                )

                axis.set_ylabel(
                    "Number of Records",
                    fontsize=11,
                )

                axis.grid(
                    axis="y",
                    alpha=0.25,
                )

                # Display the record count above each bar.
                axis.bar_label(
                    bars,
                    padding=3,
                    fontsize=9,
                )

                figure.tight_layout()

                chart_path = (
                    outputs_dir
                    / "transport_mode_distribution.png"
                )

                figure.savefig(
                    chart_path,
                    dpi=FIGURE_DPI,
                    bbox_inches="tight",
                )

                generated_charts.append(chart_path)

                print(
                    f"Chart saved successfully: {chart_path}"
                )

            except Exception as error:

                print(
                    "WARNING: Failed to generate the "
                    "transport-mode chart."
                )

                print(f"Details: {error}")

            finally:

                plt.close(figure)

    # --------------------------------------------------------
    # 10.4 Display visualization results
    # --------------------------------------------------------

    print("\n" + "-" * 65)
    print("VISUALIZATION SUMMARY")
    print("-" * 65)

    print(f"Charts successfully generated: {len(generated_charts)}")

    if generated_charts:

        for chart_path in generated_charts:
            print(f"  - {chart_path.name}")

    else:

        print("No charts were generated.")

    return generated_charts


# ============================================================
# SECTION 11: GENERATE AN ANALYSIS SUMMARY
# ============================================================

def print_analysis_summary(
    df,
    quality_summary,
    numerical_summary,
    categorical_summary,
    outlier_summary,
    generated_charts,
    outputs_dir,
):
    """
    Print a final summary of the exploratory data analysis.

    This function summarizes the work completed without
    changing the source dataset.
    """

    print("\n" + "=" * 75)
    print("FINAL EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 75)

    print(f"\nProject name: {PROJECT_NAME}")

    print("\n1. DATASET OVERVIEW")
    print("-" * 65)

    print(f"Total records        : {len(df):,}")
    print(f"Total columns        : {len(df.columns):,}")
    print(
        f"Missing cells        : "
        f"{quality_summary['missing_cells']:,}"
    )
    print(
        f"Duplicate rows       : "
        f"{quality_summary['duplicate_rows']:,}"
    )
    print(
        f"Missing percentage   : "
        f"{quality_summary['missing_percentage']:.2f}%"
    )

    print("\n2. NUMERICAL ANALYSIS")
    print("-" * 65)

    if numerical_summary.empty:

        print("No numerical summary was generated.")

    else:

        for _, row in numerical_summary.iterrows():

            print(f"\nFeature: {row['Column']}")

            print(f"  Valid observations : {row['Valid_Count']}")

            print(f"  Missing values     : {row['Missing_Count']}")

            if pd.notna(row["Mean"]):
                print(f"  Mean               : {row['Mean']:.4f}")
            else:
                print("  Mean               : Not available")

            if pd.notna(row["Median"]):
                print(f"  Median             : {row['Median']:.4f}")
            else:
                print("  Median             : Not available")

            if pd.notna(row["Standard_Deviation"]):
                print(
                    "  Standard deviation : "
                    f"{row['Standard_Deviation']:.4f}"
                )
            else:
                print("  Standard deviation : Not available")

    print("\n3. CATEGORICAL ANALYSIS")
    print("-" * 65)

    if categorical_summary.empty:

        print("No categorical summary was generated.")

    else:

        for column in CATEGORICAL_COLUMNS:

            column_summary = categorical_summary.loc[
                categorical_summary["Column"] == column
            ]

            if column_summary.empty:
                continue

            print(f"\nFeature: {column}")

            print(
                f"  Categories including missing: "
                f"{len(column_summary):,}"
            )

            print(
                column_summary[
                    ["Category", "Frequency", "Percentage"]
                ].to_string(index=False)
            )

    print("\n4. OUTLIER ANALYSIS")
    print("-" * 65)

    if outlier_summary.empty:

        print("No outlier summary was generated.")

    else:

        for _, row in outlier_summary.iterrows():

            print(f"\nFeature: {row['Column']}")

            if pd.isna(row["Q1"]) or pd.isna(row["Q3"]):

                print("  Outlier bounds could not be calculated.")

                continue

            print(f"  Lower bound   : {row['Lower_Bound']:.4f}")

            print(f"  Upper bound   : {row['Upper_Bound']:.4f}")

            print(f"  Outlier count : {row['Outlier_Count']}")

            print(
                f"  Outlier rate  : "
                f"{row['Outlier_Percentage']:.2f}%"
            )

    print("\n5. GENERATED REPORT FILES")
    print("-" * 65)

    report_files = [
        "missing_values_summary.csv",
        "numerical_summary.csv",
        "categorical_summary.csv",
        "outlier_summary.csv",
    ]

    for filename in report_files:

        report_path = outputs_dir / filename

        if report_path.exists():

            print(f"[OK] {filename}")

        else:

            print(f"[NOT FOUND] {filename}")

    print("\n6. GENERATED VISUALIZATION FILES")
    print("-" * 65)

    if generated_charts:

        for chart_path in generated_charts:

            if chart_path.exists():

                print(f"[OK] {chart_path.name}")

            else:

                print(f"[NOT FOUND] {chart_path.name}")

    else:

        print("No visualization files were generated.")

    print("\n7. IMPORTANT DATA-PREPARATION NOTES")
    print("-" * 65)

    print(
        "1. This module performs exploratory analysis only."
    )

    print(
        "2. The original CSV file is not intentionally modified."
    )

    print(
        "3. Missing values are reported, not filled here."
    )

    print(
        "4. Duplicate rows are reported, not removed here."
    )

    print(
        "5. Potential outliers require review before any removal."
    )

    print(
        "6. Data cleaning and normalization are handled by "
        "separate modules."
    )

    print("\n" + "=" * 75)
    print("EXPLORATORY DATA ANALYSIS COMPLETED")
    print("=" * 75)

    print("\nOutput directory:")
    print(outputs_dir)


# ============================================================
# SECTION 12: MAIN PROGRAM
# ============================================================

def main():
    """
    Execute the complete exploratory data analysis pipeline.

    Execution order:
    1. Locate project paths.
    2. Load and validate the dataset.
    3. Analyze data quality.
    4. Analyze numerical features.
    5. Analyze categorical features.
    6. Detect potential outliers.
    7. Generate visualizations.
    8. Print the final analysis summary.
    """

    print("\n" + "#" * 75)
    print(" " * 13 + "LOGISTICS DATA EXPLORATION")
    print(" " * 15 + "WEEK 2 PROJECT")
    print("#" * 75)

    print(
        "\nThis program performs exploratory data analysis "
        "on the logistics dataset."
    )

    print(
        "The source dataset will not be cleaned or normalized "
        "by this module."
    )

    # --------------------------------------------------------
    # Step 1: Locate project paths
    # --------------------------------------------------------

    paths = get_project_paths()

    dataset_path = paths["dataset_path"]

    outputs_dir = paths["outputs_dir"]

    # --------------------------------------------------------
    # Step 2: Load and validate the dataset
    # --------------------------------------------------------

    df = load_and_validate_dataset(
        dataset_path,
    )

    # --------------------------------------------------------
    # Step 3: Analyze data quality
    # --------------------------------------------------------

    quality_summary = analyze_data_quality(
        df,
        outputs_dir,
    )

    # --------------------------------------------------------
    # Step 4: Analyze numerical features
    # --------------------------------------------------------

    numerical_summary = analyze_numerical_features(
        df,
        outputs_dir,
    )

    # --------------------------------------------------------
    # Step 5: Analyze categorical features
    # --------------------------------------------------------

    categorical_summary = analyze_categorical_features(
        df,
        outputs_dir,
    )

    # --------------------------------------------------------
    # Step 6: Detect potential numerical outliers
    # --------------------------------------------------------

    outlier_summary = detect_outliers_iqr(
        df,
        outputs_dir,
    )

    # --------------------------------------------------------
    # Step 7: Generate visualization charts
    # --------------------------------------------------------

    generated_charts = generate_visualizations(
        df,
        outputs_dir,
    )

    # --------------------------------------------------------
    # Step 8: Print final analysis summary
    # --------------------------------------------------------

    print_analysis_summary(
        df=df,
        quality_summary=quality_summary,
        numerical_summary=numerical_summary,
        categorical_summary=categorical_summary,
        outlier_summary=outlier_summary,
        generated_charts=generated_charts,
        outputs_dir=outputs_dir,
    )


# ============================================================
# SECTION 13: PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n\nProgram interrupted by the user.")

        sys.exit(130)

    except SystemExit:

        # Preserve intentional exits from dataset validation.
        raise

    except Exception as error:

        print("\n" + "!" * 75)
        print("UNEXPECTED ERROR")
        print("!" * 75)

        print(f"Error type: {type(error).__name__}")

        print(f"Error details: {error}")

        print(
            "\nPlease review the error message and check "
            "the dataset and project configuration."
        )

        sys.exit(1)