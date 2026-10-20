"""
Week 2: Data Collection, Cleaning, and Preprocessing for Logistics Analysis

File: data_collection.py
Purpose: Load, inspect, and validate the logistics dataset.
"""

from pathlib import Path
import sys
import pandas as pd


# Expected columns in the logistics dataset
EXPECTED_COLUMNS = [
    "Order_ID",
    "Transport_Mode",
    "Distance_KM",
    "Delivery_Days",
    "Shipping_Cost",
    "Customer_City"
]


def get_dataset_path():
    """Return the path of the original logistics CSV file."""

    # Locate the main project folder
    project_folder = Path(__file__).resolve().parent

    # Build the dataset path
    dataset_path = project_folder / "data" / "logistics_data.csv"

    return dataset_path


def load_and_inspect_data():
    """Load the dataset and display basic information."""

    dataset_path = get_dataset_path()

    print("\n" + "=" * 55)
    print("LOGISTICS DATA COLLECTION AND INSPECTION")
    print("=" * 55)

    print(f"\nDataset location: {dataset_path}")

    # Check whether the dataset exists
    if not dataset_path.is_file():
        print("\nError: The dataset file was not found.")
        print("Please check that logistics_data.csv is inside the data folder.")
        sys.exit(1)

    # Load the CSV file
    try:
        df = pd.read_csv(dataset_path)

    except pd.errors.EmptyDataError:
        print("\nError: The CSV file is empty.")
        sys.exit(1)

    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        print(f"\nError reading the dataset: {error}")
        sys.exit(1)

    print("\nDataset loaded successfully.")

    # Display dataset dimensions
    print("\n1. DATASET SIZE")
    print("-" * 35)
    print(f"Total rows    : {df.shape[0]}")
    print(f"Total columns : {df.shape[1]}")

    # Display the first five records
    print("\n2. FIRST FIVE RECORDS")
    print("-" * 35)

    if df.empty:
        print("The dataset contains column headers but no data rows.")
    else:
        print(df.head().to_string(index=False))

    # Display column names and data types
    print("\n3. COLUMN NAMES AND DATA TYPES")
    print("-" * 35)
    df.info()

    # Check for expected columns
    print("\n4. COLUMN VALIDATION")
    print("-" * 35)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Some expected columns are missing:")
        for column in missing_columns:
            print(f"- {column}")
    else:
        print("All six expected columns are present.")

    # Display missing-value counts
    print("\n5. MISSING VALUES")
    print("-" * 35)

    missing_values = df.isnull().sum()
    print(missing_values.to_string())

    total_missing = int(missing_values.sum())
    print(f"\nTotal missing values: {total_missing}")

    # Count duplicate rows
    print("\n6. DUPLICATE RECORDS")
    print("-" * 35)

    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows found: {duplicate_count}")

    # Display a summary
    print("\n7. INSPECTION SUMMARY")
    print("-" * 35)
    print("Dataset loading completed.")
    print("Original dataset has not been modified.")

    print("\n" + "=" * 55)
    print("DATA COLLECTION AND INSPECTION FINISHED")
    print("=" * 55)

    return df


if __name__ == "__main__":
    load_and_inspect_data()