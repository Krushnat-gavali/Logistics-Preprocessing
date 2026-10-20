"""
Project: Week 2 - Data Collection, Cleaning, and Preprocessing
Purpose: Generate a sample logistics dataset for preprocessing practice.

Dataset details:
- 65 shipment records
- 6 columns
- 2 duplicate rows
- 4 missing numerical values
- 3 intentionally unusual numerical values

Note: This dataset is simulated for educational purposes.
Shipping costs are expressed in Indian Rupees (INR).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_logistics_dataset():
    """Generate and save a sample logistics dataset."""

    # Use a fixed random seed to generate reproducible data.
    rng = np.random.default_rng(42)

    # Transport modes and Indian destination cities.
    transport_modes = ["Road", "Rail", "Air"]

    cities = [
        "Mumbai",
        "Pune",
        "Delhi",
        "Bengaluru",
        "Hyderabad",
        "Chennai",
        "Kolkata",
        "Ahmedabad",
    ]

    # Generate 63 initial shipment records.
    number_of_records = 63

    order_ids = [
        f"ORD-{1001 + i}"
        for i in range(number_of_records)
    ]

    transport = rng.choice(
        transport_modes,
        size=number_of_records,
        p=[0.55, 0.30, 0.15],
    )

    customer_cities = rng.choice(
        cities,
        size=number_of_records,
    )

    # Generate numerical shipment information.
    distances = rng.integers(
        120, 1800, size=number_of_records
    ).astype(float)

    delivery_days = rng.integers(
        1, 9, size=number_of_records
    ).astype(float)

    shipping_costs = np.round(
        distances * 0.45
        + rng.uniform(25, 120, size=number_of_records),
        2,
    )

    # Combine the generated information into a DataFrame.
    df = pd.DataFrame(
        {
            "Order_ID": order_ids,
            "Transport_Mode": transport,
            "Distance_KM": distances,
            "Delivery_Days": delivery_days,
            "Shipping_Cost": shipping_costs,
            "Customer_City": customer_cities,
        }
    )

    # Introduce unusual values for later outlier analysis.
    df.loc[10, "Delivery_Days"] = 42.0
    df.loc[25, "Shipping_Cost"] = 3450.00
    df.loc[40, "Distance_KM"] = 6800.0

    # Introduce missing numerical values.
    df.loc[5, "Distance_KM"] = np.nan
    df.loc[18, "Distance_KM"] = np.nan
    df.loc[32, "Delivery_Days"] = np.nan
    df.loc[48, "Shipping_Cost"] = np.nan

    # Append two existing records to create duplicate rows.
    duplicate_records = df.iloc[[2, 7]].copy()

    df = pd.concat(
        [df, duplicate_records],
        ignore_index=True,
    )

    # Create the data folder inside the project directory.
    project_folder = Path(__file__).resolve().parent
    data_folder = project_folder / "data"

    data_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Define the CSV output path.
    csv_path = data_folder / "logistics_data.csv"

    # Validate the dataset before saving it.
    expected_columns = [
        "Order_ID",
        "Transport_Mode",
        "Distance_KM",
        "Delivery_Days",
        "Shipping_Cost",
        "Customer_City",
    ]

    assert len(df) == 65, "Expected exactly 65 rows."
    assert list(df.columns) == expected_columns, (
        "Dataset columns do not match the expected structure."
    )
    assert int(df.duplicated().sum()) == 2, (
        "Expected exactly 2 duplicate rows."
    )
    assert int(df["Distance_KM"].isna().sum()) == 2
    assert int(df["Delivery_Days"].isna().sum()) == 1
    assert int(df["Shipping_Cost"].isna().sum()) == 1

    # Save the dataset without an unnecessary index column.
    df.to_csv(csv_path, index=False)

    # Display a summary of the generated dataset.
    print("\n" + "=" * 55)
    print("LOGISTICS DATASET CREATED SUCCESSFULLY")
    print("=" * 55)

    print(f"\nFile location: {csv_path}")
    print(f"Total records: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values by column:")
    print(df.isnull().sum().to_string())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nDataset validation: PASSED")
    print("Original data-quality issues have been preserved.")
    print("=" * 55)


if __name__ == "__main__":
    generate_logistics_dataset()