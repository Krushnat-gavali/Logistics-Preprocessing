"""
Main Controller for Logistics Data Preprocessing
================================================

Project:
    Logistics Data Collection, Cleaning and Preprocessing

Purpose:
    Run the existing preprocessing scripts in the correct order
    from a single Python file.

Default workflow:
    1. Data Collection
    2. Data Exploration
    3. Data Cleaning
    4. Outlier Detection
    5. Normalization

Optional:
    Recreate the sample dataset using --create-dataset.

Important:
    The original dataset is not recreated by default.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ------------------------------------------------------------
# STEP 1: LOCATE THE PROJECT DIRECTORY
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ------------------------------------------------------------
# STEP 2: DEFINE THE REQUIRED PYTHON SCRIPTS
# ------------------------------------------------------------

CREATE_DATASET_SCRIPT = PROJECT_ROOT / "create_dataset.py"

DATA_COLLECTION_SCRIPT = PROJECT_ROOT / "data_collection.py"

DATA_EXPLORATION_SCRIPT = PROJECT_ROOT / "data_exploration.py"

DATA_CLEANING_SCRIPT = PROJECT_ROOT / "data_cleaning.py"

OUTLIER_DETECTION_SCRIPT = PROJECT_ROOT / "outlier_detection.py"

NORMALIZATION_SCRIPT = PROJECT_ROOT / "normalization.py"


# ------------------------------------------------------------
# STEP 3: DISPLAY PROJECT INFORMATION
# ------------------------------------------------------------

def display_project_information():
    """
    Display the project directory and important file locations.
    """

    print("\n" + "=" * 65)
    print("       LOGISTICS DATA PREPROCESSING PROJECT")
    print("=" * 65)

    print("\nProject Information")
    print("-" * 65)

    print(f"Project Directory : {PROJECT_ROOT}")
    print(f"Data Directory    : {DATA_DIR}")
    print(f"Outputs Directory : {OUTPUTS_DIR}")
    print(f"Python Executable : {sys.executable}")

    print("\nThe preprocessing workflow is ready to begin.")
    print("=" * 65)


# ------------------------------------------------------------
# STEP 4: CHECK THE PROJECT STRUCTURE
# ------------------------------------------------------------

def check_project_structure():
    """
    Check whether the required folders and Python scripts exist.

    Return:
        True  - if all required items are available.
        False - if an item is missing.
    """

    print("\n[CHECK] Checking the project structure...")

    required_directories = [
        DATA_DIR,
        OUTPUTS_DIR,
    ]

    required_scripts = [
        DATA_COLLECTION_SCRIPT,
        DATA_EXPLORATION_SCRIPT,
        DATA_CLEANING_SCRIPT,
        OUTLIER_DETECTION_SCRIPT,
        NORMALIZATION_SCRIPT,
    ]

    missing_items = []

    # Check required directories.
    for directory in required_directories:

        if directory.is_dir():

            print(f"[OK] Folder found: {directory.name}")

        else:

            print(f"[ERROR] Folder missing: {directory}")

            missing_items.append(directory)

    # Check required Python scripts.
    for script in required_scripts:

        if script.is_file():

            print(f"[OK] Script found: {script.name}")

        else:

            print(f"[ERROR] Script missing: {script.name}")

            missing_items.append(script)

    # Check the optional dataset-generation script.
    if CREATE_DATASET_SCRIPT.is_file():

        print("[OK] Script found: create_dataset.py")

    else:

        print(
            "[WARNING] create_dataset.py was not found. "
            "Dataset recreation will not be available."
        )

    if missing_items:

        print("\n[ERROR] Project structure check failed.")

        print(
            "Please check the missing folders or files "
            "before running the workflow."
        )

        return False

    print("\n[SUCCESS] Project structure check completed.")

    return True


# ------------------------------------------------------------
# STEP 5: RUN ONE PYTHON SCRIPT
# ------------------------------------------------------------

def run_script(script_path, step_number, total_steps):
    """
    Execute one Python script as a separate process.

    Parameters:
        script_path:
            Full path of the Python script.

        step_number:
            Current step number.

        total_steps:
            Total number of steps.

    Return:
        True  - if the script completes successfully.
        False - if the script fails.
    """

    script_name = script_path.name

    print("\n" + "=" * 65)

    print(
        f"STEP {step_number} OF {total_steps}: "
        f"{script_name}"
    )

    print("=" * 65)

    print(f"[RUNNING] {script_name}")

    print("-" * 65)

    try:

        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=str(PROJECT_ROOT),
            check=False,
        )

        if result.returncode == 0:

            print("-" * 65)

            print(f"[SUCCESS] {script_name} completed successfully.")

            return True

        print("-" * 65)

        print(
            f"[ERROR] {script_name} stopped with "
            f"exit code {result.returncode}."
        )

        print(
            "[STOPPED] The remaining steps will not run. "
            "Please review the error above."
        )

        return False

    except OSError as error:

        print("\n[ERROR] Python could not start the script.")

        print(f"Details: {error}")

        return False


# ------------------------------------------------------------
# STEP 6: RUN THE COMPLETE PREPROCESSING WORKFLOW
# ------------------------------------------------------------

def run_preprocessing_workflow(recreate_dataset=False):
    """
    Run all preprocessing stages in the intended order.

    By default, the existing raw dataset is used.

    If recreate_dataset is True, create_dataset.py runs first.
    This may replace the existing raw CSV file.
    """

    # Define the scripts in execution order.
    workflow = []

    # Dataset recreation is optional.
    if recreate_dataset:

        if not CREATE_DATASET_SCRIPT.is_file():

            print(
                "\n[ERROR] Cannot recreate the dataset because "
                "create_dataset.py does not exist."
            )

            return False

        workflow.append(CREATE_DATASET_SCRIPT)

    # Add the regular preprocessing stages.
    workflow.extend([
        DATA_COLLECTION_SCRIPT,
        DATA_EXPLORATION_SCRIPT,
        DATA_CLEANING_SCRIPT,
        OUTLIER_DETECTION_SCRIPT,
        NORMALIZATION_SCRIPT,
    ])

    total_steps = len(workflow)

    print("\n" + "=" * 65)

    print("STARTING DATA PREPROCESSING WORKFLOW")

    print("=" * 65)

    if recreate_dataset:

        print(
            "\n[NOTICE] Dataset recreation is enabled."
        )

        print(
            "[NOTICE] create_dataset.py may overwrite "
            "the existing raw dataset."
        )

    else:

        print(
            "\n[NOTICE] The existing raw dataset will be used."
        )

        print(
            "[NOTICE] create_dataset.py will not run "
            "unless --create-dataset is specified."
        )

    # Execute each stage.
    for step_number, script_path in enumerate(
        workflow,
        start=1,
    ):

        success = run_script(
            script_path=script_path,
            step_number=step_number,
            total_steps=total_steps,
        )

        # Stop immediately if a stage fails.
        if not success:

            print("\n" + "=" * 65)

            print("WORKFLOW FAILED")

            print("=" * 65)

            print(
                f"Failed script: {script_path.name}"
            )

            print(
                "Correct the problem and run main.py again."
            )

            return False

    print("\n" + "=" * 65)

    print("ALL PREPROCESSING STEPS COMPLETED")

    print("=" * 65)

    print("\nFinal Project Locations")

    print("-" * 65)

    print(f"Original Dataset   : {DATA_DIR / 'logistics_data.csv'}")

    print(
        "Cleaned Dataset    : "
        f"{DATA_DIR / 'logistics_data_cleaned.csv'}"
    )

    print(
        "Normalized Dataset : "
        f"{DATA_DIR / 'logistics_data_normalized.csv'}"
    )

    print(f"Output Files       : {OUTPUTS_DIR}")

    print("\nThe preprocessing workflow finished successfully.")

    print(
        "Please check the generated CSV files and charts "
        "to verify the results."
    )

    print("\n" + "=" * 65)

    return True


# ------------------------------------------------------------
# STEP 7: DEFINE THE MAIN PROGRAM
# ------------------------------------------------------------

def main():
    """
    Entry point for the main preprocessing controller.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run the logistics data preprocessing workflow."
        )
    )

    parser.add_argument(
        "--create-dataset",
        action="store_true",
        help=(
            "Run create_dataset.py before preprocessing. "
            "This may overwrite the existing raw dataset."
        ),
    )

    args = parser.parse_args()

    # Display project details.
    display_project_information()

    # Verify the project structure.
    if not check_project_structure():

        print("\nPlease fix the project structure and try again.")

        raise SystemExit(1)

    # Confirm that the raw dataset exists when it is needed.
    raw_dataset = DATA_DIR / "logistics_data.csv"

    if not args.create_dataset and not raw_dataset.is_file():

        print("\n[ERROR] The original dataset was not found.")

        print(f"Expected location: {raw_dataset}")

        print(
            "\nIf you intend to generate a new sample dataset, "
            "first check that create_dataset.py is correct."
        )

        print(
            "Then run: py main.py --create-dataset"
        )

        raise SystemExit(1)

    # Run the workflow.
    success = run_preprocessing_workflow(
        recreate_dataset=args.create_dataset
    )

    if not success:

        raise SystemExit(1)

    print("\n[EXIT] Main program finished successfully.")


# ------------------------------------------------------------
# STEP 8: START THE PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    main()