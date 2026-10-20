# Logistics Data Preprocessing

## 1. Introduction

This project is part of my Week 2 internship assignment on data collection,
cleaning, and preprocessing for logistics analysis.

Logistics companies maintain records of shipments, transport methods,
delivery times, and shipping costs. Sometimes these records contain
missing values, duplicate entries, or unusually high and low values.

The purpose of this project is to use Python to identify these problems
and prepare the data for further analysis.

The project uses a CSV dataset and libraries such as Pandas, NumPy,
Matplotlib, and Scikit-learn.

## 2. Objectives

The main objectives of this project are:

- Load and examine a logistics dataset using Python.
- Identify missing values and duplicate records.
- Clean the dataset using suitable methods.
- Detect unusual values using the IQR method.
- Apply Min-Max scaling to numerical columns.
- Save the processed data for further use.
- Prepare a Word report explaining the work completed.

## 3. Tools and Libraries

The following tools and libraries are used in this project.

| Tool or Library | Purpose |
|-----------------|---------|
| Python 3.14.3 | Programming language |
| VS Code | Writing and running Python programs |
| Pandas | Reading and processing CSV data |
| NumPy | Numerical calculations |
| Matplotlib | Creating charts |
| Scikit-learn | Scaling numerical data |
| python-docx | Creating a Word report |

## 4. Dataset Description

The project uses a sample logistics dataset stored in CSV format.

The dataset contains the following columns:

| Column | Description |
|--------|-------------|
| Order_ID | Identifier for an order |
| Transport_Mode | Method used to transport the shipment |
| Distance_KM | Shipment distance in kilometres |
| Delivery_Days | Time taken to deliver the shipment |
| Shipping_Cost | Cost of transporting the shipment |
| Customer_City | City associated with the customer |

The original dataset contains 65 rows and 6 columns.

During the initial inspection, missing values were found in
Distance_KM, Delivery_Days, and Shipping_Cost. Two duplicate rows
were also identified.

These issues will be examined during the preprocessing stage.

## 5. Project Structure

The project files are organised as follows:

    Logistics_Preprocessing/
    |
    |-- data/
    |   |-- logistics_data.csv
    |
    |-- src/
    |   |-- data_collection.py
    |   |-- data_exploration.py
    |   |-- data_cleaning.py
    |   |-- outlier_detection.py
    |   |-- normalization.py
    |   |-- main.py
    |
    |-- outputs/
    |
    |-- report/
    |   |-- generate_report.py
    |
    |-- requirements.txt
    |-- README.md

The data folder contains the original dataset.

The src folder contains the Python programs used for preprocessing.
The outputs folder will contain the processed datasets and other
results. The report folder contains the script used to generate
the final report.

Some files will be added as the project progresses.

## 6. Setting Up the Project

Python and VS Code are used to develop this project.

First, open PowerShell or the terminal in VS Code and move to the
project directory:

    cd C:\Logistics_Preprocessing

Install the required libraries using:

    py -m pip install -r requirements.txt

To check the installed packages, run:

    py -m pip list

## 7. Data Preprocessing Steps

The project follows these steps.

### 7.1 Data Collection

The program reads the logistics dataset from the data folder.

The data is loaded into a Pandas DataFrame so that its contents
can be examined and processed.

### 7.2 Data Exploration

The dataset is checked to understand its structure and quality.

This includes examining column names, data types, missing values,
duplicate records, and numerical summaries.

### 7.3 Data Cleaning

Duplicate records are identified and removed where appropriate.

Missing numerical values can be filled using the median of the
corresponding column. The method used will depend on the data
and the reason for the missing values.

The original dataset is kept unchanged.

### 7.4 Outlier Detection

The Interquartile Range (IQR) method is used to identify unusually
high or low numerical values.

IQR is calculated as:

    IQR = Q3 - Q1

The lower and upper limits are:

    Lower limit = Q1 - 1.5 * IQR
    Upper limit = Q3 + 1.5 * IQR

Values outside these limits are flagged as potential outliers.

An unusual value is not always an error. Therefore, the results
will be examined before deciding whether any treatment is needed.

### 7.5 Data Normalization

Min-Max scaling is used to bring selected numerical values into
the range from 0 to 1.

The formula is:

    X_scaled = (X - X_min) / (X_max - X_min)

Columns such as Distance_KM, Delivery_Days, and Shipping_Cost
can be considered for scaling.

Order_ID is an identifier, so it should not be normalized.
Categorical columns are also excluded from numerical scaling.

### 7.6 Saving the Results

The processed data will be saved separately in the outputs folder.

Keeping separate files makes it easier to compare the original
dataset with the cleaned and normalized versions.

## 8. Running the Programs

The Python programs will be developed and tested one at a time.

After the required files have been created, individual programs
can be run from the project directory.

For example, the data collection program can be started with:

    py src/data_collection.py

The complete preprocessing pipeline will be run using:

    py src/main.py

The Word report will be generated using:

    py report/generate_report.py

These commands can be used after the corresponding scripts
have been implemented.

## 9. Expected Output

After the preprocessing programs are completed and executed,
the following files are expected to be created:

- cleaned_logistics_data.csv
- normalized_logistics_data.csv
- preprocessing_summary.txt
- outlier_distribution.png
- Logistics_Preprocessing_Report.docx

The CSV files will contain the processed data. The summary
will record important preprocessing results, and the chart
will help show the distribution of numerical values.

The Word report will explain the dataset, methods used,
results obtained, and conclusions.

## 10. Conclusion

This project provides practical experience in preparing logistics
data for analysis using Python.

It covers the identification of missing values and duplicate
records, outlier detection, and normalization.

Working through these steps helps me understand why data quality
is important before performing further analysis.

The completed project will also provide experience in writing
Python programs, working with CSV files, and documenting the
results of a data analysis task.