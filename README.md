# Logistics Data Collection, Cleaning and Preprocessing Using Python

## About This Project

This project was developed as part of the **YuvaIntern Week 2 assignment: Data Collection, Cleaning, and Preprocessing for Logistics Analysis**.

The main purpose of this project is to understand how raw data can be collected, checked, cleaned, and prepared before using it for further analysis. In real-world logistics operations, data may contain missing values, duplicate records, inconsistent information, and unusually high or low values. If these issues are not handled properly, they can affect the accuracy of reports and the results of future data analysis.

For this assignment, I created a simulated logistics dataset and used Python to perform different preprocessing operations. The dataset contains information such as order ID, transport mode, delivery distance, delivery time, shipping cost, and customer city.

I divided the work into separate Python scripts so that each stage could be developed, tested, and understood independently. I also created a main program to run the complete preprocessing pipeline in the correct order.

The project generates cleaned and normalized datasets, summary CSV files, and visualizations. These outputs help me understand the condition of the original data and check the results after preprocessing.

The dataset used in this project is simulated for educational purposes. It is not collected from an actual logistics company.

---

## Objectives

The main objectives of this project are:

1. To understand the importance of data collection in logistics analysis.
2. To create a sample logistics dataset using Python.
3. To inspect the dataset and identify its columns, data types, and basic statistics.
4. To detect missing values and duplicate records.
5. To handle missing numerical values using median imputation.
6. To identify potential outliers using the Interquartile Range (IQR) method.
7. To apply Min-Max normalization to selected numerical columns.
8. To compare the data before and after preprocessing.
9. To generate CSV summaries and graphical visualizations.
10. To organize the project into separate Python scripts and run them through a main program.
11. To understand how data preprocessing prepares a dataset for future analysis.

---

## Technologies Used

The following tools and Python libraries were used in this project.

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading, cleaning, and processing tabular data |
| NumPy | Numerical operations |
| Matplotlib | Creating charts and visualizations |
| Seaborn | Creating statistical visualizations |
| Scikit-learn | Supporting data preprocessing and machine-learning workflows |
| Python-docx | Supporting the generation of the project report |
| Visual Studio Code | Writing and running Python programs |
| Git | Tracking changes in project files |
| GitHub | Hosting the project repository |

---

## Dataset Description

The project uses a simulated logistics dataset stored in CSV format.

The original dataset contains **65 rows and 6 columns**. Each row represents a logistics order.

### Dataset Columns

| Column Name | Description |
|---|---|
| `Order_ID` | Unique identifier assigned to an order |
| `Transport_Mode` | Mode of transportation used for the order |
| `Distance_KM` | Distance travelled or recorded for the order, in kilometres |
| `Delivery_Days` | Number of days taken for delivery |
| `Shipping_Cost` | Shipping cost associated with the order, in Indian rupees |
| `Customer_City` | City associated with the customer |

The dataset contains both numerical and categorical information. Numerical columns are useful for statistical analysis and normalization, while categorical columns help summarize information such as transport modes and customer cities.

### Dataset Files

The `data` folder contains three main CSV files:

- `logistics_data.csv` — the original simulated dataset.
- `logistics_data_cleaned.csv` — the dataset after duplicate removal and missing-value treatment.
- `logistics_data_normalized.csv` — the cleaned dataset with additional normalized numerical columns.

The original dataset is kept separate so that the results of preprocessing can be compared with the initial data.

---

## Project Structure

I organized the project into separate Python programs and folders. Each script handles a particular stage of the preprocessing process, while the `data` and `outputs` folders store the datasets and generated results.

```text
Logistics-Preprocessing/
│
├── data/
│   ├── logistics_data.csv
│   ├── logistics_data_cleaned.csv
│   └── logistics_data_normalized.csv
│
├── outputs/
│   ├── missing_values_summary.csv
│   ├── numerical_summary.csv
│   ├── categorical_summary.csv
│   ├── outlier_summary.csv
│   ├── data_cleaning_summary.csv
│   ├── missing_values_before_after.csv
│   ├── outlier_detection_summary.csv
│   ├── detected_outlier_records.csv
│   ├── normalization_summary.csv
│   ├── normalization_comparison.csv
│   ├── distance_distribution.png
│   ├── delivery_days_distribution.png
│   ├── shipping_cost_distribution.png
│   ├── numerical_boxplots.png
│   ├── transport_mode_distribution.png
│   ├── numerical_outlier_boxplots.png
│   ├── original_vs_normalized_boxplots.png
│   └── normalized_feature_distributions.png
│
├── create_dataset.py
├── data_collection.py
├── data_exploration.py
├── data_cleaning.py
├── outlier_detection.py
├── normalization.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The project structure may change if additional analysis scripts or output files are added in the future.

### Purpose of the Main Files

**`create_dataset.py`**

Creates the simulated logistics dataset used for this educational project.

**`data_collection.py`**

Handles the dataset collection or preparation stage of the workflow.

**`data_exploration.py`**

Examines the dataset and generates descriptive statistics, missing-value information, categorical summaries, and visualizations.

**`data_cleaning.py`**

Removes exact duplicate records and handles missing numerical values.

**`outlier_detection.py`**

Uses the IQR method to identify potential outliers in selected numerical columns.

**`normalization.py`**

Applies Min-Max normalization to the selected numerical columns and creates a normalized dataset.

**`main.py`**

Runs the preprocessing scripts in sequence, helping execute the complete workflow from a single command.

**`requirements.txt`**

Lists the Python libraries required to run the project.

**`.gitignore`**

Specifies files and folders that Git should ignore, such as Python cache files, virtual environments, and environment configuration files.

**`data/`**

Stores the original, cleaned, and normalized datasets.

**`outputs/`**

Stores the CSV summaries, comparison tables, and visualizations generated during preprocessing.

---

## Project Workflow

The project follows a sequence of data preparation steps.

### Step 1: Data Collection and Dataset Preparation

The first stage prepares the simulated logistics dataset for analysis. The dataset contains order identifiers, transport modes, delivery distances, delivery durations, shipping costs, and customer cities.

For this assignment, the dataset is generated or prepared locally rather than retrieved from a real logistics database or an external company API.

The original CSV file is stored in the `data` folder and is used as the input for the remaining stages.

### Step 2: Data Exploration

Before cleaning the data, I examined the dataset to understand its structure and identify possible data-quality issues.

The exploration stage includes:

- Checking the number of rows and columns.
- Reviewing column names and data types.
- Calculating descriptive statistics for numerical columns.
- Counting the values in categorical columns.
- Identifying missing values.
- Checking the distribution of numerical features.
- Creating charts to understand the data visually.

The exploration results are saved in CSV files and PNG images inside the `outputs` folder.

### Step 3: Data Cleaning

The next stage focuses on improving the quality of the dataset.

The cleaning process includes:

- Identifying and removing exact duplicate rows.
- Checking missing values in numerical columns.
- Filling missing numerical values using the median.
- Checking the dataset again after cleaning.
- Saving the cleaned dataset separately.

The median was selected for missing numerical values because it is less affected by extreme observations than the mean.

The original CSV file is preserved, and the cleaned data is saved as `logistics_data_cleaned.csv`.

### Step 4: Outlier Detection

After cleaning, the project uses the **Interquartile Range (IQR) method** to identify potential outliers in the following columns:

- `Distance_KM`
- `Delivery_Days`
- `Shipping_Cost`

The IQR is calculated as:

**IQR = Q3 − Q1**

Where:

- **Q1** is the first quartile, or 25th percentile.
- **Q3** is the third quartile, or 75th percentile.

The lower and upper boundaries are calculated using:

**Lower Bound = Q1 − 1.5 × IQR**

**Upper Bound = Q3 + 1.5 × IQR**

Values outside these boundaries are flagged as potential outliers.

In this project, outlier detection is performed separately from data cleaning. The detected records are retained in the dataset rather than being automatically deleted or replaced. This helps preserve potentially important logistics observations for further investigation.

### Step 5: Data Normalization

The final preprocessing stage applies Min-Max normalization to the three selected numerical columns.

The formula is:

**Normalized Value = (Value − Minimum) / (Maximum − Minimum)**

This transformation scales the selected numerical values to a range between 0 and 1.

The following columns are normalized:

- `Distance_KM` → `Distance_KM_Normalized`
- `Delivery_Days` → `Delivery_Days_Normalized`
- `Shipping_Cost` → `Shipping_Cost_Normalized`

The normalized columns are added to a separate dataset. The original numerical columns remain available for comparison and interpretation.

The normalized dataset is saved as `logistics_data_normalized.csv`.

---

## Results and Observations

The following results were obtained from the dataset used in this project.

### 1. Original Dataset

The original dataset contains:

| Measurement | Result |
|---|---:|
| Total rows | 65 |
| Total columns | 6 |
| Exact duplicate rows | 2 |
| Missing values in `Distance_KM` | 2 |
| Missing values in `Delivery_Days` | 1 |
| Missing values in `Shipping_Cost` | 1 |
| Total missing numerical values | 4 |

These results show that the dataset contained duplicate records and missing numerical values that needed to be handled before further analysis.

### 2. Data Cleaning Results

After removing exact duplicate rows and filling missing numerical values using their respective medians, the cleaned dataset contains 63 rows and 6 columns.

| Measurement | Before Cleaning | After Cleaning |
|---|---:|---:|
| Number of rows | 65 | 63 |
| Number of columns | 6 | 6 |
| Exact duplicate rows | 2 | 0 |
| Missing `Distance_KM` values | 2 | 0 |
| Missing `Delivery_Days` values | 1 | 0 |
| Missing `Shipping_Cost` values | 1 | 0 |
| Total missing values in these numerical columns | 4 | 0 |

The median values used to fill missing numerical data were:

| Column | Median Used |
|---|---:|
| `Distance_KM` | 947 |
| `Delivery_Days` | 4 |
| `Shipping_Cost` | 507.065 |

The cleaned dataset was saved separately so that the original data remained available for reference.

### 3. Outlier Detection Results

The IQR method identified three potential outlier observations in the cleaned dataset.

| Column | Q1 | Q3 | IQR | Upper Bound | Detected Value | Order ID |
|---|---:|---:|---:|---:|---:|---|
| `Distance_KM` | 601.5 | 1313 | 711.5 | 2380.25 | 6800 | `ORD-1041` |
| `Delivery_Days` | 2.5 | 6 | 3.5 | 11.25 | 42 | `ORD-1011` |
| `Shipping_Cost` | 352.605 | 664.62 | 312.015 | 1132.6425 | 3450 | `ORD-1026` |

The results indicate unusually high values in the three numerical columns.

However, an observation outside the IQR boundaries is not automatically an error. For example, a long-distance order or a delayed delivery may be unusual but still valid. Therefore, the project flags these observations for review instead of automatically removing them.

### 4. Normalization Results

Min-Max normalization was applied to the cleaned dataset.

| Feature | Minimum | Maximum | Range |
|---|---:|---:|---:|
| `Distance_KM` | 152 | 6800 | 6648 |
| `Delivery_Days` | 1 | 42 | 41 |
| `Shipping_Cost` | 185.72 | 3450 | 3264.28 |

The normalized columns contain values between 0 and 1.

The normalization process preserves the number and order of the cleaned records. The original columns are retained, and the normalized values are stored in additional columns.

### 5. Final Dataset Comparison

| Dataset | Rows | Columns | Description |
|---|---:|---:|---|
| `logistics_data.csv` | 65 | 6 | Original simulated dataset |
| `logistics_data_cleaned.csv` | 63 | 6 | Dataset after duplicate removal and missing-value treatment |
| `logistics_data_normalized.csv` | 63 | 9 | Cleaned dataset with three additional normalized columns |

The final normalized dataset contains three additional columns because normalized values are stored separately from the original numerical values.

---

## Output Files and Visualizations

The project generates several files to document the preprocessing process.

### CSV Summary Files

| File | Purpose |
|---|---|
| `missing_values_summary.csv` | Summarizes missing values identified during exploration |
| `numerical_summary.csv` | Contains descriptive statistics for numerical columns |
| `categorical_summary.csv` | Summarizes categorical data |
| `outlier_summary.csv` | Records outlier-related statistics from the exploration stage |
| `data_cleaning_summary.csv` | Summarizes the data-cleaning results |
| `missing_values_before_after.csv` | Compares missing values before and after cleaning |
| `outlier_detection_summary.csv` | Contains the IQR outlier-detection summary |
| `detected_outlier_records.csv` | Records the observations flagged as potential outliers |
| `normalization_summary.csv` | Summarizes the normalization process |
| `normalization_comparison.csv` | Helps compare original and normalized numerical values |

### Visualization Files

**`distance_distribution.png`**

Shows the distribution of recorded delivery distances.

**`delivery_days_distribution.png`**

Shows the distribution of delivery durations.

**`shipping_cost_distribution.png`**

Shows the distribution of shipping costs.

**`numerical_boxplots.png`**

Provides a visual summary of the numerical features and their distributions.

**`transport_mode_distribution.png`**

Shows the number of records associated with each transport mode.

**`numerical_outlier_boxplots.png`**

Helps visualize potential outliers in the numerical columns.

**`original_vs_normalized_boxplots.png`**

Compares the distributions of original numerical features with their normalized versions.

**`normalized_feature_distributions.png`**

Shows the distributions of the normalized numerical features.

These files help make the preprocessing results easier to inspect and understand.

---

## How to Run the Project

Follow these steps to run the project on a Windows computer.

### Prerequisites

Make sure that the following are installed:

- Python
- Visual Studio Code or another Python-compatible editor
- Required Python libraries listed in `requirements.txt`

The commands below use the `py` launcher, which was used to run this project in the Windows environment.

### Step 1: Open the Project Folder

Open the project folder in Visual Studio Code:

```text
C:\Logistics_Preprocessing
```

Open the VS Code terminal using:

**Terminal → New Terminal**

Make sure the terminal is working in the project root directory.

### Step 2: Install the Required Libraries

Run:

```powershell
py -m pip install -r requirements.txt
```

Wait for the installation to finish.

### Step 3: Run the Complete Pipeline

Run the main program:

```powershell
py main.py
```

The main program runs the preprocessing scripts in sequence.

The pipeline performs the following stages:

1. Data collection or preparation.
2. Data exploration.
3. Data cleaning.
4. Outlier detection.
5. Data normalization.

When the stages complete successfully, the program displays a completion message and the locations of the original, cleaned, and normalized datasets.

### Step 4: Check the Generated Files

After the pipeline finishes, inspect the following folders:

- `data/` for the original, cleaned, and normalized datasets.
- `outputs/` for CSV summaries and visualizations.

You can open the CSV files in Excel, a spreadsheet application, or a text editor. The PNG files can be opened directly to view the charts.

### Running Individual Scripts

The scripts can also be executed individually from the project root directory.

Run data preparation:

```powershell
py data_collection.py
```

Run data exploration:

```powershell
py data_exploration.py
```

Run data cleaning:

```powershell
py data_cleaning.py
```

Run outlier detection:

```powershell
py outlier_detection.py
```

Run normalization:

```powershell
py normalization.py
```

Running the main program is the recommended way to execute the complete pipeline because it coordinates the stages in the intended order.

### Important Note About Dataset Generation

The `create_dataset.py` script is used to create the simulated dataset.

Running this script may overwrite the existing original CSV file. Therefore, do not run it unnecessarily if you want to preserve the current dataset or any manual changes.

The regular command:

```powershell
py main.py
```

is intended to run the preprocessing workflow using the existing dataset.

---

## Challenges and Decisions

During this project, I worked on several data-preprocessing problems.

### Handling Missing Values

Some numerical columns contained missing values. I used median imputation to fill these gaps while keeping the records available for analysis.

I also checked the missing-value counts after cleaning to confirm that the missing values in the selected numerical columns had been handled.

### Removing Duplicate Records

The original dataset contained two exact duplicate rows. Keeping duplicate records could affect statistical summaries, so I removed them during the cleaning stage.

### Handling Potential Outliers

The dataset contained some unusually high values. Instead of deleting these records automatically, I used the IQR method to identify and document them.

This decision keeps potentially meaningful logistics observations available for further investigation.

### Preserving the Original Data

I saved the cleaned and normalized datasets as separate files. This makes it easier to compare the results with the original dataset and reduces the risk of losing the initial data during preprocessing.

### Organizing the Workflow

I separated the operations into individual Python scripts. This makes the project easier to understand, test, maintain, and extend.

The main program brings the individual stages together so that the entire workflow can be run using a single command.

---

## What I Learned

Through this project, I gained practical experience with the early stages of a data-analysis workflow.

I learned how to load and inspect CSV data using Pandas, calculate descriptive statistics, identify missing values, remove duplicate records, and use median imputation to handle missing numerical data.

I also learned how the IQR method can be used to identify potential outliers and why unusual observations should be investigated rather than removed automatically.

Another important part of the project was Min-Max normalization. I learned how scaling numerical values to a common range can help prepare data for certain statistical and machine-learning methods.

I also gained experience organizing a Python project into separate scripts, generating CSV reports and visualizations, installing libraries through `requirements.txt`, and using Git and GitHub to maintain the project.

Most importantly, I understood that preprocessing is not just about modifying a dataset. Each change should have a clear purpose, and the results should be checked to make sure that the data is suitable for the next stage of analysis.

---

## Limitations and Future Improvements

This project demonstrates basic data-preprocessing techniques using a simulated dataset. It is intended for learning and does not represent a production logistics system.

The project could be improved in several ways:

1. Use a real logistics dataset from a reliable source.
2. Add stronger validation for data types, invalid values, and inconsistent categories.
3. Record more detailed information about each cleaning operation.
4. Compare median imputation with other missing-value treatment methods.
5. Investigate detected outliers using business rules before deciding whether to retain or remove them.
6. Add automated tests to verify the results of each preprocessing stage.
7. Introduce logging and clearer error handling for unexpected input files.
8. Extend the project with exploratory analysis or predictive modelling after the data has been prepared.

These improvements would help make the workflow more suitable for larger and more realistic datasets.

---

## Conclusion

This project demonstrates a basic end-to-end data-preprocessing workflow for logistics analysis using Python.

Starting with a simulated dataset containing 65 rows and 6 columns, I explored the data, removed two duplicate records, handled four missing numerical values, identified three potential outlier observations using the IQR method, and applied Min-Max normalization to three numerical features.

The final cleaned dataset contains 63 rows and 6 columns. The normalized dataset contains 63 rows and 9 columns because it includes three additional normalized features.

The project also generates summary CSV files and visualizations to help inspect the results at each stage.

Overall, this assignment helped me understand how data collection, cleaning, outlier detection, and normalization contribute to preparing data for further analysis. It also gave me practical experience in building and organizing a Python project with separate scripts and a main execution program.