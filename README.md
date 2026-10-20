# Logistics Data Collection, Cleaning and Preprocessing Using Python

## About This Project

This project is part of my Week 2 internship assignment at YuvaIntern. The main purpose of this assignment is to understand how raw data is collected, inspected, cleaned and prepared before it is used for further analysis.

I chose a logistics dataset because shipment information contains several types of data, such as delivery distance, delivery time, shipping cost, transportation mode and customer location. If this information contains missing values, duplicate records or unusual values, the results of the analysis may not be reliable.

For this project, I created a sample logistics dataset and used Python to perform the preprocessing steps. I started by checking the structure and quality of the data. After that, I explored the numerical and categorical columns, handled missing values, removed duplicate records, identified potential outliers and applied Min-Max normalization to selected numerical features.

I also saved the processed datasets, summary tables and charts so that the results can be checked separately without running every calculation again.

The dataset used in this project is simulated for educational purposes. It does not represent actual shipment records from a logistics company.

## Objectives

The objectives of this assignment are:

- To create and work with a sample logistics dataset.
- To understand the structure of a CSV dataset using Python.
- To identify missing values and duplicate records.
- To explore numerical columns using descriptive statistics.
- To examine categorical columns such as transportation mode and customer city.
- To handle missing numerical values using median imputation.
- To detect potential outliers using the Interquartile Range (IQR) method.
- To apply Min-Max normalization to selected numerical features.
- To save the cleaned and normalized datasets separately.
- To generate summary files and visualizations for understanding the results.
- To organize the project so that the individual preprocessing steps can be reviewed and executed.

## Technologies and Libraries Used

I used Python for the main data preprocessing tasks. The following tools and libraries were used in the project.

| Technology or Library | Purpose |
|---|---|
| Python | Writing and running the preprocessing programs |
| Pandas | Reading CSV files, examining data, cleaning records and saving results |
| NumPy | Supporting numerical operations |
| Matplotlib | Creating charts and visualizations |
| Scikit-learn | Providing data preprocessing functionality where required |
| Visual Studio Code | Writing, editing and running Python files |
| Git | Tracking changes to project files |
| GitHub | Storing and sharing the project repository |

## Dataset Description

I created a sample dataset containing logistics and shipment-related information. The original dataset contains 65 rows and 6 columns.

Each row represents a sample shipment record. The columns describe the order, the transportation method, the distance travelled, the delivery time, the shipping cost and the customer's city.

| Column Name | Description | Data Type |
|---|---|---|
| Order_ID | Unique identification number assigned to an order | Text |
| Transport_Mode | Transportation method used for the shipment | Text |
| Distance_KM | Distance covered during transportation, measured in kilometres | Numerical |
| Delivery_Days | Time taken to deliver the order, measured in days | Numerical |
| Shipping_Cost | Cost associated with shipping the order, recorded in INR | Numerical |
| Customer_City | City associated with the customer | Text |

The dataset contains three main numerical features: `Distance_KM`, `Delivery_Days` and `Shipping_Cost`. These columns are useful for studying the distribution of shipment distances, delivery times and transportation costs.

The categorical features, `Transport_Mode` and `Customer_City`, help describe the types of transportation and the locations represented in the sample data.

## Project Structure

I organized the project into separate Python programs and folders. Each Python file handles a particular part of the preprocessing process, while the `data` and `outputs` folders store the input datasets, processed datasets and generated results.

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
├── requirements.txt
└── README.md
```

The output folder may contain additional files if more reports or visualizations are generated during later development.

## Work Performed

### 1. Data Collection

I started by creating a sample logistics dataset using Python. The dataset contains order identifiers, transportation modes, delivery distances, delivery durations, shipping costs and customer cities.

I stored the original dataset as `data/logistics_data.csv`. Keeping the raw data in a separate file makes it possible to compare the original records with the processed results.

I also used a separate data collection script to load and inspect the dataset. This helped me confirm that the file could be read correctly and that the expected columns were available for the next steps.

Because the data is simulated, the results are intended to demonstrate preprocessing techniques rather than describe the actual performance of a logistics business.

### 2. Initial Data Exploration

Before modifying the dataset, I examined its structure and checked the quality of the available information.

The initial dataset contained 65 rows and 6 columns. During this stage, I checked the number of records, column names, data types, missing values, duplicate rows and basic statistical information.

I also examined the categorical columns to understand how the transportation modes and customer cities were distributed.

The initial inspection identified the following issues:

- Two duplicate rows were present.
- Four numerical values were missing.
- The missing values were found in `Distance_KM`, `Delivery_Days` and `Shipping_Cost`.
- Some numerical values were much larger than the other observations and required further investigation.

The missing-value distribution was:

| Column | Missing Values |
|---|---:|
| Distance_KM | 2 |
| Delivery_Days | 1 |
| Shipping_Cost | 1 |
| **Total** | **4** |

I generated summary CSV files and charts during this stage. The charts helped me examine the distributions of distance, delivery time and shipping cost, while the summary tables provided numerical information about the dataset.

### 3. Data Cleaning

After exploring the original data, I worked on removing duplicate records and handling missing numerical values.

#### Removing Duplicate Records

The original dataset contained two exact duplicate rows. I removed these duplicate records so that identical rows would not unnecessarily affect the subsequent analysis.

After removing the two duplicate rows, the dataset contained 63 rows.

I kept the original dataset unchanged and saved the cleaned version separately as:

`data/logistics_data_cleaned.csv`

#### Handling Missing Values

The missing values were found in three numerical columns. Instead of deleting every row containing a missing value, I used median imputation.

The median is the middle value of a sorted set of observations. It can be useful when dealing with numerical data that contains unusually large or small values because it is generally less affected by extreme observations than the mean.

The following median values were used:

| Column | Missing Values | Median Used |
|---|---:|---:|
| Distance_KM | 2 | 947 |
| Delivery_Days | 1 | 4 |
| Shipping_Cost | 1 | 507.065 |

The missing values were replaced using the median of the corresponding numerical column.

After cleaning, the dataset contained 63 rows and 6 columns, with no missing values in the columns checked by the cleaning program.

I also verified the cleaned data and saved summary files showing the cleaning results and missing values before and after processing.

One important decision was to retain unusual numerical observations during cleaning. A large value is not automatically an error, so I investigated potential outliers separately instead of deleting them without further evidence.

### 4. Outlier Detection Using the IQR Method

After cleaning the dataset, I used the Interquartile Range method to identify potentially unusual numerical observations.

The IQR method is based on the first quartile (Q1), third quartile (Q3) and interquartile range.

The calculation is:

`IQR = Q3 - Q1`

The lower and upper boundaries are calculated as:

`Lower Bound = Q1 - 1.5 × IQR`

`Upper Bound = Q3 + 1.5 × IQR`

Values below the lower boundary or above the upper boundary are flagged as potential outliers.

I applied this method to the following columns:

- `Distance_KM`
- `Delivery_Days`
- `Shipping_Cost`

The analysis identified three potential outliers in the cleaned dataset.

| Column | Potential Outlier | Order ID |
|---|---:|---|
| Distance_KM | 6800 | ORD-1041 |
| Delivery_Days | 42 | ORD-1011 |
| Shipping_Cost | 3450 | ORD-1026 |

Each of these observations was outside the corresponding IQR boundary.

I did not automatically remove these observations. A shipment might genuinely cover a long distance, take longer to arrive or cost more than other shipments. Removing such a record without checking its validity could remove useful information.

Instead, I saved the detected records and the outlier summary in the `outputs` folder. I also generated boxplots to make the unusual values easier to inspect.

This step helped me understand that outlier detection and outlier removal are two different operations. Detecting an unusual value is only the first step; deciding whether it should be corrected or removed requires additional information.

### 5. Min-Max Normalization

The numerical features in a dataset can have different ranges. For example, delivery time is measured in days, distance is measured in kilometres, and shipping cost is recorded in INR.

To demonstrate feature scaling, I applied Min-Max normalization to the three numerical columns.

The formula used is:

`Normalized Value = (Value - Minimum) / (Maximum - Minimum)`

This transformation scales values to a range between 0 and 1, provided that the minimum and maximum are calculated from the same dataset being transformed.

I applied normalization to:

- `Distance_KM`
- `Delivery_Days`
- `Shipping_Cost`

The normalized values were stored in three additional columns:

- `Distance_KM_Normalized`
- `Delivery_Days_Normalized`
- `Shipping_Cost_Normalized`

I retained the original numerical columns so that the original measurements could still be compared with the scaled values.

The normalized dataset was saved as:

`data/logistics_data_normalized.csv`

The original cleaned dataset was not overwritten. This allowed me to preserve the cleaned measurements and keep the normalized values in a separate output file.

I also checked that the normalized columns had minimum values of 0.0 and maximum values of 1.0, and generated comparison tables and charts.

One limitation of Min-Max normalization is that extreme values can affect the scaling range. Since the potential outliers were retained, they also influenced the minimum and maximum values used in normalization.

### 6. Saving Results and Visualizations

Throughout the project, I saved the results of each processing stage rather than keeping all the work inside a single Python script.

The `data` folder contains the original, cleaned and normalized datasets. The `outputs` folder contains summary tables and charts created during data exploration, cleaning, outlier detection and normalization.

The CSV summary files make it easier to inspect numerical results, while the charts provide a visual way to understand the data.

Keeping these outputs separate also makes the project easier to review and helps me check whether a particular preprocessing step produced the expected result.

## Final Results

The following table summarizes the main results obtained from the preprocessing pipeline.

| Item | Result |
|---|---|
| Original dataset size | 65 rows and 6 columns |
| Duplicate rows identified | 2 |
| Rows remaining after duplicate removal | 63 |
| Missing numerical values identified | 4 |
| Missing numerical values handled | 4 |
| Missing values in the checked cleaned columns | 0 |
| Potential distance outliers identified | 1 |
| Potential delivery-time outliers identified | 1 |
| Potential shipping-cost outliers identified | 1 |
| Total potential outliers identified | 3 |
| Outliers automatically removed | 0 |
| Numerical features normalized | 3 |
| Normalized value range | 0 to 1 |

These results show that the original dataset was inspected, cleaned and prepared for further analysis. Duplicate records were removed, missing numerical values were handled, potential outliers were documented, and Min-Max normalization was applied to the selected numerical columns.

The outlier observations were retained intentionally because the project does not contain enough external information to confirm that they are errors.

## How to Set Up and Run the Project

The project was developed using Python and Visual Studio Code on Windows.

### Step 1: Install Python

Install a compatible Python version if it is not already available on your computer.

To check the Python installation on Windows, open PowerShell or the VS Code terminal and run:

```powershell
py --version
```

### Step 2: Open the Project Folder

Open the `Logistics-Preprocessing` folder in Visual Studio Code.

Make sure the terminal is opened in the project root directory, where the Python files and `requirements.txt` are located.

### Step 3: Install the Required Libraries

Run the following command in the terminal:

```powershell
py -m pip install -r requirements.txt
```

This command installs the libraries listed in `requirements.txt`.

### Step 4: Run the Python Programs

The following commands run the individual scripts:

```powershell
py create_dataset.py
py data_collection.py
py data_exploration.py
py data_cleaning.py
py outlier_detection.py
py normalization.py
```

Run the programs in the appropriate order because some stages depend on files created by earlier stages.

**Important:** `create_dataset.py` creates the sample dataset and may overwrite the existing raw CSV file. If you want to preserve changes made to the input data, keep a backup before running that script again.

The exploration, cleaning, outlier detection and normalization programs generate the corresponding results in the `outputs` and `data` folders.

## Challenges and Decisions

While working on this assignment, I learned that preprocessing is not simply about removing incorrect-looking data. It is also necessary to understand the type of problem, choose a suitable method and verify the result.

One challenge was handling missing numerical values without unnecessarily losing complete information from other columns. I used median imputation for the missing numerical entries and checked the dataset after processing.

Another important decision concerned the potential outliers. The distance, delivery-time and shipping-cost columns contained values that were much larger than the typical observations. Although these values were flagged by the IQR method, I kept them in the cleaned dataset because the available information did not prove that they were incorrect.

I also kept the original dataset separate from the cleaned and normalized versions. This made it easier to compare the different stages and reduced the risk of losing the original data.

These decisions helped me understand why each preprocessing operation should be checked rather than applied without considering its effect on the dataset.

## What I Learned

Through this assignment, I gained practical experience in preparing a dataset for analysis using Python.

I practised reading CSV files, examining rows and columns, checking data types, calculating descriptive statistics and identifying missing values and duplicate records using Pandas.

I also learned how median imputation can be used to handle missing numerical values, how the IQR method identifies potential outliers, and how Min-Max normalization changes the scale of numerical features.

Another useful part of the assignment was saving intermediate results and visualizations. This helped me understand the importance of documenting the work and making the output easier to verify.

I also gained experience organizing Python scripts and project files in Visual Studio Code and maintaining the project using Git and GitHub.

Overall, this assignment helped me connect the concepts of data collection, data cleaning and preprocessing with a practical example. It gave me a better understanding of why data quality matters before performing more advanced analysis or building a machine learning model.

## Limitations and Future Improvements

This project uses a simulated dataset with a limited number of records. Therefore, the results should be treated as a demonstration of data preprocessing techniques rather than conclusions about a real logistics company.

The IQR method flags statistical outliers but cannot determine by itself whether a record is incorrect. Additional shipment information or domain knowledge would be needed to make that decision.

Similarly, median imputation is a practical way to fill the missing numerical values in this exercise, but another dataset might require a different strategy.

In the future, the project could be improved by using a larger real-world dataset, adding more validation rules, comparing different missing-value handling techniques and recording more detailed information about each processing step.

A combined execution script and an automatically generated project report could also make the workflow easier to run and document.

## Conclusion

This project demonstrates a basic logistics data preprocessing workflow using Python. Starting with a simulated dataset of 65 records, I inspected the data, identified two duplicate rows and four missing numerical values, removed the duplicates, and handled the missing values using median imputation.

I then used the IQR method to identify three potential outliers and applied Min-Max normalization to distance, delivery time and shipping cost. The cleaned and normalized datasets were saved separately, along with summary tables and visualizations.

The main outcome of this assignment is a structured preprocessing workflow that shows how raw data can be checked and prepared before further analysis. The work also helped me practise Python programming, data handling, visualization and project organization.