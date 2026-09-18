# Telco Customer Churn Analysis

## Overview

This project is a Python practice project based on the Telco Customer Churn dataset.

The main purpose of this project is to practice data cleaning, data inspection, feature creation, missing value handling, and different imputation techniques using Pandas and Scikit-learn.

## Dataset

The dataset contains information about telecommunications customers, including:

* Customer ID
* Gender
* Senior Citizen
* Partner
* Dependents
* Tenure
* Phone Service
* Internet Service
* Contract
* Monthly Charges
* Total Charges
* Churn

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

## Concepts Practiced

* Loading CSV data
* Data inspection using `head()`, `info()`, and `describe()`
* Checking data types using `dtypes`
* Detecting missing values using `isnull()`
* Converting columns to numeric data types
* Creating date features
* Selecting numerical and categorical columns
* Removing rows with missing values using `dropna()`
* Mean imputation
* Median imputation
* Most frequent imputation
* Constant imputation
* Missing indicators
* Random sample imputation
* KNN imputation
* Iterative imputation
* Comparing different imputation methods

## Project Structure

```text
Telco-Customer-Churn-Analysis/
│
├── telco_data_project.py
├── telco_data.csv
└── README.md
```

## How to Run

1. Install Python.
2. Install the required libraries:

```bash
pip install pandas numpy scikit-learn
```

3. Place `telco_data.csv` in the same folder as the Python file.
4. Run the program:

```bash
python telco_data_project.py
```

## Purpose

This project is part of my Machine Learning and Data Science practice. It focuses on understanding data preprocessing and missing value imputation before moving further into machine learning models.
