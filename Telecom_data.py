import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Loading data
df = pd.read_csv("telco_data.csv")

# Basic data inspection
print("\nFIRST 5 ROWS")
print(df.head())

print("\nINFO")
print(df.info())

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

# Converting TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nTOTALCHARGES AFTER CONVERSION")
print("Data Type:", df["TotalCharges"].dtype)
print("Missing Values:", df["TotalCharges"].isnull().sum())

# Defining column types
numerical_col = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_col = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn"
]

id_col = ["customerID"]

print("\nNUMERICAL COLUMNS")
print(numerical_col)

print("\nCATEGORICAL COLUMNS")
print(categorical_col)

print("\nID COLUMN")
print(id_col)

# Creating signup date features
df["SignupDate"] = pd.date_range(
    start="2020-01-01",
    periods=len(df),
    freq="D"
)

df["SignupYear"] = df["SignupDate"].dt.year
df["SignupMonth"] = df["SignupDate"].dt.month
df["SignupDay"] = df["SignupDate"].dt.day

print("\nDATE FEATURES")
print(df[["SignupDate", "SignupYear", "SignupMonth", "SignupDay"]].head())

# Checking missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Removing rows containing missing values
df_complete = df.dropna()

print("\nCOMPLETE CASE ANALYSIS")
print("Shape after removing missing rows:", df_complete.shape)

# Checking numerical missing values
print("\nNUMERICAL MISSING VALUES")
print(df[numerical_col].isnull().sum())

# Mean imputation
mean_imputer = SimpleImputer(strategy="mean")
df_mean = df.copy()

df_mean[numerical_col] = mean_imputer.fit_transform(
    df_mean[numerical_col]
)

print("\nAFTER MEAN IMPUTATION")
print(df_mean[numerical_col].isnull().sum())

# Median imputation
median_imputer = SimpleImputer(strategy="median")
df_median = df.copy()

df_median[numerical_col] = median_imputer.fit_transform(
    df_median[numerical_col]
)

print("\nAFTER MEDIAN IMPUTATION")
print(df_median[numerical_col].isnull().sum())

# Most frequent imputation
categorical_imputer = SimpleImputer(strategy="most_frequent")
df_cat_mode = df.copy()

df_cat_mode[categorical_col] = categorical_imputer.fit_transform(
    df_cat_mode[categorical_col]
)

print("\nAFTER MOST FREQUENT IMPUTATION")
print(df_cat_mode[categorical_col].isnull().sum())

# Constant imputation
constant_imputer = SimpleImputer(
    strategy="constant",
    fill_value="Missing"
)

df_cat_constant = df.copy()

df_cat_constant[categorical_col] = constant_imputer.fit_transform(
    df_cat_constant[categorical_col]
)

print("\nAFTER CONSTANT IMPUTATION")
print(df_cat_constant[categorical_col].isnull().sum())

# Missing indicator
indicator_imputer = SimpleImputer(
    strategy="mean",
    add_indicator=True
)

indicator_result = indicator_imputer.fit_transform(
    df[numerical_col]
)

print("\nMISSING INDICATOR")
print("Shape:", indicator_result.shape)

# Random sample imputation
df_random = df.copy()

for col in numerical_col:
    missing = df_random[col].isnull()

    random_values = df_random.loc[~missing, col].sample(
        n=missing.sum(),
        replace=True,
        random_state=42
    )

    df_random.loc[missing, col] = random_values.values

print("\nAFTER RANDOM SAMPLE IMPUTATION")
print(df_random[numerical_col].isnull().sum())

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df.copy()

df_knn[numerical_col] = knn_imputer.fit_transform(
    df_knn[numerical_col]
)

print("\nAFTER KNN IMPUTATION")
print(df_knn[numerical_col].isnull().sum())

# Iterative imputation
iterative_imputer = IterativeImputer(
    max_iter=10,
    random_state=42
)

df_iterative = df.copy()

df_iterative[numerical_col] = iterative_imputer.fit_transform(
    df_iterative[numerical_col]
)

print("\nAFTER ITERATIVE IMPUTATION")
print(df_iterative[numerical_col].isnull().sum())

# Compare imputation methods
print("\nTENURE COMPARISON")

comparison = pd.DataFrame({
    "Original": df["tenure"].head(10),
    "Mean": df_mean["tenure"].head(10),
    "Median": df_median["tenure"].head(10),
    "Random": df_random["tenure"].head(10),
    "KNN": df_knn["tenure"].head(10),
    "Iterative": df_iterative["tenure"].head(10)
})

print(comparison)