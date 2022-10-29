# Data Handling
Implementation and handling of missing data, class imbalance and outliers in datasets

**Dataset:**
- corrupted [Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- the dataset includes all kinds of errors
    - missing values with different encodings (-999, 0, np.nan, ...)
    - typos for categorical/object column
    - columns with wrong data types
    - wrong/mixed separators and decimals in one row
    - "slipped values" where one separator has been forgotten and values from adjacent columns land in one column
    - combined columns as one column
    - unnecessary text at the start/end of the file

## **Steps:**
(1) load dataset

(2) repair the dataset:
- consistent NA encodings. 
- correct data types for all columns
- correct categories (unique values) for object type columns
- read all rows, including those with wrong/mixed decimal, separating characters

(3) find duplicates, exclude them and remove only the unnecessary rows

(4) find outliers and exclude them :
- function to plot histograms/densities etc.
- can explore a dataset quickly to find errors


(5) impute missing values using the KNNImputer
- including the excluded outliers (values gets imputed)
- only the original wine features are used as predictors (no age, season, color, ...)
- original wine features can be find by using load_wine()

(6) find the class distribution
- groupby() method
