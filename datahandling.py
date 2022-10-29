########################################################################################################################
# EX1
########################################################################################################################

import seaborn as sns
from matplotlib import pyplot as plt

########################################################################################################################
# Solution
########################################################################################################################

# set pandas options to make sure you see all info when printing dfs
import pandas as pd
import plotly.express as px
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.impute import KNNImputer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# ----------------------------------------------------------------------------------------------------------------------
# >>> step 1, just try it
# ----------------------------------------------------------------------------------------------------------------------
'''
start by just loading the data
'''
df = pd.read_csv("datahandling_wine.csv", sep=";")
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 2, use skip rows
# ----------------------------------------------------------------------------------------------------------------------
'''
skip row to ignore text inside the file
'''
df = pd.read_csv("datahandling_wine.csv", sep=";", skiprows=1)
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 3, use skipfooter
# ----------------------------------------------------------------------------------------------------------------------
'''
the footer is also a problem, skip it with skipfooter
'''
df = pd.read_csv("datahandling_wine.csv", sep=";", skiprows=1, skipfooter=1, engine='python')
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 4, check na, data types
# ----------------------------------------------------------------------------------------------------------------------
'''
now the df looks fine, but is it really?
only 3 attributes should be categorical but many are, invesigate those
you'll need to set pandas options to see all issues
'''
# print("Are there any NULL values? -->", df.isnull().values.any())
# print("How many NULL values are there? -->", df.isnull().sum().sum())

#print(df.info())
#print(df.describe())
#print(df.index)
#print(df.columns)
# print(df.dtypes)
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 5 try to convert data types to find issues
# ----------------------------------------------------------------------------------------------------------------------
'''
hint: rows 50, 51, 142 are problematic due to mixed/wrong separators or wrong commas
How could you find such issues in an automated way?
'''


def check_int(value):
    try:
        float(value)
        return np.NaN
    except:
        return value


def check_float(value):
    try:
        float(value)
        return np.NaN
    except:
        return value


def convert(dataframe_column, type):
    try:
        dataframe_column = dataframe_column.astype(type)
    except ValueError:
        if type == "int":
            print("\nError trying to convert to int:")
            return dataframe_column.apply(check_int).dropna()
        if type == "float":
            print("\nError trying to convert to float:")
            return dataframe_column.apply(check_float).dropna()

#print(df.head())
#print(convert(df.alcohol, "float"))
#print(convert(df.ash, "float"))
#print(convert(df.target, "int"))
# ----------------------------------------------------------------------------------------------------------------------
# >>> step 6, exclude the three problematic rows
# ----------------------------------------------------------------------------------------------------------------------
'''
the three rows are completely ruined and can only be fixed in isolation
you can read the dataset an skip these rows
'''
df = pd.read_csv("datahandling_wine.csv", sep=";", skiprows=[0, 52, 53, 144], skipfooter=1,
                 engine='python')
# ----------------------------------------------------------------------------------------------------------------------
# step 7, handle rows separately
# ----------------------------------------------------------------------------------------------------------------------
'''
If this is too much data dirt for you continue without handling these three rows (continue with step 8)
Otherwise you can follow the workflow indicated below (steps 7.1, 7.2, 7.3, 7.4)
'''
# ----------------------------------------------------------------------------------------------------------------------
# step 7.1, first get the column names from the df
'''
get column names so you can assign them to the single rows you did read
'''
df_names = list(df.columns)
# ----------------------------------------------------------------------------------------------------------------------
# step 7.2, handle row 52
'''
read only row 52 and repair it in isolation
write it to disk wit correct separators, decimals
'''
df_1 = pd.read_csv("datahandling_wine.csv", sep=";", skiprows=52, skipfooter=130, engine='python')
row = [x for xs in list(df_1.columns) for x in xs.split(',')]
df_1 = pd.DataFrame([row], columns=df_names)
# ----------------------------------------------------------------------------------------------------------------------
# step 7.3, handle row 53
'''
read only row 53 and repair it in isolation
write it to disk wit correct separators, decimals
'''

df_2 = pd.read_csv("datahandling_wine.csv", sep=";", skiprows=53, skipfooter=129, engine='python',
                   names=df_names)
counter = 0
list_tmp = []

for x in df_2:
    tmp = str(df_2.iloc[0][counter]).replace(",", ".")
    list_tmp.append(tmp)
    counter += 1

row = [x for xs in list_tmp for x in xs.split(',')]
df_2 = pd.DataFrame([row], columns=df_names)
# ----------------------------------------------------------------------------------------------------------------------
# step 7.4, handle row 144
'''
read only row 144 and repair it in isolation
write it to disk wit correct separators, decimals
'''
df_3 = pd.read_csv("datahandling_wine", sep=";", skiprows=144, skipfooter=38, engine='python')
row = [x for xs in list(df_3.columns) for x in xs.split(',')]
df_3 = pd.DataFrame([row], columns=df_names)
# ----------------------------------------------------------------------------------------------------------------------
# step 8, re-read and check dtypes again to find errors
# ----------------------------------------------------------------------------------------------------------------------
'''
now re read all data (4 dataframes - the original one without rows51, 52, 144 and the three repaired rows)
combine the three dataframes and recheck for data types (try to convert numeric attributes into float - you'll see the problems then
If you have skipped the three ruined rows just read the df without the three ruined rows and continue to check dtypes
'''
frames = [df, df_1, df_2, df_3]
df_new = pd.concat(frames, ignore_index=True)
# ----------------------------------------------------------------------------------------------------------------------
# step 8, handle categorical data
# ----------------------------------------------------------------------------------------------------------------------
'''
now you can look at unique values of categorical attributes using e.g. value_counts()
this way you'll find problematic values that need recoding (e.g. AUT to AUTUMN)
Here you can also check if there is a column in which two columns are combined and split it
'''
#for x in df_new:
 #print(df[x].value_counts(), "\n")

counter = 0
for x in df_new["season"]:
    if x == "spring":
        df_new["season"][counter] = "SPRING"
    if x == "aut":
        df_new["season"][counter] = "AUTUMN"
    counter += 1
# ----------------------------------------------------------------------------------------------------------------------
# step 9, check split columns
# ----------------------------------------------------------------------------------------------------------------------
'''
data type changes might be needed for split columns
'''
list_tmp = df_new["country-age"].str.split("-", expand=True)

# column country
df_new.rename(columns={"country-age": "country"}, inplace=True)
df_new["country"] = list_tmp[0]

# column age
list_tmp = list_tmp[1].str.split("years", expand=True)
df_new["age"] = list_tmp[0]
# ----------------------------------------------------------------------------------------------------------------------
# step 10, exclude duplicates
# ----------------------------------------------------------------------------------------------------------------------
df_new = df_new.drop_duplicates()
# df_new = df_new.reset_index()
# print(df_new)
# ----------------------------------------------------------------------------------------------------------------------
# step 11, find outliers
# ----------------------------------------------------------------------------------------------------------------------
'''
try to use plots to find outliers "visually"
you can also try to use statistical measures to automatically exclude problematic values but be careful
'''
#fig = px.box(df_new, y="total_phenols")
#fig.show()
# malic_acid: -999 median: 1.715
# total_phenols: 155, 148 median: 2.4

# replace outliers with NaN
df_new.malic_acid = df_new.malic_acid.astype("float")
df_new["malic_acid"].replace(-999, "NaN", inplace=True)

# replace outliers with NaN
df_new["total_phenols"].replace(155, "NaN", inplace=True)
df_new["total_phenols"].replace(148, "NaN", inplace=True)

# replace missing with NaN
df_new["ash"].replace("missing", "NaN", inplace=True)

# replace 0 with NaN
df_new["proline"].replace(0, "NaN", inplace=True)


# split value and add to correct columns
# print(df_new.loc[df_new["magnesium"] == "111 1.7"])
list_tmp = df_new["magnesium"][163].split(" ")
df_new["magnesium"][163] = list_tmp[0]
df_new["total_phenols"][163] = list_tmp[1]

df_new.rename(columns={"od280/od315_of_diluted_wines": "diluted_wines"}, inplace=True)


# change datatypes
convert(df_new.magnesium, "int")
convert(df_new.color, "int")
convert(df_new.target, "int")
convert(df_new.age, "int")
convert(df_new.diluted_wines, "float")
convert(df_new.alcohol, "float")
convert(df_new.alcalinity_of_ash, "float")
convert(df_new.ash, "float")
convert(df_new.malic_acid, "float")
convert(df_new.total_phenols, "float")
convert(df_new.flavanoids, "float")
convert(df_new.nonflavanoid_phenols, "float")
convert(df_new.proanthocyanins, "float")
convert(df_new.color_intensity, "float")
convert(df_new.hue, "float")
df_new.season = df_new.season.astype("string")
df_new.country = df_new.country.astype("string")
# print(df_new)
# print(df.dtypes)
# ----------------------------------------------------------------------------------------------------------------------
# step 12, impute values
# ----------------------------------------------------------------------------------------------------------------------
'''
impute missing values and excluded values using the KNN-Imputer
'''
# print(df_new.isna().sum())

# excluded all non numeric columns
num = [col for col in df_new.columns if df_new[col].dtypes != 'string']

knn = KNNImputer()
df_new[num] = knn.fit_transform(df_new[num])
# print(df_new)

# change datatypes
convert(df_new.magnesium, "int")
convert(df_new.color, "int")
convert(df_new.target, "int")
convert(df_new.age, "int")
convert(df_new.diluted_wines, "float")
convert(df_new.alcohol, "float")
convert(df_new.alcalinity_of_ash, "float")
convert(df_new.ash, "float")
convert(df_new.malic_acid, "float")
convert(df_new.total_phenols, "float")
convert(df_new.flavanoids, "float")
convert(df_new.nonflavanoid_phenols, "float")
convert(df_new.proanthocyanins, "float")
convert(df_new.color_intensity, "float")
convert(df_new.hue, "float")
df_new.season = df_new.season.astype("string")
df_new.country = df_new.country.astype("string")

# ----------------------------------------------------------------------------------------------------------------------
# step 13, some more info on the ds
# ----------------------------------------------------------------------------------------------------------------------
'''
get the class distribution of the target variable
'''
#df_new.groupby("target").hist()
#print(df_new.groupby("target").count())
#plt.show()

# scatterplot
# df_new.plot(kind='scatter', x="magnesium", y="color");
# plt.show()

# group magnesium by color
# hist() #first()

#plt.hist(df_new.groupby("magnesium")["color"].groups)
#plt.xlabel("Groups")
#plt.ylabel("Frequency")
#plt.show()
