# Coronary  Heart Disease  Prediction Using Machine Learning  Techniques
'''About the dataset:

The dataset is publically available on the Kaggle website, and it is from an ongoing ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information. It includes over 4,240 records and 15 attributes.

Attributes:

sex: male(1) or female(0);(Nominal)

age: age of the patient;(Continuous - Although the recorded ages have been truncated to whole numbers, the concept of age is continuous)

currentSmoker: whether or not the patient is a current smoker (Nominal)

cigsPerDay: the number of cigarettes that the person smoked on average in one day.(can be considered continuous as one can have any number of cigarretts, even half a cigarette.)

BPMeds: whether or not the patient was on blood pressure medication (Nominal)

prevalentStroke: whether or not the patient had previously had a stroke (Nominal)

prevalentHyp: whether or not the patient was hypertensive (Nominal)

diabetes: whether or not the patient had diabetes (Nominal)

totChol: total cholesterol level (Continuous)

sysBP: systolic blood pressure (Continuous)

diaBP: diastolic blood pressure (Continuous)

BMI: Body Mass Index (Continuous)

heartRate: heart rate (Continuous - In medical research, variables such as heart rate though in fact discrete, yet are considered continuous because of large number of possible values.)

glucose: glucose level (Continuous)

10 year risk of coronary heart disease CHD (binary: “1”, means “Yes”, “0” means “No”) - Target Variable

Objective: Build a classification model that predicts heart disease in a subject.
(note the target column to predict is 'TenYearCHD' where CHD = Coronary heart disease) '''
####################################################################
###################################################################
# Please do the following steps:
# Read the file and display columns.
# Handle missing values, Outliers and Duplicate Data
# Calculate basic statistics of the data (count, mean, std, etc) and
# exploratory analysts and describe your observations.
# Select columns that will be probably important to predict heart disease.
# If you remove columns explain why you removed those.
# Create training and testing sets (use 80% of the data for the training and
# reminder for testing).
# Build a machine learning model to predict TenYearCHD
# Evaluate the model (f1 score, Acuuracy, Precision ,Recall and Confusion Matrix)
# Conclude your findings (Model which is giving best f1 score and why)
print()

#  task

# prepare a document  on heart disease    ----- 5 pages
# what is heart disease
# causes
# statistics of patients in world

# preapre a document on Machinelearning ---------10 pages
# dataset  parameter description


# import  the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
# load  the  dataset
###  load  the  dataset
import pandas as pd
import pickle

df = pd.read_csv("./framingham.csv")
df.head()  #### display  5 rows
# display last 5 rows
df.tail()
# display 5 samples randomly
df.sample(5)
# Display  the column names
###  display  the columns
df.columns
##  first  15  columns are  input  and last column(TenYearCCHD) is output
# Display  total  rows and columns
df.shape
#  Education column as no relation with heart disease
df1 = df.drop(columns=["education"])
df1.head()
df1.columns

#  Draw  the corelation matrix
df1.corr()  ####  will display  the corelation matrix
df1.shape
#  Now  draw  the  heat map
###  Now  draw  the  heat map
plt.figure(figsize=(15, 15))
sns.heatmap(df1.corr(), annot=True, linewidths=0.1)
#  Draw   the  pair plot
sns.pairplot(df1.loc[:, 'totChol':'glucose'])

###  From  the  heat  map  and pair plot  it was found that
# sysBP and diaBP are highly correlated

# And currentSmoker and cigsPerDay are highly correlated

#  so drop  the features  which are high corelated  ##  we should take one
# from each as they
##  are  highly corelated
features_to_drop = ['currentSmoker', 'diaBP']

df2 = df1.drop(columns=features_to_drop)  # df1.drop(features_to_drop,axis=1)
df2.sample(5)
df2.shape
#  Lets  handle  the missing  values
## disply  the missig values  in   each column
missing_value = df2.isnull().sum()
missing_value
##  display the columns that have missing value
missing_value_count = missing_value[missing_value > 0]
missing_value_count
###  find  the missing  value  percentage  in each column
missing_value_percentage = (missing_value_count * 100) / (df2.shape[0])
missing_value_percentage
## Find the maximum  missing value percentage
max(missing_value_percentage)

# we found that the maximum missing value percentage is 9.15 %.
# so we can use  imputation to fill the missing values


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
df3 = pd.DataFrame(imputer.fit_transform(df2))
df3.columns = df2.columns
df3.index = df3.index
df3.head()

##  now cheack for missing values
df3.isnull().sum()
## we  can  see  now  there  is no missing values are present
df3.head()

#  Now  check for  outliers
fig, ax = plt.subplots(figsize=(10, 10), nrows=3, ncols=4)
ax = ax.flatten()

i = 0
for k, v in df3.items():
    sns.boxplot(y=v, ax=ax[i])
    i += 1
    if i == 12:
        break
plt.tight_layout(pad=1.25, h_pad=0.8, w_pad=0.8)
# Outliers found in features named
# ['totChol', 'sysBP', 'BMI','heartRate', 'glucose']
#  so we have to handle  the  outliers
# Outliers handling


a = len(df3[df3['BMI'] > 43])
b = len(df3[df3['heartRate'] > 125])
c = len(df3[df3['glucose'] > 200])
d = len(df3[df3['totChol'] > 450])
e = len(df3[df3['sysBP'] > 220])
x = a + b + c + d + e
print('Number of training examples to be deleted for outliers removal is ', x)

a, b, c, d, e

# deleting outliers

df3 = df3[~(df3['sysBP'] > 220)]
df3 = df3[~(df3['BMI'] > 43)]
df3 = df3[~(df3['heartRate'] > 125)]
df3 = df3[~(df3['glucose'] > 200)]
df3 = df3[~(df3['totChol'] > 450)]
print(df3.shape)
#  Lets find  if any duplicated  row  exists
duplicate = df3[df3.duplicated()]
print(duplicate)
##  it shows   no duplicate  data frame  exists
#  lets extract some useful insights
# 1 Display how many people suffer from heart disease and not
df3['TenYearCHD'].value_counts()
#  it means 615  people had heart disease problem
# 2 show the above information in a bar chart
import seaborn as sns

sns.countplot(df3['TenYearCHD'])
# 3 Show the percentage of pepple with and with out heart disease
plt.pie(df3['TenYearCHD'].value_counts(), labels=['No', 'Yes'], autopct="%0.2f%%")
plt.show()
# 4   Display no of male and female
df3['male'].value_counts()
# 0---female
# 1 --male
# 5 show percentage of distribution of male and female
plt.pie(df3['male'].value_counts(), labels=['Female', 'Male'], autopct="%0.2f%%")
plt.show()
#  percentage of  Female is more than male  in our dataset
# 6 #  Display how many male have heart disease

df3.loc[(df3['male'] == 1) & (df3['TenYearCHD'] == 1)]
len(df3.loc[(df3['male'] == 1) & (df3['TenYearCHD'] == 1)])
#  333  male have heart problem
# total male =1801
333 / 1801
# 18.4 percentage of male  have heart problem
# 7 Display how many female have heart disease

df3.loc[(df3['male'] == 0) & (df3['TenYearCHD'] == 1)]
len(df3.loc[(df3['male'] == 0) & (df3['TenYearCHD'] == 1)])
#  no of female with heart problem =282
# total no of female=2381

282 / 2381
#  it means  11.8  %  of female  have heart problem
## male and female having disease or not
# sns.countplot(df3['male'], hue=df3['TenYearCHD'])

# first bar is  for  female
# second bar is for  male
# 8 #  how many peploe  dont smoke  but suffered  from heart disease


len(df3.loc[(df3['cigsPerDay'] == 0) & (df3['TenYearCHD'] == 1)])
# 9 Display Ho many people with diabetes and heart disease

len(df3.loc[(df3['diabetes'] == 1) & (df3['TenYearCHD'] == 1)])
# 10 in which age range heart disease is more
# age between  20 to 40
len(df3.loc[(df3['TenYearCHD'] == 1) & (df3['age'].between(20, 40))])
# age between 40 to 60
len(df3.loc[(df3['TenYearCHD'] == 1) & (df3['age'].between(40, 60))])
# age between 60 to 80
len(df3.loc[(df3['TenYearCHD'] == 1) & (df3['age'].between(60, 80))])
#  from abobe we conclude  that  most people from age between 40 to 60
#  suffer from heart disease

# Standardise some features

# Standardise some features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cols_to_standardise = ['age', 'totChol', 'sysBP', 'BMI', 'heartRate', 'glucose', 'cigsPerDay']
df3[cols_to_standardise] = scaler.fit_transform(df3[cols_to_standardise])
df3.head()

#  Devide the data into  2 parts  one for input  and another for output
X = df3.drop(columns=["TenYearCHD"])
Y = df3.TenYearCHD
#  Now  split the data  into training and testing
#   80%  for training and 20% for testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.2, random_state=40)
print(X_train.shape)  # (3345, 12)
print(X_test.shape)  # (837, 12)
print(Y_train.shape)  # (3345,)
print(Y_test.shape)  # (837,)
# Apply  Logistic Regression
from sklearn.linear_model import LogisticRegression

L = LogisticRegression(solver='liblinear')
# train the model
L.fit(X_train, Y_train)
# test the model
pickle.dump(L,open("model.pkl","wb"))



