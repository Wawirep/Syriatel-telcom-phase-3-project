# Syriatel-telcom-phase-3-project
THE SYRIATEL CUSTOMER CHURN 
project by Phoebe Wawire 
Class of DSF-PT7

A.	THE PROBLEM STATEMENT

I want to predict whether a customer with the company named SyriaTel churn will stop doing business with them based on various factors.
My Audience is the telecom business who are interested to know how much money they will lose when customers do not stick around for long. 
My target variable will be categorical with two classes, labeled as 1 for churn and 0 for no churn.

B.	DATA COLLECTION

	I imported all the necessary libraries as below that enabled me analyse the data
import pandas as pd
import numpy as np  
import seaborn as sns
from hashlib import sha256
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

	Loaded the data into the Visual studio code to generate the datasets i was to work with
	viewed dimensions of dataset to know it's shape; It showed there were 3333 instances and 21 variables.
	I previewed the dataset and saw the firts few columns and rows of the dataset;
	Summary of type od data for the 21 columns of the dataset were;
	1 Boolean column
	8 Floating Point columns
	8 Integer columns
	Object (String) columns

	I identified the column named Churn to be the target variable

	I Checked the column names for the whole dataset; ['state', 'account length', 'area code', 'phone number',
                'international plan', 'voice mail plan', 'number vmail messages',
                'total day minutes', 'total day calls', 'total day charge',
                'total eve minutes', 'total eve calls', 'total eve charge',
                'total night minutes', 'total night calls', 'total night charge',
                'total intl minutes', 'total intl calls', 'total intl charge',
                'customer service calls', 'churn'],

	Viewed the summary of the datasets; Types of variables

C.	DATA PREPARATION

	I sorted the dataset into data types. There are a mixture of categorical, numerical, intergers and boolean variables in the dataset. 
	Categorical variables have data type object, 
	Numerical variables have data type float64,
	Boolean variables have data type bool and
	Intergers have data type int64
	There are 4 categorical variables are : ['state', 'phone number', 'international plan', 'voice mail plan']
	I checked the missing values and there were no missing values in categorical variables
	I also checked the frequency and the frequency distribution for the categorical variables 
	I checked for the cardinality. The number of labels within a categorical variable is known as cardinality. A high number of labels within a variable is known as high cardinality. High cardinality may pose some serious problems in the machine learning model.
	state contains  51  labels
	phone number contains  3333  labels
	international plan contains  2  labels
	voice mail plan contains  2  labels
	I converted all the 4 categorical variables; state, phone number, international plan and the voice mail plan to numeric format so that they fit in the models
	I checked for missing variables and they were none in the dataset.
	I check for outliers and were identified in the number vmail messages, account length, today calls, total day minutes and the total day charge. I managed to remove them by calculating their mean and imputing them with the mean.
	I rechecked the data for missing variables
	Declared feature vector and target variable 'churn' then dropped it the 
	Split data into separate training and test set(X and Y)
	Feature engineering and scaling to fine tune the data.

D.	DATA MODELLING
1. I modelled using Logistic Regression

CONCLUSION FOR LOGISTIC REGRESSION MODEL
Key Metrics Overview:
True Negatives (TN): 566 (The number of non-churn customers correctly identified as non-churn)
False Positives (FP): 0 (The number of non-churn customers incorrectly identified as churn)
False Negatives (FN): 91 (The number of churn customers incorrectly identified as non-churn)
True Positives (TP): 10 (The number of churn customers correctly identified as churn)

Explanation
Precision: The proportion of true positive predictions out of all positive predictions made by the model.
For churn (class 1): Precision = 0.77
For non-churn (class 0): Precision = 1.00
Recall: The proportion of actual positives that were correctly predicted by the model.
For churn (class 1): Recall = 0.10
For non-churn (class 0): Recall = 0.99
F1-Score: The harmonic mean of precision and recall, which provides a balanced measure of both metrics.
For churn (class 1): F1-Score = 0.18
For non-churn (class 0): F1-Score = 0.92
Model Performance:
The logistic regression model shows an overall accuracy of 86%, indicating that it correctly classifies 86% of the samples. However, the performance on the churn class (class 1) is notably weaker, with a recall of only 0.10. This means the model identifies only 10% of actual churn cases.
The model performs very well on the non-churn class (class 0), with a recall of 0.99 and an F1-score of 0.92, indicating high reliability in predicting non-churn customers.
Precision: For non-churn (class 0), the precision is perfect, suggesting that when the model predicts non-churn, it is accurate. For churn (class 1), the precision is relatively high but not perfect.
Recall: The low recall for churn (class 1) indicates that many actual churn cases are missed. This could be critical if identifying churners is important for business decisions.
Recommendations:
Data Collection: Obtain more data for the churn class to balance the dataset and improve model performance.


2. I also modelled using the decision tree model

CONCLUSION FOR THE DECISION TREE MODEL
Key Metrics Overview:
True Negatives (TN): 566 (The number of non-churn customers correctly identified as non-churn)
False Positives (FP): 51 (The number of non-churn customers incorrectly identified as churn)
False Negatives (FN): 42 (The number of churn customers incorrectly identified as non-churn)
True Positives (TP): 59 (The number of churn customers correctly identified as churn)

Explanation
Precision: For Churn (class 1): Precision = 0.55. This indicates that when the model predicts a customer will churn, it is correct 55% of the time.
For Non-Churn (class 0): Precision = 0.93. This indicates that when the model predicts a customer will not churn, it is correct 93% of the time.
Recall: For Churn (class 1): Recall = 0.59. This means the model correctly identifies 59% of actual churn cases.
For Non-Churn (class 0): Recall = 0.91. This means the model correctly identifies 91% of actual non-churn cases.
F1-Score: For Churn (class 1): F1-Score = 0.57. This reflects a balance between precision and recall for churn cases, showing moderate performance.
For Non-Churn (class 0): F1-Score = 0.92. This indicates a strong balance between precision and recall for non-churn cases.
Model Performance:
Non-Churn (class 0): The model performs very well with a high precision (0.93) and a good recall (0.91), resulting in a strong F1-Score (0.92). This indicates reliable performance in predicting non-churn customers.
Churn (class 1): The model shows lower precision (0.55) and recall (0.59) for churn cases, with a moderate F1-Score (0.57). This suggests that the model struggles with identifying churn customers.

Data Collection: To improve the model's ability to detect churn cases, consider collecting more data, especially for the churn class, to better balance the dataset.

Model Enhancement: Explore additional modeling techniques to improve the model's performance

E.	CONCLUSION FOR THE 2 MODELS USED
The decision tree classifier outperformed the logistic regression model in this scenario. The decision tree achieved perfect precision, recall, and F1-scores for both classes, whereas the logistic regression model had a slight drop in recall for the churn class.

Decision tree performed better due to its Complexity and Flexibility. It partitioned the feature space into more distinct regions compared to logistic regression, which assumed a linear relationship between features and the target.

While the decision tree performed perfectly here, it’s worth noting that decision trees can easily over fit, Overfitting might have caused it to perform exceptionally well 

Model Validation:
To ensure that the decision tree’s performance is robust and not just due to overfitting, it would be important to test it with a larger and more diverse dataset or use techniques like cross-validation.

