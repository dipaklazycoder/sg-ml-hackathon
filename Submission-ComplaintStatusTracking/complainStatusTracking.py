# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 00:48:05 2019

@author: DIPAK
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import category_encoders as ce

# Importing the dataset
dataset = pd.read_csv('train.csv')
datasetTest = pd.read_csv('test.csv')
res = pd.read_csv('test.csv')

target_col = ["Complaint-Status"]
cat_cols = ['Transaction-Type','Company-response','Complaint-reason']
data_col =['Consumer-disputes']

from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()
dataset['Consumer-disputes']=imputer.fit_transform(dataset['Consumer-disputes'])
dataset['Transaction-Type']=imputer.fit_transform(dataset['Transaction-Type'])
dataset['Company-response']=imputer.fit_transform(dataset['Company-response'])
dataset['Complaint-reason']=imputer.fit_transform(dataset['Complaint-reason'])
datasetTest['Consumer-disputes']=imputer.fit_transform(datasetTest['Consumer-disputes'])
datasetTest['Transaction-Type']=imputer.fit_transform(datasetTest['Transaction-Type'])
datasetTest['Company-response']=imputer.fit_transform(datasetTest['Company-response'])
datasetTest['Complaint-reason']=imputer.fit_transform(datasetTest['Complaint-reason'])
X = dataset[cat_cols+data_col]
Y= dataset[target_col];
X1 = datasetTest[cat_cols+data_col]

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for var in cat_cols+data_col:
    label_x = LabelEncoder()
    X[var] = label_x.fit_transform(X[var])
    X1[var] = label_x.transform(X1[var])
    

#Target variable is also a categorical so convert it
label_y = LabelEncoder()
Y = label_y.fit_transform(Y)


ce_binary = ce.BinaryEncoder(cols = ['Transaction-Type','Company-response','Complaint-reason','Consumer-disputes'])
#oneHotEncoder = OneHotEncoder(categorical_features=[0,1,2,3])
X = ce_binary.fit_transform(X)

X1 = ce_binary.transform(X1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X1_test = sc.transform(X1);


from sklearn import tree
classifierDT = tree.DecisionTreeClassifier()
classifierDT.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierDT.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracyDT = accuracy_score(y_test, y_pred)
result=pd.DataFrame({'Complaint-ID':res['Complaint-ID'],'Complaint-Status':label_y.inverse_transform(classifierDT.predict(X1))})

result.to_csv('C:/Users/DIPAK/Downloads/model_DToutput.csv',columns=['Complaint-ID','Complaint-Status'],index=False)
