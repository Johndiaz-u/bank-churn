#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import string
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('C:/Users/Dominga Genao/OneDrive/Escritorio/Churn Modeling.csv')
data.head()

data.describe()
data.info()
data.columns
data.dtypes
data.value_counts()
data.isnull().sum()
data.isnull().any()
data.shape
data = data.apply(pd.to_numeric, errors='coerce')
data= data.fillna(0)
data.mean()

campos = ['RowNumber','CustomerId','CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']


data[campos].corr()


plt.figure(figsize=(16,13))
sns.heatmap(data[campos].corr(), annot=True)

data.hist(figsize=(18,15))
plt.show()

sns.scatterplot(x='Balance', y= 'CreditScore', data=data)

sns.set_style("whitegrid") 
mean_col = ['RowNumber','Gender','Age','Tenure','Balance','Exited']

sns.pairplot(data[mean_col],palette='Accent')

sns.relplot(x='Age', y= 'CreditScore', data=data)

sns.jointplot(x='Balance', y= 'CreditScore', data=data)

plt.style.use("ggplot")
plt.figure(figsize=(14,8))
plt.xlabel('Balance')
plt.ylabel('CreditScore')
sns.kdeplot(data['Balance'],shade=True,color='blue')
plt.show()

plt.style.use("default")
sns.barplot(x="Balance", y="CreditScore",data=data[179:190])
plt.title("Balance vs CreditScore",fontsize=15)
plt.xlabel("Balance")
plt.ylabel("CreditScore")
plt.show()

plt.style.use("default")
sns.barplot(x="EstimatedSalary", y="CreditScore",data=data[183:190])
plt.title("EstimatedSalary vs CreditScore",fontsize=15)
plt.xlabel("EstimatedSalary")
plt.ylabel("CreditScore")
plt.show()

plt.style.use("default")
sns.barplot(x="Tenure", y="Balance",data=data[170:190])
plt.title("Tenure vs Balance",fontsize=15)
plt.xlabel("Tenure")
plt.ylabel("Balance")
plt.show()

plt.style.use("default")
sns.barplot(x="HasCrCard", y="NumOfProducts",data=data[160:190])
plt.title("HasCrCard vs NumOfProducts",fontsize=15)
plt.xlabel("HasCrCard")
plt.ylabel("NumOfProducts")
plt.show()

list_1=list(data.columns)

list_cate=[]
for i in list_1:
    if data[i].dtype=='object':
        list_cate.append(i)


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


for i in list_cate:
    data[i]=le.fit_transform(data[i])

data

#drop the columns as it is no longer required
X = data.drop('Geography',axis=1)
y = data['Geography']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(len(X_test))
print(len(X_train))
print(len(y_test))
print(len(y_train))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pip install tensorflow

pip install keras

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Crear un modelo secuencial
model = Sequential()

# Capa de convolución 2D con 32 filtros, tamaño del kernel 3x3, función de activación ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Capa de Max Pooling 2D
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa de convolución 2D con 64 filtros, tamaño del kernel 3x3, función de activación ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))

# Capa de Max Pooling 2D
model.add(MaxPooling2D(pool_size=(2, 2)))

# Aplanar la salida de la capa anterior
model.add(Flatten())

# Capa completamente conectada con 128 neuronas y función de activación ReLU
model.add(Dense(128, activation='relu'))

# Capa de salida con 1 neurona (en problemas de clasificación binaria) y función de activación sigmoide
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calcula el Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calcula el Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')


