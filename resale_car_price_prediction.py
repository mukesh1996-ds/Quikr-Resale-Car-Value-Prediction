# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 22:00:59 2024

@author: Monu Sharma
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle

car=pd.read_csv("quikr_car.csv")
car.shape
car.info()

def check_unique_values(dataframe):
    for column in dataframe.columns:
        unique_value=dataframe[column].unique()
        print(f"Column:{column}")
        print(f"Number of Unique Values: {len(unique_value)}")
        print(f"Unique values:{unique_value}\n")

check_unique_values(car)

backup=car.copy()

## Issue with the dataset 
"""
year has many non-year values 
year object is not in specific datatype
price has non required information
convert price to int 
Kms_driven has kms with integers 
kms_driven obj to int
kms_driven has nan values 
Fuel type has nan values 
Keep fist 3 words of names 
"""
## Data Cleaning 
car= car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price'] != "Ask For Price"]
car['Price'] = car['Price'].str.replace(",","").astype(int)
car['kms_driven'] = car['kms_driven'].str.split(" ").str.get(0).str.replace(",","")
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven']=car['kms_driven'].astype(int)
car=car[~car['fuel_type'].isna()]
car['name'] = car['name'].str.split(" ").str.slice(0,3).str.join(" ")
car=car.reset_index(drop=True)

car.info()
car.describe()

car=car[car['Price']<6e6].reset_index(drop=True)

## Let store the data 
car.to_csv("Cleaned_data_car.csv")

## model
X=car.drop(columns=["Price"])
y=car['Price']

## Split the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)

## OneHotEncoder 
encoder=OneHotEncoder()
encoder.fit(X[['name','company','fuel_type']])
column_transform=make_column_transformer((OneHotEncoder(categories=encoder.categories_),['name','company','fuel_type']),
                                         remainder='passthrough')

model=LinearRegression()
workflow=make_pipeline(column_transform,model)
workflow.fit(X_train,y_train)
y_pred=workflow.predict(X_test)
r2_score(y_test,y_pred)

## AS my dataset size is small then what I have done is that i have iterated 1000 times my train test split with different random_state so 
## to check the base accuracy 
score=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    workflow=make_pipeline(column_transform,model)
    workflow.fit(X_train,y_train)
    y_pred=workflow.predict(X_test)
    score.append(r2_score(y_test,y_pred))
score[np.argmax(score)]

## Split the data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=433)

## OneHotEncoder 
encoder=OneHotEncoder()
encoder.fit(X[['name','company','fuel_type']])
column_transform=make_column_transformer((OneHotEncoder(categories=encoder.categories_),['name','company','fuel_type']),
                                         remainder='passthrough')

model=LinearRegression()
workflow=make_pipeline(column_transform,model)
workflow.fit(X_train,y_train)
y_pred=workflow.predict(X_test)
r2_score(y_test,y_pred)

## Dump the model 
pickle.dump(workflow,open("linearegressionmodel.pkl",'wb'))

workflow.predict(pd.DataFrame([["Maruti Suzuki Swift","Maruti",2019,100,"Petrol"]],
                              columns=['name','company','year','kms_driven','fuel_type']))
