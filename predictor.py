# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv('heart.csv')

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values

print(df.isnull().sum()) #dataset seems to have all values for every column every row

