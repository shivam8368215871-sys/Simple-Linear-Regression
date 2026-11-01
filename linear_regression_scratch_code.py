import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class model:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
    
           num = 0
           den = 0
           
           for i in range(X_train.shape[0]):
               
               num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
               den = den + ((X_train[i] - X_train.mean()) * ((X_train[i] - X_train.mean())))
               
           self.m = num / den
           self.b = y_train.mean() - (self.m * X_train.mean())
           print(self.m)
           print(self.b)

    def predict(self , x_test):
        return self.m * x_test + self.b

df = pd.read_csv('/home/shivam/Documents/vs code insider/data/placement.csv')
print(df.head())

x = df.iloc[:,0].values
y = df.iloc[:,1].values

print(x)
print(y)

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 2)

print(x_train.shape)

lr = model()
print(lr.fit(x_train , y_train))
print(lr.predict(x_test[0]))