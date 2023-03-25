#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import catboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, f1_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import gc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import xgboost as xg
from sklearn.model_selection import cross_val_score
import pickle



dataset = pd.read_csv("/Users/saijena/Desktop/datascience/ML/Tbilisi_Housing_Challenge/data/housing_clean_1.csv")
dataset = dataset[['price', 'space', 'room', 'bedroom', 'furniture', 'latitude', 'longitude']]
dataset = dataset[dataset["latitude"].isnull()==False]
print(dataset.head(3))


columns = list(set(dataset.columns) - set(["price"]))
X = dataset[columns]
Y = dataset[["price"]]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)



model_cat = CatBoostRegressor()
model_cat.fit(X_train, y_train)
prediction_cat = model_cat.predict(X_test)
print(r2_score(y_test,prediction_cat))

filename = '/Users/saijena/Desktop/datascience/ML/Tbilisi_Housing_Challenge/model.pkl'
pickle.dump(model_cat, open(filename, 'wb'))


model = pickle.load(open(filename, 'rb'))
print("Estimated Price: " + str(model.predict([27.0, 2.0, 3.0, 1.0, 50, 30])))



