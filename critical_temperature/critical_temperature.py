import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

formula_train = pd.read_csv('formula_train.csv')
train = pd.read_csv('train.csv')

formula_train.drop(columns='critical_temp', inplace=True)

train_full = pd.concat([train, formula_train], axis=1)
train_full.drop(columns='material', inplace=True)

stdsc = StandardScaler()
mms = MinMaxScaler()

X, y = train_full.drop(columns=['critical_temp']), train_full['critical_temp']

X_std = stdsc.fit_transform(X)
X_mms = mms.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

print('Веса признаков: ', model.coef_)
print('свободный коэф: ', model.intercept_)

# features = X_train.columns
#
# coeff_df = pd.DataFrame(model.coef_, columns=['Coeff'])
# coeff_df['Features'] = features


y_pred = model.predict(X_test)
#
print('MAE: ', mean_absolute_error(y_test, y_pred))
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))


