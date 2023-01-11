import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from  sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

df['size'] = df['size'].map(size_mapping)

X = df[['color', 'size', 'price']].values

print(df)

# преобразование признака "color" в массив признаков по цветам. OneHotEncoder кодирует в массив np
# remainder='passthrough' оставляет остальные колонки в датасете нетронутыми, по умолчанию их дропает
ohe = ColumnTransformer([('color', OneHotEncoder(), [0])], remainder='passthrough')
print(ohe.fit_transform(X))

print(pd.get_dummies(df[['price', 'color', 'size']])) # преобразование строковых столбцов









