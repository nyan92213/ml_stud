import numpy as np
import pandas as pd
import perceptron
import decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

df = pd.read_csv('Iris.csv')
# print(df.head())

y = df.iloc[0:100, 5].values
y = np.where(y == 'Iris-setosa', -1, 1)
# print(y)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# # plot data
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='blue', marker='x', label='versicolor')
#
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
#
# plt.tight_layout()
# plt.show()

ppn = perceptron.Perceptron(eta=0.1, n_iter=20)

ppn.fit(X, y)
#
# plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
#
# plt.tight_layout()
# plt.show()

decision_regions.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
plt.show()