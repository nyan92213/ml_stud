import numpy as np
import pandas as pd
import adaline_classificator
import decision_regions
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')
# print(df.head())

y = df.iloc[0:100, 5].values
y = np.where(y == 'Iris-setosa', -1, 1)
# print(y)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# выполняем стандартизацию
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

ada1 = adaline_classificator.AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = adaline_classificator.AdalineGD(n_iter=10, eta=0.01).fit(X_std, y)
# ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum-squared-error')
# ax[1].set_title('Adaline - Learning rate 0.01+стандартизация')
#
# ada3 = adaline_classificator.AdalineGD(n_iter=10, eta=0.000001).fit(X, y)
# ax[2].plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
# ax[2].set_xlabel('Epochs')
# ax[2].set_ylabel('Sum-squared-error')
# ax[2].set_title('Adaline - Learning rate 0.000001')
#
# plt.tight_layout()
# # plt.savefig('./adaline_1.png', dpi=300)
# plt.show()

ada = adaline_classificator.AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

decision_regions.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
plt.show()
