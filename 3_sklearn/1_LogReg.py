from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# lr = LogisticRegression(C=1000.0, random_state=0)
# lr.fit(X_train_std, y_train)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# decision_regions.plot_decision_regions(X_combined_std, y_combined,
#                       classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('./figures/logistic_regression.png', dpi=300)
# plt.show()
#
# print(lr.predict_proba(X_test_std[0, :].reshape(1, -1)))

# weights, params = [], []
# for c in np.arange(-5., 15.):
#     lr = LogisticRegression(C=10. ** c, random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10 ** c)
#
# weights = np.array(weights)
# plt.plot(params, weights[:, 0],
#          label='petal length')
# plt.plot(params, weights[:, 1], linestyle='--',
#          label='petal width')
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)  #метод опорных векторов
svm.fit(X_train_std, y_train)

decision_regions.plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()