#%%
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
class Perceptron:
    def __init__(self, l_r=0.1) -> None:
        self.w = 0
        self.b = 0
        self.l_r = 0.1

    def sign(self, x):
        return np.sign(np.dot(self.w, x) + self.b)

    def fit(self, X, Y, iter=50, w=None, b=None, l_r=None):
        if w is not None:
            if np.shape(X)[1] == len(w):
                self.w = w
            else:
                raise ValueError(
                    "dimesion of w {} differs from X {}.".format(
                        len(w), np.shape(X)[1]
                    )
                )
        else:
            self.w = np.zeros(np.shape(X)[1])
        if b is not None:
            self.b = b
        if l_r is not None:
            self.l_r = l_r

        cur = wrong_times = 0
        while cur < iter:
            id = 0
            while id < len(X):
                if -1 * Y[id] * self.sign(X[id]) >= 0:
                    self.w += self.l_r * np.dot(Y[id], X[id])
                    self.b += self.l_r * Y[id]
                    wrong_times += 1
                id += 1
            cur += 1
            print("\riterated {}/{} times.".format(cur, iter), end="")
        print("", end="\n")
        print(
            "w {}, b {}, learning_rate {}, wrong times {}".format(
                self.w, self.b, self.l_r, wrong_times
            )
        )


################################################################################
# MNIST Data
#%%
def load_mnist_data(path) -> Tuple[list, list]:
    X = []
    Y = []
    with open(path, "r") as fi:
        print("start to read {}".format(path))
        for line in fi.readlines():
            curLine = line.strip().split(",")
            Y.append(1 if int(curLine[0]) >= 5 else -1)
            X.append([int(num) / 255 for num in curLine[1:]])
        print("read end.")
    return X, Y


#%%
# load mnist data
X, Y = load_mnist_data(
    "/home/ias/workdir/ml-primary/ml-notes/data/Mnist/mnist_train/mnist_train.csv"
)
X_t, Y_t = load_mnist_data(
    "/home/ias/workdir/ml-primary/ml-notes/data/Mnist/mnist_test/mnist_test.csv"
)

# %%
# train on mnist train data
start_time = time.time()
pc = Perceptron(l_r=0.001)
pc.fit(X, Y, 50)
end_time = time.time()
print("training costs {} s".format(end_time - start_time))

#%%
# test on mnist test data
right = 0
wrong = 0
errors = []
for i in range(len(X_t)):
    y_t = 1 if Y_t[i] >= 5 else -1
    if -1 * Y_t[i] * pc.sign(X_t[i]) < 0:
        right += 1
    else:
        wrong += 1
print(
    "right {}, error {}, rate {}%".format(
        right, wrong, right / (right + wrong) * 100
    )
)

################################################################################
# Sklearn iris data
#%%
def load_sk_data():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_iris

    # load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["label"] = iris.target
    df.columns = [
        "sepal length",
        "sepal width",
        "petal length",
        "petal width",
        "label",
    ]
    df.label.value_counts()
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    Y = np.array([1 if i == 1 else -1 for i in y])
    return X, Y, data


# %%
# load sk data
X, Y, data = load_sk_data()

# %%
# train on sk data
start_time = time.time()
pc = Perceptron()
pc.fit(X, Y, 200)
end_time = time.time()
print("training costs {} s".format(end_time - start_time))

#%%
x_ = np.linspace(4, 7, 10)
y_ = -(pc.w[0] * x_ + pc.b) / pc.w[1]

plt.plot(x_, y_)
plt.plot(data[:50, 0], data[:50, 1], "bo", color="blue", label="0")
plt.plot(data[50:100, 0], data[50:100, 1], "bo", color="orange", label="1")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()


# %%
# sklearn example
from sklearn.linear_model import Perceptron

# sk_pc = Perceptron(fit_intercept=True, max_iter=50, shuffle=True)
# sk_pc.fit(X, Y)
# print(sk_pc.coef_)
# print(sk_pc.intercept_)
sk_pc = Perceptron(fit_intercept=True, max_iter=100, shuffle=True)
sk_pc.fit(X, Y)
sk_pc.score(X, Y)

# %%
