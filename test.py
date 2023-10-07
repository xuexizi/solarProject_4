from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


class MyModel:
    def __init__(self):
        self.model = XGBClassifier(eval_metric="logloss", early_stopping_rounds=10)

    def xgboost_learn(self, filename):
        dataset = loadtxt('learn_xt/' + filename, delimiter=",")

        seed = 7
        test_size = 0.5
        X = dataset[:, 0:6]
        y = dataset[:, 6]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        eval_set = [(X_test, y_test)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)  # verbose=True：可视化loss

        # y_pred = model.predict(X_test)
        # predictions = [round(value) for value in y_pred]
        #
        # # 比较两个数组中不同元素的数量
        # comparison = y_test == predictions
        # num_different = len(comparison) - np.count_nonzero(comparison)
        # print("不同元素的数量:", num_different)
        #
        # accuracy = accuracy_score(y_test, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))
        #
        # y_pred_0 = model.predict([[1007.000,409.000,982.856,0.000,130.000,85.000]])
        # print(y_pred_0)


if __name__ == "__main__":
    for i in range(4, 5):
        print("----i = ", i, "----")
        xgboost_learn(str(i) + ".txt")
