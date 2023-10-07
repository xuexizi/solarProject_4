from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


class MyModel:
    def __init__(self):
        self.model = XGBClassifier(eval_metric="logloss", early_stopping_rounds=10)

    def xgboost_learn(self, filename, test_size=0.75):
        dataset = loadtxt(filename, delimiter=",")

        seed = 7
        X = dataset[:, 0:6]
        y = dataset[:, 6]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        eval_set = [(X_test, y_test)]
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)  # verbose=True：可视化loss

    def xgboost_predict(self, x):
        return self.model.predict([x])

    def xgboost_predicts(self, X):
        return self.model.predict(X)


if __name__ == "__main__":
    for i in range(4, 5):
        print("----i = ", i, "----")
        model = MyModel()
        model.xgboost_learn(str(i) + ".txt")
