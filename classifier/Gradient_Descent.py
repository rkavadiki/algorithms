import numpy as np
from sklearn.metrics import roc_auc_score


class Gradient_Descent(object):
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        #self.tolerance = tolerance

    def initializeWeights(self, dim):
        w = np.random.rand(dim).reshape(dim, 1)
        return w

    def sigmoid(self, w, x):
        z = x.T.dot(w)
        return 1/(1+np.exp(-z))

    def logloss(self, ytrue, ypred):
        epsilon = 1e-15
        ypred = ypred.clip(epsilon, 1-epsilon)
        lloss = np.sum(ytrue*np.log(ypred)+(1-ytrue)*np.log(1-ypred))
        return lloss

    def optimizeWeights(self, x, y):
        # refer to https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression
        m, dim = x.shape
        y = y.values.reshape((m, 1))
        x = x.T
        w = self.initializeWeights(dim)
        for i in range(self.epochs):
            y_pred = self.sigmoid(w, x)
            error = y_pred - y
            w = w - (self.learning_rate * (1/m) * (x.dot(error)))
            y_pred = self.sigmoid(w, x)
            print(self.logloss(y, y_pred), roc_auc_score(1-y, y_pred[0]))
        return w
