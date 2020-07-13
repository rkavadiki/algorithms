import numpy as np
import os
from classifier.Gradient_Descent import Gradient_Descent


class LogisticRegression(object):

    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate

    def fit(self, x, y):
        gd = Gradient_Descent()
        gd.optimizeWeights(x, y)
