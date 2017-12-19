import random
import pandas as pd
import numpy as np
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):
    """Construct perceptron."""

    def __init__(self):
        self.lr = 0.00001
        self.max_iteration = 5000
        self.w = None

    def predict_(self, x):
        pass

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        iters = 0

        while iters < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum(self.w[j] * x[j] for j in range(len(self.w)))

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.lr * (y * x[i])
