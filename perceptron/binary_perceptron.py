import random
import cv2
import pandas as pd
import numpy as np
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def get_hog_feature(trainset):
    features = []
    hog = cv2.HOGDescriptor('hog.xml')
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features, (-1, 324))
    return features


class Perceptron(object):
    """Construct perceptron."""

    def __init__(self):
        self.lr = 0.00001
        self.max_iteration = 5000
        self.w = None

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        iters = 0

        while iters < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.lr * (y * x[i])

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1.0)
            labels.append(self.predict_(x))

        return labels


if __name__ == '__main__':
    print('Start read data')
