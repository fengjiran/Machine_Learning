from __future__ import print_function

import random
import cv2
import pandas as pd
import numpy as np
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

feature_length = 324
object_num = 0
study_total = 10000


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
        self.w = None
        self.b = None

    def train(self, trainset, labels):
        trainset_size = len(labels)
        self.w = np.zeros((feature_length))
        self.b = 0.0

        study_count = 0
        nochange_count = 0
        nochange_upper_limit = 100000

        while True:
            nochange_count += 1
            if nochange_count > nochange_upper_limit:
                break

            index = random.randint(0, trainset_size - 1)
            img = trainset[index]
            label = labels[index]

            yi = int(label != object_num) * 2 - 1
            result = yi * (np.dot(img, self.w) + self.b)

            if result < 0:
                img = np.reshape(trainset[index], (feature_length, 1))
                self.w += img * yi * self.lr
                self.b += yi * self.lr

                study_count += 1
                if study_count > study_total:
                    break
                nochange_count = 0

    def predict(self, testset):
        predict = []
        for img in testset:
            result = np.dot(img, self.w) + self.b
            result = result > 0
            predict.append(result)

        return np.array(predict)


if __name__ == '__main__':
    print('Start read data')
    print('done')
