from __future__ import print_function
from __future__ import division
from math import log
import numpy as np


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def calcShannonEnt_(labels):
    numEntries = len(labels)
    labelCounts = {}

    for currentLabel in labels:
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    features = ['no surfacing', 'flippers']
    return dataSet, features


if __name__ == '__main__':
    data, features = createDataSet()
    print(calcShannonEnt(data))

    labels = ['yes', 'yes', 'no', 'no', 'no']
    print(calcShannonEnt_(labels))
