import numpy as np
from numba import njit

@njit
def calculateEuclideanDist(sample,attributes):
    diff =  np.array(attributes) - np.array(sample)
    elevate = np.sum((diff) ** 2)
    result = np.sqrt(elevate)
    return result

def KNN(xtrain, ytrain, xtest, k):
    predicts = []


    for testSample in xtest:
        _dists = []
        _distancesIndices = []
        _KNNIndex = []
        _KNNAttributes = []

        for trainSample in xtrain:
            _dists.append(calculateEuclideanDist(testSample,trainSample))
        for dist in _dists:
            _distancesIndices.append([_dists.index(dist),dist])
        _sortedDistancesIndices = sorted(_distancesIndices, key=lambda x: x[1])
        for id in _sortedDistancesIndices[:k]:
            _KNNIndex.append(id[0])
        for KNNid in _KNNIndex:
            _KNNAttributes.append(ytrain[KNNid])
        uniqueLabels = set(_KNNAttributes)
        votedLabel = max(uniqueLabels, key=_KNNAttributes.count)
        predicts.append(votedLabel)


    return predicts


def DMC(xtrain, ytrain, xtest):
    predicts = []
    centroids = {}
    uniqueLabels = np.unique(ytrain)
    for label in uniqueLabels:
        xtrain = np.array(xtrain)
        classSamples = xtrain[ytrain == label]
        centroids[label] = np.mean(classSamples, axis=0)
    for testSample in xtest:
        selectedLabel = None
        closestDistance = None
        for label, centroid in centroids.items():
            centroid = centroid.tolist()
            dist = calculateEuclideanDist(testSample, centroid)
            if closestDistance is None or dist < closestDistance:
                closestDistance = dist
                selectedLabel = label
        predicts.append(selectedLabel)

    return predicts
