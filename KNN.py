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
        dists = []
        distancesIndices = []
        KNNIndex = []
        KNNAttributes = []
        for trainSample in xtrain:
            dists.append(calculateEuclideanDist(testSample,trainSample))
        for dist in dists:
            distancesIndices.append([dists.index(dist),dist])
        sortedDistancesIndices = sorted(distancesIndices, key=lambda x: x[1])
        for id in sortedDistancesIndices[:k]:
            KNNIndex.append(id[0])
        for KNNid in KNNIndex:
            KNNAttributes.append(ytrain[KNNid])
        uniqueLabels = set(KNNAttributes)
        votedLabel = max(uniqueLabels, key=KNNAttributes.count)
        predicts.append(votedLabel)
    return predicts



