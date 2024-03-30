import numpy as np
def calculateEuclideanDist(sample,attributes):
    diff =  np.array(attributes) - np.array(sample)
    elevate = np.sum((diff) ** 2)
    result = np.sqrt(elevate)
    return result

def KNN(xtrain, ytrain, xtest, ytest,k):

    # etapa de treino
    dataMemory = xtrain
    labelMemory = ytrain

    # etapa de treino

    distanceList = []

    for test in xtest:
        distanceList.append(calculateEuclideanDist(test,dataMemory))

    originalOrdenationList = distanceList

    distanceList.sort()

    distanceList= distanceList[0:k]

    distanceLabels = []

    kLabels = []

    for dist in distanceList:
        distanceLabels.append(originalOrdenationList.index(dist))

    for label in distanceLabels:
            kLabels.append(ytest[label])
    print(kLabels)

