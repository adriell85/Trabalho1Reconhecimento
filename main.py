import pandas as pd
import numpy as np

def openIrisDataset():
    x = []
    y = []
    ConvertLabel = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    with open("bases/iris/iris.data") as file:
        for line in file:
            label = ConvertLabel[str(line.split(',')[-1].strip())]
            y.append(label)
            x.append([float(feature) for feature in line.split(',')[0:4]])
    print('IRIS Dataset Opened!')
    return [x,y]
def split_data_randomly(data, percentage):
    if percentage < 0 or percentage > 100:
        raise ValueError("A porcentagem deve estar entre 0 e 100.")
    total_data = len(data)
    size_first_group = int(total_data * (percentage / 100))
    indices = np.random.permutation(total_data)
    first_group_indices = indices[:size_first_group]
    second_group_indices = indices[size_first_group:]
    first_group = []
    second_group = []
    for indice in first_group_indices:
        first_group.append(data[int(indice)])
    for indice in second_group_indices:
        second_group.append(data[int(indice)])

    return first_group, second_group
def datasetSplitTrainTest(x,y):

    dataToSplit = [[x,y] for x,y in zip(x,y)]

    percentageTrain = 20  # porcentagem de treino

    group1, group2 = split_data_randomly(dataToSplit, percentageTrain)
    print("Grupo 1 (20%):", group1)
    print("Grupo 2 (80%):", group2)

    xtrain, ytrain = zip(*[(group[0],group[1]) for group in group1])
    xtest, ytest = zip(*[(group[0], group[1]) for group in group2])

    return xtrain, ytrain, xtest, ytest

def openColumnDataset():
    print('Open Column!')




if __name__ =='__main__':
    out = openIrisDataset()
    x = out[0]
    y = out[1]
    xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x,y, 80)



    print(xtrain)
