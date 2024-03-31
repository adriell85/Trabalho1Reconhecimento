import pandas as pd
import numpy as np
from KNN import KNN
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit

def openIrisDataset():
    x = []
    y = []
    originalLabel = []
    ConvertLabel = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    with open("bases/iris/iris.data") as file:
        for line in file:
            label = ConvertLabel[str(line.split(',')[-1].strip())]
            originalLabel.append(str(line.split(',')[-1].strip()))
            y.append(label)
            x.append([float(feature) for feature in line.split(',')[0:4]])
    print('IRIS Dataset Opened!')
    return [x,y,np.unique(originalLabel)]



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

def datasetSplitTrainTest(x,y,percentageTrain):

    dataToSplit = [[x,y] for x,y in zip(x,y)]

    percentageTrain = 100 - percentageTrain  # porcentagem de treino

    group1, group2 = split_data_randomly(dataToSplit, percentageTrain)

    xtrain, ytrain = zip(*[(group[0],group[1]) for group in group2])
    xtest, ytest = zip(*[(group[0], group[1]) for group in group1])

    return xtrain, ytrain, xtest, ytest

def openColumnDataset():
    print('Open Column!')


def confusionMatrix(y_true, y_pred):
    # Identifica o número de classes assumindo que y_true e y_pred contêm todas as possíveis classes
    num_classes = max(max(y_true), max(y_pred)) + 1
    # Cria uma matriz de confusão vazia
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        conf_matrix[true][pred] += 1

    return conf_matrix

def plotConfusionMatrix(conf_matrix, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    # Seaborn adiciona uma camada de visualização a mais, mas é opcional
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('**True Label**')
    plt.xlabel('**Predicted Label**')
    plt.title('Confusion Matrix')
    plt.show()



if __name__ =='__main__':
    out = openIrisDataset()
    x = out[0]
    y = out[1]
    originalLabels = out[2]
    accuracyList = []

    fileName = "KNNRuns.txt"
    with open(fileName, 'w') as arquivo:
        arquivo.write("Execução Iterações KNN.\n\n")
        for i in range(20):
            print('\nIteração {}\n'.format(i))
            xtrain, ytrain, xtest, ytest = datasetSplitTrainTest(x,y,80)
            ypredict = KNN(xtrain, ytrain, xtest,5)
            confMatrix = confusionMatrix(ytest,ypredict)
            print('Confusion Matrix:\n',confMatrix)
            # plotConfusionMatrix(confMatrix,originalLabels)
            accuracy = np.trace(confMatrix) / np.sum(confMatrix)
            print('ACC:',accuracy)
            arquivo.write("ACC: {}\n".format(accuracy))
            arquivo.write("Confusion Matrix: \n {} \n\n".format(confMatrix))
            accuracyList.append(i)
        print('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList),np.std(accuracyList)))
        arquivo.write('\nAcurácia média das 20 iterações: {:.2f} ± {:.2f}'.format(np.mean(accuracyList),np.std(accuracyList)))




