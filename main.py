import numpy as np
import dataset as ds
import classification as cs
import q1 as q1

pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'
pathToSub = './data/train_sub.txt'
pathToTest = './data/simple1.txt'
pathToToy = './data/toy.txt'


dataset = ds.Dataset()
dataset.initFromFile(pathToToy)

dtc = cs.DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)

# print("predicted: " + str(dtc.predict(np.array([
# [8,12,7,8],
# [5,11,6,7],
# [1,8,7,6],
# [0,8,7,4],
# [2,11,11,2],
# [0,8,7,4],
# [7,14,8,8],
# [2,9,9,3],
# [3,8,7,7],
# [5,11,7,6],
# [3,7,4,3],
# [3,8,8,2],
# [6,11,8,8],
# [5,10,12,8],
# [1,12,11,0],
# [1,8,7,5],
# [3,14,12,0],
# ]))))