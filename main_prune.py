import numpy as np
import dataset as ds
from prune import Prune
import classification as cs
import q1 as q1

pathToSimple1 = './data/simple1.txt'
pathToSimple2 = './data/simple2.txt'
pathToTest = './data/test.txt'
pathToToy = './data/toy.txt'
pathToToy2 = './data/toy2.txt'
pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'
pathToSub = './data/train_sub.txt'
pathToValid = './data/validation.txt'
pathToToyValid = "./data/toyvalid.txt"

dataset = ds.ClassifierDataset()
dataset.initFromFile(pathToFull)

dtc = cs.DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)


validationDataset = ds.Dataset()
validationDataset.initFromFile(pathToValid)

Prune(dtc, validationDataset.attrib, validationDataset.labels)


print(dtc)