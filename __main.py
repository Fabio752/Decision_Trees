import numpy as np

from dataset import ClassifierDataset
from classification import DecisionTreeClassifier
from prune import Prune

pathToSimple1 = './data/simple1.txt'
pathToSimple2 = './data/simple2.txt'
pathToTest = './data/test.txt'
pathToToy = './data/toy.txt'
pathToToy2 = './data/toy2.txt'
pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'
pathToSub = './data/train_sub.txt'
pathToValid = './data/validation.txt'
pathToExample = './data/example.txt'


dataset = ClassifierDataset()
dataset.initFromFile(pathToNoisy)

dtc = DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)

validationDataset = ClassifierDataset()
validationDataset.initFromFile(pathToValid)

Prune(dtc, validationDataset.attrib, validationDataset.labels)

print(dtc.__repr__(10))