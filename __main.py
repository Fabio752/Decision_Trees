import numpy as np
import dataset as ds
import classification as cs
import prune

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


dataset = ds.ClassifierDataset()
dataset.initFromFile(pathToNoisy)

dtc = cs.DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)

print(dtc.__repr__(10))