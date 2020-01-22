import numpy as np
import dataset as ds
import q1 as q1

pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'
pathToSub = './data/train_sub.txt'

datasetFull = ds.Dataset()
datasetFull.initFromFile(pathToSub)
print(datasetFull.getColumnEntropy(1))
