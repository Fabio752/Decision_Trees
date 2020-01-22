import numpy as np 
import dataset_reader as ds

pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'

datasetFull = ds.Dataset(pathToFull)
datasetNoisy = ds.Dataset(pathToNoisy)

# print(ds.getWrongNumbers(datasetFull, datasetNoisy))

print(datasetFull.getLabelFractions())
print(datasetNoisy.getLabelFractions())