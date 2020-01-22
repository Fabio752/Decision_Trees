import numpy as np 
import dataset_reader as ds

pathToFilename = './data/train_full.txt'

dataset = ds.Dataset(pathToFilename)
print(dataset.getLabelFractions())
