import numpy as np 
import dataset_reader as ds
import q1 as q1

pathToFull = './data/train_full.txt'
pathToNoisy = './data/train_noisy.txt'
pathToSub = './data/train_sub.txt'

datasetFull = ds.Dataset(pathToFull)
datasetNoisy = ds.Dataset(pathToNoisy)
datasetSub = ds.Dataset(pathToSub)

# q1.q2(datasetNoisy, 'train_noisy')

q1.q3(datasetFull, datasetNoisy)
