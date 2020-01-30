from visualise import TreeVisualiser
from classification import DecisionTreeClassifier
from dataset import ClassifierDataset
import matplotlib.pyplot as plt

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
dataset.initFromFile(pathToFull) # CHANGE PATH HERE

dtc = DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)

# first arg: Decision Tree Classifier object; second arg: max tree depth, third arg: compact mode
tv = TreeVisualiser(dtc, None, True)