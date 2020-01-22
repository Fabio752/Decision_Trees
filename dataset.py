import numpy as np
import math

# take in an array of probabilities, calculate entropy
def calculateEntropy(probs):
    ret = 0.
    for p in probs:
        ret -= (p * math.log2(p))
    return ret

class Dataset: 
    def __init__(self):
        self.nCol = 0
        self.attrib = np.array([])
        self.labels = np.array([])
        self.totalInstances = 0
        self.pathToFile = ''

    def initFromFile(self, pathToFile): 
        self.pathToFile = pathToFile
        self.nCol = self.getNCol()
        self.attrib = self.loadAttributes()
        self.labels = self.loadLabels()
        self.totalInstances = len(self.attrib)

    def initFromData(self, attrib, labels):
        self.attrib = attrib
        self.labels = labels
        self.nCol = len(self.attrib[0])
    
    # get number of cols in one line
    def getNCol(self):
        with open(self.pathToFile, 'r') as file:
            first_line = file.readline()
        return(first_line.count(',') + 1)

    def loadAttributes(self):
        return np.genfromtxt(self.pathToFile, delimiter=',', dtype=np.int32, usecols=range(0,self.nCol-1))

    def loadLabels(self):
        return np.genfromtxt(self.pathToFile, delimiter=',', dtype=np.unicode_, usecols=(self.nCol-1))

    # returns tuple, first: label; second: fraction
    def getLabelFractions(self):
        unique_elems, count_elems = self.getLabelCount()
        # fraction = []
        fractions = np.array([])
        for count_elem in count_elems:
            percentage = float(count_elem) / float(len(self.labels))
            fractions = np.append(fractions, percentage)
        return unique_elems, fractions

    # returns tuple, first: label; second: count
    def getLabelCount(self):
        unique_elems, count_elems = np.unique(self.labels, return_counts=True)
        return unique_elems, count_elems

    def getDictionary(self):
        dict = {}
        for i in range(len(self.labels)):
            key = ','.join(str(v) for v in self.attrib[i])
            dict[key] = str(self.labels[i])
        return(dict)

    def getLabelEntropy(self):
        _, fractions = self.getLabelFractions()
        return calculateEntropy(fractions)


            



