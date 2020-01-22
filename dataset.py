import numpy as np
import math
import classification

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
        self.splitter = classification.Splitter()

    def initFromFile(self, pathToFile):
        self.pathToFile = pathToFile
        self.nCol = self.getNCol()

        assert self.nCol > 1, \
            "File must have more than 1 attributes"

        self.attrib = self.loadAttributes()
        self.labels = self.loadLabels()

        #assert self.attrib == self.labels, \
            #"File must have same number of attributes and labels"

        self.totalInstances = len(self.attrib)

    def initFromData(self, attrib, labels):
        assert attrib == labels, \
            "Data must have same number of attributes and labels"
        self.attrib = attrib
        self.labels = labels
        self.nCol = len(self.attrib[0])
        self.totalInstances = len(self.attrib)

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

    # calculate overall entropy of labels in dataset
    def getLabelEntropy(self):
        _, fractions = self.getLabelFractions()
        return calculateEntropy(fractions)

    # calculate entropy of data based on column
    def getColumnEntropy(self, col):
        # < because nCol includes column of label
        assert col < self.nCol, \
            "Column out of range of dataset"

        binCounts = {} # dict containing key consisting of (splitterKey and Label) to value number of Counts
        # Example: [0,4] : A: 10

        for i in range(self.totalInstances):
            attr = self.attrib[i]
            label = self.labels[i] #A
            val = attr[col] #Value at column of interest

            key = self.splitter.getKey(val) #Will generate key such as [0,4]

            #Nested dictionary, the key will map to a dictionary containing label:count pair
            #Eg. [0,4] = {}
            if not key in binCounts:
                binCounts[key] = {}
                #Initialize total count
                binCounts[key]["total"] = 0

            #If label does not exit, then create an instance in the nested dictionary and if not increment counter
            #Eg. [0,4]:A = 1
            if not label in binCounts[key]:
                binCounts[key][label] = 1
                binCounts[key]["total"] += 1
            else:
                binCounts[key][label] += 1
                binCounts[key]["total"] += 1

            
            #print("Key: ", key, "Label: ", label, "Count. ", binCounts[key][label])

        keyEntropies = {}
        #Calculate the entropy of each key based on count results
        for key in binCounts:
            keyEntropies[key] = {}
            probabilities = []
            for label in binCounts[key]:
                #Append probability to probabilities array
                p = (binCounts[key][label])/(binCounts[key]["total"])
                probabilities.append(p)
            #Calculate entropy based on the probabilities
            entropy = calculateEntropy(probabilities)
            keyEntropies[key] = entropy
            #print("Key: ", key, "Counts:", binCounts[key])
            #print("Key: ", key, "Entropy:", entropy)

        #return keyEntropies dictionary
        return keyEntropies
