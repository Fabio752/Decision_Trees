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
        self.attribCount = 0

    def initFromFile(self, pathToFile):
        self.pathToFile = pathToFile
        self.nCol = self.getNCol()
        self.attribCount = self.nCol - 1

        assert self.attribCount > 0, \
            "File must have more than 1 attributes"

        self.attrib = self.loadAttributes()
        self.labels = self.loadLabels()

        #assert self.attrib == self.labels, \
            #"File must have same number of attributes and labels"

        self.totalInstances = len(self.attrib)

    def initFromData(self, attrib, labels):
        assert len(attrib) == len(labels), \
            "Data must have same number of attributes and labels"
        self.attrib = attrib
        self.labels = labels
        self.nCol = len(self.attrib[0])
        self.attribCount = self.nCol - 1

        assert self.attribCount > 0, \
            "File must have more than 1 attributes"

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

    # # calculate overall entropy of labels in dataset
    # def getLabelEntropy(self):
    #     _, fractions = self.getLabelFractions()
    #     return calculateEntropy(fractions)

    # calculate overall entropy, majority elem in given rows
    def getOverallEntropyAndMajorityElem(self, rows):
        elem_counts = {} # map of elem to number of occurences

        for i in rows:
            label = self.labels[i]
            
            if (label not in elem_counts):
                elem_counts[label] = 0

            elem_counts[label] += 1

        totalRows = len(rows)

        probabilities = []

        for elem, count in elem_counts.items():
            probabilities.append(float(count) / totalRows)

        overallEntropy = calculateEntropy(probabilities)

        # get majority element, if duplicate, return the earlier one
        max_elem = None
        max_elem_count = 0

        for elem, count in elem_counts.items():
            if (count > max_elem_count):
                max_elem_count = count
                max_elem = elem
        return overallEntropy, max_elem


    # calculate entropy of data based on column
    # return overallEntropy, binRows (useful for tree later)
    # binRows is the rows associated with the split after the split
    # only search through rows defined in rows
    def getColumnEntropy(self, col, rows):
        # make sure columns within range of dataset
        # < because nCol includes column of label
        assert col < self.nCol, \
            "Column out of range of dataset"

        # make sure last row within range of rows in whole dataset
        assert rows[-1] < self.totalInstances, \
            "Rows out of range of dataset"

        # dict of bin to label to count
        '''
        Ex: 
        {
            "0,4": {
                "A": 4, // 4 A's in the 0,4 range
                "B": 5,
                ...
            }
        }
        '''
        binCounts = {}

        '''
        Total number of instances in a bin
        Ex:
        {
            "0,4": 22, // 22 total labels in the 0,4 range
            "5,9": 42,
            ...
        }
        ''' 
        binTotals = {}

        '''
        Rows left after a split at that bin 
        Ex:
        {
            "0,4": [1,2,3], // after we split at 0,4, these rows are left
            "5,9": [0,4,5],
            ...
        }
        '''
        binRows = {}
        for i in rows:
            attr = self.attrib[i]
            label = self.labels[i] # A
            val = attr[col] # Value at column of interest

            key = classification.getBinKey(val) # Will generate key such as "0,4"

            if not key in binCounts:
                binCounts[key] = {}
                binTotals[key] = 0
                binRows[key] = []

            #If label does not exit, then create an instance in the nested dictionary and if not increment counter
            if not label in binCounts[key]:
                binCounts[key][label] = 0

            binCounts[key][label] += 1
            binTotals[key] += 1
            binRows[key].append(i)


        # Calculate the entropy of each key based on count results
        keyEntropies = {}
        for key, binCount in binCounts.items():
            keyEntropies[key] = {}
            probabilities = []
                
            for label, count in binCount.items():
                #Append probability to probabilities array
                p = float(count) / binTotals[key]
                probabilities.append(p)

            #Calculate entropy based on the probabilities
            entropy = calculateEntropy(probabilities)
            keyEntropies[key] = entropy

        # Calculate total entropy weighted by total rows
        overallEntropy = 0.
        totalRows = len(rows)

        for key, total in binTotals.items():
            overallEntropy += float(total) / totalRows * keyEntropies[key]

        return overallEntropy, binRows
