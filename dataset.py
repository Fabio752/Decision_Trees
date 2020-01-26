import numpy as np
import math
import classification
from copy import deepcopy

# take in an array of probabilities, calculate entropy
def calculateEntropy(probs):
    ret = 0.
    for p in probs:
        ret -= (p * math.log2(p))
    return ret

class Dataset:
    def __init__(self):
        self.nCol = 0 # number of columns including label
        self.attrib = np.array([]) # 2D array, each element is a row of attributes
        self.labels = np.array([]) # 1D array, all attributes
        self.totalInstances = 0 # number of lines
        self.pathToFile = '' # path to file if initFromFile
        self.attribCount = 0 # total number of attributes (nCol - 1)


    '''
    initialize from file
    '''
    def initFromFile(self, pathToFile):
        self.pathToFile = pathToFile
        self.nCol = self.getNCol()
        self.attribCount = self.nCol - 1

        assert self.attribCount > 0, \
            "File must have more than 1 attributes"

        self.attrib = self.loadAttributes()
        self.labels = self.loadLabels()

        assert len(self.attrib) == len(self.labels), \
            "File must have same number of attributes and labels"

        self.totalInstances = len(self.attrib)

    '''
    initialize from given data
    '''
    def initFromData(self, attrib, labels):
        assert len(attrib) == len(labels), \
            "Data must have same number of attributes and labels"
        self.attrib = attrib
        self.labels = labels

        self.nCol = len(self.attrib[0]) + 1
        self.attribCount = self.nCol - 1

        assert self.attribCount > 0, \
            "File must have more than 1 attributes"

        self.totalInstances = len(self.attrib)

    '''
    get number of columns in one line (including label column)
    '''
    def getNCol(self):
        with open(self.pathToFile, 'r') as file:
            first_line = file.readline()
        return(first_line.count(',') + 1)

    '''
    get attributes (from file)
    '''
    def loadAttributes(self):
        ret = np.array([])
        attrib = np.genfromtxt(self.pathToFile, delimiter=',', dtype=np.int32, usecols=range(0,self.nCol-1))

        # if only one column for attributes, make into 2D array
        if self.nCol == 2:
            ret = np.reshape(attrib, (-1, 1))
        else:
            ret = attrib
        return ret

    '''
    get labels (from file)
    '''
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

    '''
    Return a dictionary of attrib to dictionary of label to list of row numbers
        e.g. 
        {
            1: { 'A': [1, 3, 5] },
            2: { 'A': [2, 4], 'C': [6, 7] },
            ...
        }
    '''
    def getAttribRowDict(self, col, rows):
        attribLabelDict = {}
        for row in rows:
            attr = self.attrib[row][col]
            label = self.labels[row]
            
            if not attr in attribLabelDict:
                attribLabelDict[attr] = {}

            if not label in attribLabelDict[attr]:
                attribLabelDict[attr][label] = []

            attribLabelDict[attr][label].append(row)

        return attribLabelDict

    '''
    get entropy for a particular dictionary containing keys labels and values list of row numbers
    also get the total_count of elements in the split
    '''
    def getEntropyForSplit(self, splitDict):
        elem_counts = [] # frequency of elements
        total_count = 0
        for _, lstRow in splitDict.items():
            rowLen = len(lstRow)
            if (rowLen > 0):
                elem_counts.append(rowLen)
                total_count += rowLen
        probs = [] # probabilities
        for elemCount in elem_counts:
            probs.append(float(elemCount) / total_count)
        return calculateEntropy(probs), total_count

    '''
    get cumulative dict
    e.g. 
    {
        1: { 'A': [1, 3, 5] },
        2: { 'A': [2, 4], 'C': [6, 7] },
        3: { 'A': [9, 10]}
        ...
    }
    to 
    {
        A: [1, 3, 5, 2, 4, 9, 10],
        C: [6, 7]
    }
    the order of row numbers in the output list is ascending according to the input list's attributes
    '''
    def getCumulativeDict(self, attribLabelDict):
        ret = {}   

        attribKeys = list(attribLabelDict.keys())
        attribKeys.sort()

        for key in attribKeys:
            innerDict = attribLabelDict[key]

            for label, lstRows in innerDict.items():

                if not label in ret:
                    ret[label] = []

                ret[label] = ret[label] + lstRows

        return ret
        

    '''
    get them min entropy, splitK, LHSSplitDict, RHSSplitDict for all K in attribRange
    '''
    def getMinEntropyForColumn(self, attribLabelDict, attribRange):
        minEntropy = float('inf')
        splitK = None # K to use
        LHSSplitDict = {}
        RHSSplitDict = {}

        # initial
        LHSDict = {} # LHS node
        RHSDict = self.getCumulativeDict(attribLabelDict) # RHS node

        # for every split point k in attrib range, we
        # calculate the entropy after splitting at x <= k (LHS) and x > k (RHS)
        for K in attribRange:
            splitDict = attribLabelDict[K]

            # add to LHSDict
            for label, lstRows in splitDict.items():
                if not label in LHSDict:
                    LHSDict[label] = []
                LHSDict[label] = LHSDict[label] + lstRows

                # remove first len(lstRows) elements from RHS dict
                # this works because rows are added sequentially
                RHSDict[label] = RHSDict[label][len(lstRows):]

                if len(RHSDict[label]) == 0:
                    del RHSDict[label]

            # Now calculate the entropy 
            LHSEntropy, LHSCount = self.getEntropyForSplit(LHSDict)
            RHSEntropy, RHSCount = self.getEntropyForSplit(RHSDict)

            # get the weighted entropy
            totalCount = LHSCount + RHSCount
            weightedEntropy = LHSEntropy * LHSCount / totalCount + RHSEntropy * RHSCount / totalCount

            # check if lower than minEntropy
            if weightedEntropy < minEntropy:
                minEntropy = weightedEntropy
                splitK = K
                LHSSplitDict = deepcopy(LHSDict)
                RHSSplitDict = deepcopy(RHSDict)

        assert not splitK is None, \
            "can't get split range K"

        return minEntropy, splitK, LHSSplitDict, RHSSplitDict 

    '''
    From 
    {
        'A': [1,2,3],
        'B': [4,5]
    }
    get
    {
        [1,2,3,4,5]
    }
    '''
    def splitDictToRows(self, splitDict):
        ret = []
        
        for _, lstRow in splitDict.items():
            ret = ret + lstRow

        return ret

    '''
    Get the best split point in the column in the given rows.
    Return entropy and resulting rows in LHS and RHS after split
    '''
    def getSplitPointForColumn(self, col, rows):
        attribLabelDict = self.getAttribRowDict(col, rows)

        # sorted range of attributes in the row range, e.g. 0..15
        attribRange = list(attribLabelDict.keys())
        attribRange.sort()

        attribRange.pop() # we do not need the last element as this means the RHS is empty

        # don't split at this column if attrib are all the same
        assert len(attribRange) > 0, \
            "attrib are all same, shouldn't try this col"

        entropy, splitK, LHSSplitDict, RHSSplitDict = self.getMinEntropyForColumn(attribLabelDict, attribRange)

        LHSSplitRows = self.splitDictToRows(LHSSplitDict)
        RHSSplitRows = self.splitDictToRows(RHSSplitDict)

        return entropy, splitK, LHSSplitRows, RHSSplitRows

    '''
    Get the best column and split
    '''
    def getBestColumnAndSplit(self, validCols, rows):
        minEntropy = float('inf')
        splitCol = None
        splitK = None
        LHSSplitRows = None
        RHSSplitRows = None

        for tryCol in validCols:

            entropy, trySplitK, tryLHSSplit, tryRHSSplit = self.getSplitPointForColumn(tryCol, rows)

            if entropy < minEntropy:
                minEntropy = entropy
                splitCol = tryCol
                splitK = trySplitK
                LHSSplitRows = deepcopy(tryLHSSplit)
                RHSSplitRows = deepcopy(tryRHSSplit)

        assert not splitCol is None, \
            "Can't split anymore"

        return splitCol, splitK, LHSSplitRows, RHSSplitRows    

    '''
    get total number of attrs in a col for given rows
    '''
    def getNumberOfAttrs(self, rows, col):
        attrDict = {}

        for row in rows:
            attr = self.attrib[row][col]

            if not attr in attrDict:
                attrDict[attr] = True

        return len(attrDict)

    '''
    get all valid cols that cabe split on (more than 1 label)
    '''
    def getValidCols(self, rows):
        allCols = range(self.attribCount)
        validCols = []

        for col in allCols:
            if (self.getNumberOfAttrs(rows, col) != 1):
                validCols.append(col)

        return validCols
        