import numpy as np
import math

# take in an array of probabilities, calculate entropy
def calculateEntropy(probs):
    ret = 0.
    for p in probs:
        ret += (p * math.log2(p))
    return -ret

'''
Split Object sent to initiate a new node in the ClassifierTree
'''
class SplitObject:
    def __init__(self, majorityElem, entropy, totalCount):
        self.majorityElem = majorityElem # majority element
        self.entropy = entropy # current entropy
        self.totalCount = totalCount # total number of elements
        self.rows = [] # rows (left unsplit) in the current node

    def addRowsFromSplitDict(self, splitDict):
        for _, lstRow in splitDict.items():
            self.rows = self.rows + lstRow

    def addRowsFromRange(self, rows):
        self.rows = rows

'''
ClassiferDataset containing dataset
Can be initialized from
1. file
2. existing attrib and labels arrays

Has member functions to calculate best split for a given dataset and range
'''
class ClassifierDataset:
    def __init__(self):
        self.nCol = None # number of columns including label
        self.attrib = None # 2D array, each element is a row of attributes
        self.labels = None # 1D array, all attributes
        self.totalInstances = None # number of lines
        self.pathToFile = None # path to file if initFromFile
        self.attribCount = None # total number of attributes (nCol - 1)


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
        assert len(attrib) > 0, \
            "Attributes are empty"
        self.attrib = attrib
        self.labels = labels

        self.nCol = len(self.attrib[0]) + 1
        self.attribCount = len(self.attrib[0])

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
        ret = None
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

    '''
    get the initial SplitObject for the root of the tree
    '''
    def getSplitObjectForRoot(self):
        # dict of elem to frequency
        elemCounts = {}

        # most frequent element
        maxElem = None 
        maxElemFreq = 0
        
        # total number of elements
        totalCount = self.totalInstances

        for label in self.labels:
            elemCounts[label] = elemCounts.get(label, 0) + 1

        for label, counts in elemCounts.items():
            if counts > maxElemFreq:
                maxElemFreq = counts
                maxElem = label

        probs = [float(cnt) / totalCount for _, cnt in elemCounts.items()]
        entropy = calculateEntropy(probs)

        rootSplitObject = SplitObject(maxElem, entropy, totalCount)
        rootSplitObject.addRowsFromRange(range(self.totalInstances))

        return rootSplitObject

    '''
    Return a dictionary of attrib to dictionary of label to list of row numbers
    in the given rows
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
    get SplitObject for a particular dictionary containing keys labels and values list of row numbers
    '''
    def getSplitObjectFromSplitDict(self, splitDict):
        # dict of elem to frequency
        elemCounts = {}

        # most frequent element
        maxElem = None 
        maxElemFreq = 0
        
        # total number of elements
        totalCount = 0

        for label, lstRow in splitDict.items():
            freq = len(lstRow)
            elemCounts[label] = freq
            
            if freq >= maxElemFreq:
                maxElem = label
            
            totalCount += freq

        probs = [float(cnt) / totalCount for _, cnt in elemCounts.items()]
        entropy = calculateEntropy(probs)

        return SplitObject(maxElem, entropy, totalCount)
            

    '''
    get cumulative dict from attribLabelDict
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
    the order of row numbers in the output list is ascending according to the input dict's keys
    '''
    def getCumulativeDict(self, attribLabelDict):
        ret = {}   

        attribKeys = list(attribLabelDict.keys())
        attribKeys.sort()

        for key in attribKeys:
            innerDict = attribLabelDict[key]

            for label, lstRows in innerDict.items():
                ret[label] = ret.get(label, []) + lstRows

        return ret
        

    '''
    try every split point in a column to determine the best split
    get the resulting entropy, K, LHS SplitObject, RHS SplitObject for the best split 
    '''
    def getMinEntropyForColumn(self, attribLabelDict, attribRange):
        minEntropy = float('inf')
        splitK = None # K to use

        # initial
        LHSDict = {} # LHS node
        RHSDict = self.getCumulativeDict(attribLabelDict) # RHS node

        LHSSplit = None
        RHSSplit = None

        # for every split point k in attrib range, we
        # calculate the entropy after splitting at x <= k (LHS) and x > k (RHS)
        for K in attribRange:
            splitDict = attribLabelDict[K]

            # add to LHSDict
            for label, lstRows in splitDict.items():
                LHSDict[label] = LHSDict.get(label, []) + lstRows

                # remove first len(lstRows) elements from RHS dict
                # this works because rows are added sequentially
                RHSDict[label] = RHSDict[label][len(lstRows):]

                if len(RHSDict[label]) == 0:
                    del RHSDict[label]

            LHSSplitObject = self.getSplitObjectFromSplitDict(LHSDict)
            RHSSplitObject = self.getSplitObjectFromSplitDict(RHSDict)

            LHSEntropy, LHSCount = LHSSplitObject.entropy, LHSSplitObject.totalCount
            RHSEntropy, RHSCount = RHSSplitObject.entropy, RHSSplitObject.totalCount

            # get the weighted entropy
            totalCount = LHSCount + RHSCount
            weightedEntropy = LHSEntropy * LHSCount / totalCount + RHSEntropy * RHSCount / totalCount

            # check if lower than minEntropy
            if weightedEntropy <= minEntropy:
                minEntropy = weightedEntropy
                splitK = K

                LHSSplitObject.addRowsFromSplitDict(LHSDict)
                RHSSplitObject.addRowsFromSplitDict(RHSDict)

                LHSSplit = LHSSplitObject
                RHSSplit = RHSSplitObject

        assert not splitK is None, \
            "can't get split range K"

        return minEntropy, splitK, LHSSplit, RHSSplit 

    '''
    Get the best split point in the column in the given rows.
    Return LHS and RHS split objects
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

        entropy, splitK, LHSSplit, RHSSplit = self.getMinEntropyForColumn(attribLabelDict, attribRange)

        return entropy, splitK, LHSSplit, RHSSplit

    '''
    Get the best column and split
    '''
    def getBestColumnAndSplit(self, validCols, rows):
        minEntropy = float('inf')
        splitCol = None # which column to split
        splitK = None # which K to split

        LHSSplit = None # SplitObject for LHS node
        RHSSplit = None # SplitObject for RHS node

        for tryCol in validCols:

            entropy, trySplitK, tryLHSSplit, tryRHSSplit = self.getSplitPointForColumn(tryCol, rows)

            # current split is better (entropy smaller)
            if entropy <= minEntropy:
                minEntropy = entropy
                splitCol = tryCol
                splitK = trySplitK
                LHSSplit = tryLHSSplit
                RHSSplit = tryRHSSplit

        assert not splitCol is None, \
            "Can't split anymore"

        return splitCol, splitK, LHSSplit, RHSSplit

    '''
    get total number of unique attrs in a col for given rows
    '''
    def getNumberOfAttrs(self, rows, col):
        attrSet = set()

        for row in rows:
            attr = self.attrib[row][col]
            attrSet.add(attr)
            
        return len(attrSet)

    '''
    get all valid cols that can be split on (more than 1 label) in the given rows
    '''
    def getValidCols(self, rows):
        allCols = range(self.attribCount)

        validCols = [col for col in allCols if self.getNumberOfAttrs(rows, col) > 1]

        return validCols
        