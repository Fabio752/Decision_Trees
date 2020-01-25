##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train() and predict() methods of the
# DecisionTreeClassifier
##############################################################################

import numpy as np
import dataset as ds
import pickle

# Defines how we split into nodes
def getBinKey(x):
    if (x < 8):
        return "0,7"
    else:
        return "8,15"

# Print Tree in JSON format
def printTree(tree, indentationLevel = ""):
    if not tree.char is None:
        print(indentationLevel + "\"" + tree.char + "\": true")
    else:
        print(indentationLevel + "\"" +str(tree.splitCol) + "\": {")
        for key, val in tree.next.items():
            print(indentationLevel + "\t\"[" + key + "]\": {")
            printTree(val, indentationLevel + "\t\t")
            print(indentationLevel + "\t},")
        print(indentationLevel + "},")


class ClassifierTree:
    '''
    usedCols: Column already used, don't use again
    leftRows: after splitting, the rows left
    '''
    def __init__(self, dataset, usedCols, leftRows):
        self.char = None # if not None, then we have reached a leaf Node
        self.dataset = dataset
        self.usedCols = usedCols
        self.leftRows = leftRows
        self.splitCol = None # which col do we use to split next
        self.entropy = None
        self.next = self.buildTree2() # maps to the next tree if not a leaf Node. They key is "0,4" for example . The values are the subsequent trees

    # second method
    def buildTree2(self):
        currentOverallEntropy, majorityElem = self.dataset.getOverallEntropyAndMajorityElem(self.leftRows)
        attribCount = self.dataset.attribCount # number of attributes
        # print(attribCount)
        unusedCols = [x for x in range(attribCount) if x not in self.usedCols] # unusedCols we can continue splitting

        # if all samples have same label (currentOverallEntropy == 0) or dataset cannot be split further, set self.char with majority label, return None
        if currentOverallEntropy == 0 or len(unusedCols) == 0:
            self.char = majorityElem
            return None

        self.entropy = currentOverallEntropy

        splitCol, splitDict = self.dataset.getBestColumnAndSplit(unusedCols, self.leftRows)
        self.splitCol = splitCol

        next = {}
        
        nextUsedCols = [splitCol]
        for c in self.usedCols:
            nextUsedCols.append(c) # deepcopy

        for splitRange, splitRows in splitDict.items():
            next[splitRange] = ClassifierTree(self.dataset, nextUsedCols, splitRows) 

        return next



    # return a dict mapping from key to next trees
    # Actually build the tree based on usedCols
    def buildTree(self):
        currentOverallEntropy, majorityElem = self.dataset.getOverallEntropyAndMajorityElem(self.leftRows)
        attribCount = self.dataset.attribCount # number of attributes
        colsToInvestigate = [x for x in range(attribCount) if x not in self.usedCols] # unusedCols we can continue splitting

        # if all samples have same label (currentOverallEntropy == 0) or dataset cannot be split further, set self.char with majority label, return None
        if currentOverallEntropy == 0 or len(colsToInvestigate) == 0:
            self.char = majorityElem
            return None

        self.entropy = currentOverallEntropy
        # best IG just means lowest overallEntropy, so we are looking to minimise overallEntropy
        overallEntropy = float('inf') # large constant
        chosenCol = -1
        binRows = {} 
        for col in colsToInvestigate:
            gottenOverallEntropy, gottenBinRows = self.dataset.getColumnEntropy(col, self.leftRows)
            if (gottenOverallEntropy < overallEntropy):
                chosenCol = col
                overallEntropy = gottenOverallEntropy
                binRows = gottenBinRows

        # Make sure that chosenCol is not -1
        assert chosenCol != -1, \
            "buildTree failed. No column chosen"

        # we have chosen a col to split with, hooray
        self.splitCol = chosenCol

        next = {}
        
        nextUsedCols = [chosenCol]
        for c in self.usedCols:
            nextUsedCols.append(c) # deepcopy

        for key in binRows:
            next[key] = ClassifierTree(self.dataset, nextUsedCols, binRows[key])
        
        return next

    def predict(self, attrib):
        tree = self
        while tree.char is None:
            attr = attrib[tree.splitCol]
            key = getBinKey(attr)
            assert key in tree.next, \
                "predict failed, key not found!"
            tree = tree.next[key]
        return tree.char

    def __repr__(self, indentationLevel = ""):
        retStr = ""
        if not self.char is None:
            retStr += indentationLevel + "\"" + self.char + "\"\n"
        else:
            for key, val in self.next.items():
                retStr += (indentationLevel + str(self.splitCol) + ":" + "[" + key + "]" + "\n")
                retStr += val.__repr__(indentationLevel + "\t")
        return retStr

# # Print Tree in JSON format
# def printTree(tree, indentationLevel = ""):
#     if not tree.char is None:
#         print(indentationLevel + "\"" + tree.char + "\": true")
#     else:
#         print(indentationLevel + "\"" +str(tree.splitCol) + "\": {")
#         for key, val in tree.next.items():
#             print(indentationLevel + "\t\"[" + key + "]\": {")
#             printTree(val, indentationLevel + "\t\t")
#             print(indentationLevel + "\t},")
#         print(indentationLevel + "},")

class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.classifierTree = None


    def train(self, x, y):
        """ Constructs a decision tree classifier from data

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        y : numpy.array
            An N-dimensional numpy array

        Returns
        -------
        DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance

        """

        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."



        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################

        dataset = ds.Dataset()
        dataset.initFromData(x, y)

        self.classifierTree = ClassifierTree(dataset, [], range(dataset.totalInstances))

        # print("{")
        # printTree(self.classifierTree, "\t")
        # print("}")

        print(self.classifierTree)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self


    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)


        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(len(x)):
            predictions[i] = self.classifierTree.predict(x[i])

        # remember to change this if you rename the variable
        return predictions

    # write object (model) to a file
    def writeToFile(self, outfilepath):
        with open(outfilepath, 'wb') as file:
            pickle.dump(self, file)
        print("Written to file" + outfilepath)

    # read object (model) from a file
    def readFromFile(self, infilepath):
        with open(infilepath, 'rb') as file:
            self = pickle.load(file)
        print("Read from file: " + infilepath)