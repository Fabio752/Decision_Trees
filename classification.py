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
from copy import deepcopy

class ClassifierTree:
    '''
    leftRows: after splitting, the rows left
    '''
    def __init__(self, dataset, leftRows, depth = 1):

        # make sure there are leftRows
        assert len(leftRows) > 0, \
            "No rows left. Please check buildTree recursive stop algorithm"
        
        self.dataset = dataset
        self.leftRows = leftRows
        self.depth = depth
        self.char = None # if not None, then we have reached a leaf Node
        self.splitCol = None # which col do we use to split next
        self.splitK = None
        self.pruned = False # whether this is tried to pruned
        self.majorityElem = None # majority elem
        self.left = None
        self.right = None 
        self.buildTree()

    def buildTree(self):
        currentOverallEntropy, majorityElem = self.dataset.getOverallEntropyAndMajorityElem(self.leftRows)

        self.entropy = currentOverallEntropy
        self.majorityElem = majorityElem

        validCols = self.dataset.getValidCols(self.leftRows)

        # if all samples have same label (currentOverallEntropy == 0) or dataset cannot be split further, set self.char with majority label, return None
        if currentOverallEntropy == 0 or len(validCols) == 0:
            self.char = majorityElem
        else:
            splitCol, splitK, LHSSplitRows, RHSSplitRows = self.dataset.getBestColumnAndSplit(validCols, self.leftRows)

            self.splitCol = splitCol
            self.splitK = splitK

            self.left = ClassifierTree(self.dataset, LHSSplitRows, self.depth + 1)
            self.right = ClassifierTree(self.dataset, RHSSplitRows, self.depth + 1)


    def predict(self, attrib): 
        if not self.char is None:
            return self.char
        
        attr = attrib[self.splitCol]

        if (attr <= self.splitK):
            return self.left.predict(attrib)

        return self.right.predict(attrib)            

    def __repr__(self, indentationLevel = ""):
        retStr = ""
        if not self.char is None:
            retStr += indentationLevel + self.char + "\n"
        else:
            retStr += indentationLevel + "COL" + str(self.splitCol) + "|ENT:" + str(round(self.entropy, 3)) + "|D" + str(self.depth) + "\n"
            # LEFT
            retStr += indentationLevel + "[<=" + str(self.splitK) + "]\n"
            retStr += self.left.__repr__(indentationLevel + "\t")
            # RIGHT
            retStr += indentationLevel + "[> " + str(self.splitK) + "]\n"
            retStr += self.right.__repr__(indentationLevel + "\t")
        return retStr

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

        self.classifierTree = ClassifierTree(dataset, range(dataset.totalInstances))

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