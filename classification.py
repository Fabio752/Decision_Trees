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
import math
from copy import deepcopy

class ClassifierTree:
    '''
    leftRows: after splitting, the rows left
    '''
    def __init__(self, dataset, splitObject, depth=0, parent=None):

        # make sure there are leftRows
        assert len(splitObject.rows) > 0, \
            "No rows left. Please check buildTree recursive stop algorithm"
        
        self.dataset = dataset

        self.leftRows = splitObject.rows # rows left in the current node
        self.entropy = splitObject.entropy # current entropy
        self.majorityElem = splitObject.majorityElem # most frequent label
        self.depth = depth # depth of tree
        self.parent = parent # parent of tree

        self.char = None # if not None, then we have reached a leaf Node (i.e. char is the label)
        self.splitCol = None # which col do we use to split next
        self.splitK = None # K at which we split (<=K to left, >K to right)
        self.left = None # left subtree (<= splitK)
        self.right = None # right subtree (> splitK)

        self.buildTree()

    '''
    build the tree
    '''
    def buildTree(self):
        validCols = self.dataset.getValidCols(self.leftRows)

        # if all samples have same label (currentOverallEntropy == 0) or dataset cannot be split further, set self.char with majority label
        if math.isclose(0.0, self.entropy) or len(validCols) == 0:
            self.char = self.majorityElem
        else:
            splitCol, splitK, LHSSplit, RHSSplit = self.dataset.getBestColumnAndSplit(validCols, self.leftRows)

            self.splitCol = splitCol
            self.splitK = splitK

            self.left = ClassifierTree(self.dataset, LHSSplit, self.depth + 1, self)
            self.right = ClassifierTree(self.dataset, RHSSplit, self.depth + 1, self)

    '''
    predict given attributes 
    '''
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
            retStr += indentationLevel + "COL" + str(self.splitCol) + "|ENT:" + str(round(self.entropy, 4)) + "|D" + str(self.depth) + "\n"
            # LEFT
            retStr += indentationLevel + "{<=" + str(self.splitK) + "}\n"
            retStr += self.left.__repr__(indentationLevel + "\t")
            # RIGHT
            retStr += indentationLevel + "{> " + str(self.splitK) + "}\n"
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

        dataset = ds.ClassifierDataset()
        dataset.initFromData(x, y)

        rootSplitObject = dataset.getSplitObjectForRoot()

        self.classifierTree = ClassifierTree(dataset, rootSplitObject)

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

    def __repr__(self):
        ret = "----------------------------------\n"
        ret += "Trained: {}\n".format(self.is_trained)
        ret += "----------------------------------\n\n"
        if self.is_trained:
            ret += self.classifierTree.__repr__()
        return ret
