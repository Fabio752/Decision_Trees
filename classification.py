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

class ClassifierTreeStats:
    '''
    ClassifierTreeStats class storing statistics of the ClassifierTree
    '''
    def __init__(self):
        self.nodes = 0
        self.leaves = 0
        self.maxDepth = 0
    
    def __repr__(self):
        retStr = "------------- LEGEND --------------\n"
        retStr += "C: column of attributes chosen to split on\n"
        retStr += "K: split boundary of column; {<=K} branch is taken if attribute A[C] <= K\n"
        retStr += "D: current depth of tree\n"
        retStr += "EN: entropy of current node\n"
        retStr += "IG: Information gain on this split\n"
        retStr += "----------------------------------\n"
        retStr += "-------------- STATS --------------\n"
        retStr += "Nodes     :{}\n".format(self.nodes)
        retStr += "Leaves    :{}\n".format(self.leaves)
        retStr += "Max Depth :{}\n".format(self.maxDepth)
        retStr += "----------------------------------\n\n"
        return retStr
    

class ClassifierTree:
    '''
    ClassifierTree structure storing the decision tree
    '''
    def __init__(self, dataset, splitObject, treeStats, depth=0, parent=None):

        # make sure there are leftRows
        assert len(splitObject.rows) > 0, \
            "No rows left. Please check buildTree recursive stop algorithm"
        
        self.dataset = dataset

        self.leftRows = splitObject.rows # rows left in the current node
        self.entropy = splitObject.entropy # current entropy
        self.majorityElem = splitObject.majorityElem # most frequent label
        self.treeStats = treeStats # ClassifierTreeStatistics
        self.depth = depth # depth of tree
        self.parent = parent # parent of tree
        self.pruned = False

        self.label = None # if not None, then we have reached a leaf Node (i.e. char is the label)
        self.splitC = None # which col do we use to split next
        self.splitK = None # K at which we split (<=K to left, >K to right)
        self.left = None # left subtree (<= splitK)
        self.right = None # right subtree (> splitK)
        self.informationGain = None # information gain on split

        self.buildTree()

    '''
    build the tree
    '''
    def buildTree(self):
        validCols = self.dataset.getValidCols(self.leftRows)

        # update treeStats
        self.treeStats.nodes += 1
        self.treeStats.maxDepth = max(self.treeStats.maxDepth, self.depth)

        # if all samples have same label (currentOverallEntropy == 0) or dataset cannot be split further, set self.label with majority label
        if math.isclose(0.0, self.entropy) or len(validCols) == 0:
            self.treeStats.leaves += 1
            self.label = self.majorityElem
            
        else:
            splitC, splitK, LHSSplit, RHSSplit, weightedEntropy = self.dataset.getOptimalSplit(validCols, self.leftRows)

            self.splitC = splitC
            self.splitK = splitK

            self.informationGain = self.entropy - weightedEntropy

            self.left = ClassifierTree(self.dataset, LHSSplit, self.treeStats, self.depth + 1, self)
            self.right = ClassifierTree(self.dataset, RHSSplit, self.treeStats, self.depth + 1, self)

    '''
    predict given attributes 
    '''
    def predict(self, attrib): 
        if not self.label is None:
            return self.label
        
        attr = attrib[self.splitC]

        if (attr <= self.splitK):
            return self.left.predict(attrib)

        return self.right.predict(attrib)            


    def __repr__(self, maxDepth=None, pre = ""):
        retStr = ""
        if not self.label is None:
            # LEAF
            if len(pre) > 4:
                pre = pre[:-4]
            retStr += pre + "+---" + self.label + " (D:{})\n".format(self.depth)
        elif maxDepth is None or (not maxDepth is None and self.depth < maxDepth):
            retStr += pre + "C:{}|K:{}|D:{}|EN:{:.3f}|IG:{:.3f}\n" \
                .format(self.splitC, self.splitK, self.depth, self.entropy, self.informationGain)

            # LEFT
            retStr += pre + "{<=" + str(self.splitK) + "}\n"
            retStr += self.left.__repr__(maxDepth, pre + "|   ")

            # RIGHT
            retStr += pre + "{> " + str(self.splitK) + "}\n"
            retStr += self.right.__repr__(maxDepth, pre + "    ")

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
        treeStats = ClassifierTreeStats()

        self.classifierTree = ClassifierTree(dataset, rootSplitObject, treeStats)

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
        # predictions = np.zeros((x.shape[0],), dtype=np.object)
        predictions = np.zeros((x.shape[0],), dtype=np.object)


        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        for i in range(len(x)):
            predictions[i] = self.classifierTree.predict(x[i])

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

    def __repr__(self, maxDepth=None):
        if not self.is_trained:
            raise Exception("Decision Tree classifier has not yet been trained.")
        ret = ""
        if not maxDepth is None:
            ret += "Printing depth is limited to {}.\n".format(maxDepth)
        ret += self.classifierTree.treeStats.__repr__()
        ret += self.classifierTree.__repr__(maxDepth)
        return ret
