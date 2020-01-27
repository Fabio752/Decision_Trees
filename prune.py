from eval import Evaluator

class Prune:
    def __init__(self, decisionTreeClassifier, validationAttrib, validationLabel, aggressive=False):
        print("========= PRUNING =========")
        self.leaves = {} # depth to List of leaf nodes
        self.validationAttrib = validationAttrib # attribs for validation set
        self.validationLabel = validationLabel # labels for validation set
        self.decisionTreeClassifier = decisionTreeClassifier # tree
        self.aggressive = aggressive # how aggressive pruning is, if True, prune even if accuracy is same

        self.findLeaves(decisionTreeClassifier.classifierTree) # store all leaves

        self.initialAccuracy = self.getAccuracy()
        self.endAccuracy = self.initialAccuracy
        self.pruneCount = 0 # how many nodes pruned

        self.startPruning() # begin pruning
        print("Pruned {} nodes.".format(self.pruneCount))
        

    def findLeaves(self, classifierTree):
        if not classifierTree is None \
            and not classifierTree.left is None \
            and not classifierTree.right is None:
            
            # a node with left and right leaves chars
            if not classifierTree.left.char is None and not classifierTree.right.char is None:
                depth = classifierTree.depth

                if not depth in self.leaves:
                    self.leaves[depth] = []

                self.leaves[depth].append(classifierTree)
            else:
                self.findLeaves(classifierTree.left)
                self.findLeaves(classifierTree.right)

    def getAccuracy(self):
        evaluator = Evaluator()
        predictions = self.decisionTreeClassifier.predict(self.validationAttrib)
        c_matrix = evaluator.confusion_matrix(predictions, self.validationLabel)
        return evaluator.accuracy(c_matrix)
                
    def pruneMaxNode(self):
        # accuracy before pruning
        startAccuracy = self.getAccuracy()

        # get a node at the max depth untried
        maxDepth = max(list(self.leaves.keys()))
        nodeToPrune = (self.leaves[maxDepth]).pop()

        if (len(self.leaves[maxDepth]) == 0):
            del self.leaves[maxDepth]

        nodeToPrune.char = nodeToPrune.majorityElem

        pruneAccuracy = self.getAccuracy()

        if pruneAccuracy < startAccuracy \
            or (pruneAccuracy == startAccuracy and not self.aggressive): # aggressive flag
            # pruning doesn't improve accuracy
            nodeToPrune.char = None
        else:
            # pruning improves accuracy
            self.pruneCount += 1
            if not nodeToPrune.parent is None \
                and not nodeToPrune.parent.left is None \
                and not nodeToPrune.parent.left.char is None \
                and not nodeToPrune.parent.right is None \
                and not nodeToPrune.parent.right.char is None:

                newDepth = maxDepth - 1
                if not newDepth in self.leaves:
                    self.leaves[newDepth] = []
                self.leaves[newDepth].append(nodeToPrune.parent)
    
    def startPruning(self):
        print("Unpruned accuracy: {}".format(self.initialAccuracy))

        while len(self.leaves) != 0:
            self.pruneMaxNode()

        print("Pruned accuracy: {}".format(self.endAccuracy))
        



