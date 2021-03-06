from eval import Evaluator

class Prune:
    def __init__(self, decisionTreeClassifier, validationAttrib, validationLabel, aggressive=False):
        print("========= PRUNING =========")
        self.leafParents = {} # depth to List of nodes that are parents of only leaves
        self.validationAttrib = validationAttrib # attribs for validation set
        self.validationLabel = validationLabel # labels for validation set
        self.decisionTreeClassifier = decisionTreeClassifier # tree
        self.aggressive = aggressive # how aggressive pruning is, if True, prune even if accuracy is same

        if self.aggressive:
            print("Pruning aggressively")

        self.findLeafParents(decisionTreeClassifier.classifierTree) # store all leaves

        self.initialAccuracy = self.getAccuracy()
        self.endAccuracy = self.initialAccuracy
        self.pruneCount = 0 # how many nodes pruned

        beginNodes = self.decisionTreeClassifier.classifierTree.treeStats.nodes
        self.startPruning() # begin pruning
        endNodes = self.decisionTreeClassifier.classifierTree.treeStats.nodes
    
        beginMaxDepth = self.decisionTreeClassifier.classifierTree.treeStats.maxDepth
        self.decisionTreeClassifier.classifierTree.treeStats.maxDepth = 0
        self.updateMaxDepth(self.decisionTreeClassifier.classifierTree) # update max depth of tree
        endMaxDepth = self.decisionTreeClassifier.classifierTree.treeStats.maxDepth

        prunedNodes = beginNodes - endNodes
        print("Pruned {} out of {} nodes. ({} ParentLeaves pruned)".format(prunedNodes, beginNodes, self.pruneCount))
        print("Max depth decreased from {} to {}.".format(beginMaxDepth, endMaxDepth))

    def findLeafParents(self, classifierTree):
        if not classifierTree is None \
            and not classifierTree.left is None \
            and not classifierTree.right is None:
            
            # a parent with two leaf nodes
            if not classifierTree.left.label is None and not classifierTree.right.label is None:
                depth = classifierTree.depth

                if not depth in self.leafParents:
                    self.leafParents[depth] = []

                self.leafParents[depth].append(classifierTree)
            else:
                self.findLeafParents(classifierTree.left)
                self.findLeafParents(classifierTree.right)

    def getAccuracy(self):
        evaluator = Evaluator()
        predictions = self.decisionTreeClassifier.predict(self.validationAttrib)
        c_matrix = evaluator.confusion_matrix(predictions, self.validationLabel)
        return evaluator.accuracy(c_matrix)
                
    def pruneMaxNode(self):
        '''
        prune deepest leafParent node
        '''

        # accuracy before pruning
        startAccuracy = self.getAccuracy()

        # get a node at the max depth untried
        maxDepth = max(list(self.leafParents.keys()))
        nodeToPrune = (self.leafParents[maxDepth]).pop()

        if (len(self.leafParents[maxDepth]) == 0):
            del self.leafParents[maxDepth]

        nodeToPrune.label = nodeToPrune.majorityElem

        pruneAccuracy = self.getAccuracy()

        if pruneAccuracy < startAccuracy \
            or (pruneAccuracy == startAccuracy and not self.aggressive): # aggressive flag
            # pruning doesn't improve accuracy
            nodeToPrune.label = None
        else:
            # pruning improves accuracy
            self.decisionTreeClassifier.classifierTree.treeStats.leaves -= 1
            self.decisionTreeClassifier.classifierTree.treeStats.nodes -= 2
            self.pruneCount += 1
            self.endAccuracy = pruneAccuracy
            nodeToPrune.pruned = True
            if not nodeToPrune.parent is None \
                and not nodeToPrune.parent.left is None \
                and not nodeToPrune.parent.left.label is None \
                and not nodeToPrune.parent.right is None \
                and not nodeToPrune.parent.right.label is None:
                newDepth = maxDepth - 1
                if not newDepth in self.leafParents:
                    self.leafParents[newDepth] = []
                self.leafParents[newDepth].append(nodeToPrune.parent)
    
    def startPruning(self):
        print("Unpruned accuracy: {}".format(self.initialAccuracy))

        while len(self.leafParents) != 0:
            self.pruneMaxNode()

        print("Pruned accuracy: {}".format(self.endAccuracy))
        
    # Update maximum depth of tree
    def updateMaxDepth(self, node):
        if not node.label is None:
            self.decisionTreeClassifier.classifierTree.treeStats.maxDepth = \
                max(node.depth, self.decisionTreeClassifier.classifierTree.treeStats.maxDepth)
        else:
            self.updateMaxDepth(node.left)
            self.updateMaxDepth(node.right)       