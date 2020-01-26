from eval import Evaluator

class Prune:
    def __init__(self, decisionTreeClassifier, validationAttrib, validationLabel):
        self.leaves = {} # depth to List of leaf nodes
        self.validationAttrib = validationAttrib
        self.validationLabel = validationLabel
        self.decisionTreeClassifier = decisionTreeClassifier
        self.findLeaves(decisionTreeClassifier.classifierTree)
        self.startPruning()

    def findLeaves(self, classifierTree):
        if not classifierTree is None \
            and not classifierTree.left is None \
            and not classifierTree.right is None:
            
            if not classifierTree.left.char is None and not classifierTree.right.char is None:
                depth = classifierTree.depth

                if not depth in self.leaves:
                    self.leaves[depth] = []

                self.leaves[depth].append(classifierTree)
            
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
        # print("START: " + str(startAccuracy))

        # get the max depth untried
        maxDepth = max(list(self.leaves.keys()))
        nodeToPrune = (self.leaves[maxDepth]).pop()

        if (len(self.leaves[maxDepth]) == 0):
            del self.leaves[maxDepth]

        nodeToPrune.char = nodeToPrune.majorityElem

        pruneAccuracy = self.getAccuracy()
        # print("END  : " + str(pruneAccuracy))

        if pruneAccuracy <= startAccuracy:
            # pruning doesn't improve accuracy
            nodeToPrune.char = None
            # print("NOT IMPROVED")
        else:
            # pruning improves accuracy
            print("IMPROVED")
            # if parent node exists and if parent node is a leaf
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
        while len(self.leaves) != 0:
            self.pruneMaxNode()
        



