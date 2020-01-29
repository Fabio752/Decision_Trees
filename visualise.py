import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class TreeVisualiser:
    def __init__(self, decisionTreeClassifier):        
        self.parentProps = dict(boxstyle='round', facecolor='wheat', alpha=1)
        self.leafProps = dict(boxstyle='round', facecolor='palegreen', alpha=1)

        self.constBoxWidth = 120
        self.constBoxHeight = 50
        self.constBoxMargin = 5
        self.constWidthSplitMultiple = self.constBoxWidth / 2 + self.constBoxMargin

        self.totalWidth = None
        self.totalHeight = None

        self.fig = None
        self.ax = None
        self.maxDepth = None

        self.decisionTreeClassifier = decisionTreeClassifier

        self.drawTree()


    def drawTree(self):
        # get figure first
        self.maxDepth = self.decisionTreeClassifier.classifierTree.treeStats.maxDepth
        self.getBlankFigure()

        initCentreX = self.totalWidth / 2
        initCentreY = self.totalHeight - self.constBoxMargin - self.constBoxHeight / 2

        self.drawNode(self.decisionTreeClassifier.classifierTree, initCentreX, initCentreY)


    #  get a blank figure
    def getBlankFigure(self):
        self.totalHeight = (self.maxDepth + 1) * (self.constBoxHeight + 2 * self.constBoxMargin)
        self.totalWidth = (2**self.maxDepth) * (self.constBoxWidth + 2 * self.constBoxMargin)

        # init figure
        self.fig = plt.figure(figsize=(self.totalWidth / float(100), self.totalHeight / float(100)), facecolor='white', dpi=100.0)

        # plot phantom axes
        self.ax = self.fig.add_axes([0,0,1,1], ylim=(0.0, self.totalHeight), xLim=(0.0, self.totalWidth))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)


    def drawNode(self, node, centreX, centreY):
        if node.label is None:
            textstr = "C={}; K={}; D={}\nEN:{:.3f}; IG:{:.3f}\nA[C]<=K    A[C] >K" \
                .format(node.splitC, node.splitK, node.depth, node.entropy, node.informationGain) 

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=8, 
            horizontalalignment='center', verticalalignment='center', bbox=self.parentProps)

            if not node.left is None:
                nextX = centreX - (self.maxDepth - node.depth) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black')
                self.drawNode(node.left, nextX, nextY)

            if not node.right is None:
                nextX = centreX + (self.maxDepth - node.depth) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black')
                self.drawNode(node.right, nextX, nextY)
        else:
            # Leaf
            textstr = node.label + "\nD={}\nEN:{:.3f}" \
                .format(node.depth, node.entropy) 

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=8, 
            horizontalalignment='center', verticalalignment='center', bbox=self.leafProps)

