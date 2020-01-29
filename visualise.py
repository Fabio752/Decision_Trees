import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

class TreeVisualiser:
    def __init__(self, decisionTreeClassifier, maxPlotDepth=None): 
        plt.clf()       
        self.parentProps = dict(boxstyle='round', facecolor='wheat', alpha=1, linewidth=0.1)
        self.leafProps = dict(boxstyle='round', facecolor='palegreen', alpha=1, linewidth=0.1)

        self.constBoxWidth = 300      
        self.constBoxHeight = 30
        self.constBoxMargin = 5
        self.constWidthSplitMultiple = self.constBoxWidth / 2 + self.constBoxMargin

        self.totalWidth = None
        self.totalHeight = None

        self.fig = None
        self.ax = None
        self.maxDepth = decisionTreeClassifier.classifierTree.treeStats.maxDepth

        self.dpi = 600.0
        self.maxPlotDepth = float('inf')
        if not maxPlotDepth is None and maxPlotDepth < self.maxDepth:
            self.maxPlotDepth = maxPlotDepth
            self.maxDepth = maxPlotDepth

        self.decisionTreeClassifier = decisionTreeClassifier

        self.drawTree()
        plt.show()


    def drawTree(self):
        # get figure first
        self.getBlankFigure()

        initCentreX = self.totalWidth / 2
        initCentreY = self.totalHeight - self.constBoxMargin - self.constBoxHeight / 2

        self.drawNode(self.decisionTreeClassifier.classifierTree, initCentreX, initCentreY)


    #  get a blank figure
    def getBlankFigure(self):
        self.totalHeight = (self.maxDepth + 1) * (self.constBoxHeight + 2 * self.constBoxMargin)
        self.totalWidth = (2**self.maxDepth) * (self.constBoxWidth + 2 * self.constBoxMargin)

        # init figure
        self.fig = plt.figure(figsize=(self.totalWidth / self.dpi, self.totalHeight / self.dpi), facecolor='white', dpi=self.dpi)

        # plot phantom axes
        self.ax = self.fig.add_axes([0,0,1,1], ylim=(0.0, self.totalHeight), xLim=(0.0, self.totalWidth))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)


    def drawNode(self, node, centreX, centreY):
        if node.label is None:
            textstr = "C{};K{};D{}\nEN:{:.3f}\nIG:{:.3f}" \
                .format(node.splitC, node.splitK, node.depth, node.entropy, node.informationGain) 

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=1, 
            horizontalalignment='center', verticalalignment='center', bbox=self.parentProps)


            if not node.left is None:
                nextX = centreX - (2**(self.maxDepth - node.depth - 1)) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black', linewidth=0.1)
                if node.depth < self.maxPlotDepth:
                    self.drawNode(node.left, nextX, nextY)

            if not node.right is None:
                nextX = centreX + (2**(self.maxDepth - node.depth - 1)) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black', linewidth=0.1)
                if node.depth < self.maxPlotDepth:
                    self.drawNode(node.right, nextX, nextY)
        else:
            # Leaf
            textstr ="D={}\n".format(node.depth) + node.label

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=1, 
            horizontalalignment='center', verticalalignment='center', bbox=self.leafProps)

