import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

'''
=====
TWEAK LINES MARKED WITH *** IF THERE EXISTS MEMORY ERRORS
====
'''

class TreeVisualiser:
    '''
    Initialise and plot

    ===== PARAMETERS =====
    decisionTreeClassifier: DecisionTreeClassifier Object
    maxPlotDepth: maximum plot depth (integer)
    compact: enable compact mode (boolean)
    filename: file name to output
    format: format, either 'svg', 'jpg', 'png' or 'pdf'
    ======================
    '''
    def __init__(self, decisionTreeClassifier, maxPlotDepth=None, compact=False, filename='visualiser_output', format='svg'):
        plt.clf()
        self.parentProps = dict(boxstyle='round', facecolor='wheat', alpha=1)
        self.leafProps = dict(boxstyle='round', facecolor='palegreen', alpha=1)
        self.prunedProps = dict(boxstyle='round', facecolor='plum', alpha=1)

        self.constBoxWidth = 90 # ***
        self.constBoxHeight = 80 # ***
        self.constBoxMargin = 5 # ***
        self.constBoxMarginVertical = 5 # ***
        self.constWidthSplitMultiple = self.constBoxWidth / 2 + self.constBoxMargin

        self.totalWidth = None
        self.totalHeight = None

        self.fig = None
        self.ax = None
        self.decisionTreeClassifier = decisionTreeClassifier
        self.maxDepth = decisionTreeClassifier.classifierTree.treeStats.maxDepth

        self.dpi = 100.0
        self.maxPlotDepth = self.maxDepth
        if not maxPlotDepth is None and maxPlotDepth < self.maxDepth:
            self.maxPlotDepth = maxPlotDepth
            self.maxDepth = maxPlotDepth

        # compact storage
        self.compact = compact
        self.depthTreeMap = None
        self.nodeCoordMap = None

        if self.compact:
            self.constBoxMarginVertical = 200 # ***
            self.constBoxMargin = 20 # ***
            self.depthTreeMap = []
            for _ in range(self.maxDepth + 1): self.depthTreeMap.append([])
            self.nodeCoordMap = {} # node to coord (x, y)
            self.drawTreeCompact()
        else:
            self.drawTree()
        outfilename = filename + "." + format
        plt.axis('off')
        plt.savefig(outfilename, dpi=100, transparent=True)

    '''
    =====================================
    FUNCTIONS FOR NON-COMPACT (TIDY) TREE
    =====================================
    '''
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

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=12,
            horizontalalignment='center', verticalalignment='center', bbox=self.parentProps)


            if not node.left is None:
                nextX = centreX - (2**(self.maxDepth - node.depth - 1)) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black')
                if node.depth < self.maxPlotDepth:
                    self.drawNode(node.left, nextX, nextY)

            if not node.right is None:
                nextX = centreX + (2**(self.maxDepth - node.depth - 1)) * self.constWidthSplitMultiple
                nextY = centreY - (2 * self.constBoxMargin + self.constBoxHeight)
                self.ax.plot([centreX, nextX], [centreY, nextY], color='black')
                if node.depth < self.maxPlotDepth:
                    self.drawNode(node.right, nextX, nextY)
        else:
            # Leaf
            textstr = node.label + "\nD{}".format(node.depth)

            props = None
            if node.pruned:
                props = self.prunedProps
            else:
                props = self.leafProps

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=12,
            horizontalalignment='center', verticalalignment='center', bbox=props)


    '''
    ==========================
    FUNCTIONS FOR COMPACT TREE
    ==========================
    '''
    def processTreeCompact(self, node):
        # store tree in depthTreeMap, indexed by depth
        if not node is None and node.depth <= self.maxDepth:
            self.depthTreeMap[node.depth].append(node)

            if not node.pruned:
                if not node.left is None:
                    self.processTreeCompact(node.left)

                if not node.right is None:
                    self.processTreeCompact(node.right)

    def getBlankFigureCompact(self):
        self.totalHeight = (self.maxDepth + 1) * (self.constBoxHeight + 2 * self.constBoxMarginVertical)
        maxWidthCount = max([len(lst) for lst in self.depthTreeMap])
        self.totalWidth = maxWidthCount * (self.constBoxWidth + 2 * self.constBoxMargin)

        # init figure
        self.fig = plt.figure(figsize=(self.totalWidth / self.dpi, self.totalHeight / self.dpi), facecolor='white', dpi=self.dpi)

        # plot phantom axes
        self.ax = self.fig.add_axes([0,0,1,1], ylim=(0.0, self.totalHeight), xLim=(0.0, self.totalWidth))
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)

    def drawTreeCompact(self):
        self.processTreeCompact(self.decisionTreeClassifier.classifierTree)
        self.getBlankFigureCompact()

        centreY = self.totalHeight - self.constBoxMarginVertical - self.constBoxHeight / 2
        for nodes in self.depthTreeMap:
            self.drawTreeForDepth(nodes, centreY)
            centreY -= (2 * self.constBoxMarginVertical + self.constBoxHeight)

    def drawNodeCompact(self, node, centreX, centreY):
        if node.label is None:
            textstr = "C{};K{};D{}\nEN:{:.3f}\nIG:{:.3f}" \
                .format(node.splitC, node.splitK, node.depth, node.entropy, node.informationGain)

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=12,
            horizontalalignment='center', verticalalignment='center', bbox=self.parentProps)

            self.nodeCoordMap[node] = (centreX, centreY)

            # draw a line to the end of page for truncated nodes
            if node.depth == self.maxDepth:
                endX = centreX - self.constBoxWidth / 2
                endY = centreY - (self.constBoxMarginVertical * 2)
                self.ax.plot([centreX, endX], [centreY, endY], color='black')
                endX = centreX + self.constBoxWidth / 2
                self.ax.plot([centreX, endX], [centreY, endY], color='black')


        else:
            # Leaf
            textstr = node.label + "\nD{}".format(node.depth)

            props = None
            if node.pruned:
                props = self.prunedProps
            else:
                props = self.leafProps

            self.ax.text(centreX / self.totalWidth, centreY / self.totalHeight, textstr, transform=self.ax.transAxes, fontsize=12,
            horizontalalignment='center', verticalalignment='center', bbox=props)

    def drawTreeForDepth(self, nodes, nextY):
        # xDistanceMultiplier = self.constBoxWidth + 2 *

        # nextX = self.totalWidth / 2 - (len(nodes) - 1) * (self.constBoxMargin + self.constBoxWidth / 2)
        XDist = (self.totalWidth / 2 )/ float(len(nodes))
        nextX = XDist
        for node in nodes:
            # draw line to parent
            if not node.parent is None:
                parentX, parentY = self.nodeCoordMap.get(node.parent, (0, 0))
                self.ax.plot([parentX, nextX], [parentY, nextY], color='black')

            self.drawNodeCompact(node, nextX, nextY)
            nextX += (XDist * 2)
