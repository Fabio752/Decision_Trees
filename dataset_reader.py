import numpy as np

class Dataset: 
    def __init__(self, _pathToFile):
        self.pathToFile = _pathToFile
        self.nCol = self.getNCol()
        self.attrib = self.loadAttributes()
        self.labels = self.loadLabels()
    
    # get number of cols in one line
    def getNCol(self):
        with open(self.pathToFile, 'r') as file:
            first_line = file.readline()
        return(first_line.count(',') + 1)

    def loadAttributes(self):
        return np.genfromtxt(self.pathToFile, delimiter=',', dtype=np.int8, usecols=range(0,self.nCol-1))

    def loadLabels(self):
        return np.genfromtxt(self.pathToFile, delimiter=',', dtype=np.unicode_, usecols=(self.nCol-1))

    # countLabels
    # returns 2-element 2D array, first array: label; second array: fraction
    def getLabelFractions(self):
        unique_elems, count_elems = np.unique(self.labels, return_counts=True)
        # fraction = []
        counts = np.array([])
        for count_elem in count_elems:
            # fraction = str(count_elem) + "/" + str(len(self.labels))
            percentage = str(float(count_elem) / float(len(self.labels)))[0:5]
            counts = np.append(counts, percentage)
        return(np.array(list(zip(unique_elems,counts))))

    def getDictionary(self):
        dict = {}
        for i in range(len(self.labels)):
            key = ','.join(str(v) for v in self.attrib[i])
            dict[key] = str(self.labels[i])
        return(dict)

# take in a full dataset and a noisy dataset
# return mistakes of noisy dataset relative to full dataset
def getWrongNumbers(full_dat, noisy_dat):
    ref_dict = full_dat.getDictionary()
    wrongNo = 0
    for i in range(len(noisy_dat.attrib)):
        key = ",".join(str(v) for v in noisy_dat.attrib[i])
        noisyVal = noisy_dat.labels[i]
        if (noisyVal != ref_dict[key]):
            wrongNo += 1
    return wrongNo


