import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from dataset import ClassifierDataset
from eval import Evaluator


# Question 1.1
# gets the minimum and maximum number from the attributes
def getMinMax(datasetFull):
    minRes = 65536
    maxRes = 0
    for attrib in datasetFull.attrib:
        minRes = min(np.amin(attrib), minRes)
        maxRes = max(np.amax(attrib), maxRes)
    return minRes, maxRes

# Question 1.2
# Get in full dataset and sub dataset, output the class label distribution
# Plot graph using matplotlib
def plotProportion(dataset, filename):
    label, fraction = dataset.getLabelFractions()

    y_pos = np.arange(len(label))
    
    plt.bar(y_pos, fraction, align='center', alpha=0.5)
    plt.xticks(y_pos, label)
    plt.ylabel('Proportion')
    plt.title('Proportion of labels in ' + filename + '.txt')

    plt.show()

# Question 1.3
# Get in ref full dataset and noisy dataset, output graph showing count difference (noisy - full)
# Positive value means how many more did noisy get compared to full
def getLabelCount(datasetFull, datasetNoisy):
    full_label, full_count = datasetFull.getLabelCount()
    _, noisy_count = datasetNoisy.getLabelCount()

    diff_count = [] # difference (noisy - full)

    for i in range(len(full_count)):
        diff_count.append(noisy_count[i] - full_count[i])

    y_pos = np.arange(len(full_label))
    
    plt.bar(y_pos, diff_count, align='center', alpha=0.5)
    plt.xticks(y_pos, full_label)
    plt.ylabel('Count difference')
    plt.title('Count difference of each label (train_noisy.txt w.r.t. train_full.txt)')

    plt.show()

# Question 1.4
# take in a full dataset and a noisy dataset
# return mistakes of noisy dataset relative to full dataset
def q4(full_dat, noisy_dat):
    ref_dict = full_dat.getDictionary()
    wrongNo = 0
    for i in range(len(noisy_dat.attrib)):
        key = ",".join(str(v) for v in noisy_dat.attrib[i])
        noisyVal = noisy_dat.labels[i]
        if (noisyVal != ref_dict[key]):
            wrongNo += 1
    return wrongNo


'''
Get confusion matrix for noisy predictions based on ground
truths of train_full.txt
'''
def q4confmat(full_dat, noisy_dat):
    ref_dict = full_dat.getDictionary()
    
    # ground truth labels
    annotations = []
    for attrib in noisy_dat.attrib:
        attribString = ','.join(str(v) for v in attrib)
        if not attribString in ref_dict:
            print("ERROR: attribString not present!")
            continue
        annotations.append(ref_dict[attribString])
    evaluator = Evaluator()
    c_matrix = evaluator.confusion_matrix(noisy_dat.labels, annotations) # KUNAL
   
# full_dat = ClassifierDataset()
# full_dat.initFromFile('./data/train_full.txt')
# noisy_dat = ClassifierDataset()
# noisy_dat.initFromFile('./data/train_noisy.txt')