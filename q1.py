import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


# Question 1.1
# gets the minimum and maximum number from the attributes
def q1(datasetFull):
    minRes = 65536
    maxRes = 0
    for attrib in datasetFull.attrib:
        minRes = min(np.amin(attrib), minRes)
        maxRes = max(np.amax(attrib), maxRes)
    return minRes, maxRes

# Question 1.2
# Get in full dataset and sub dataset, output the class label distribution
# Plot graph using matplotlib
def q2(dataset, filename):
    label, fraction = dataset.getLabelFractions()

    y_pos = np.arange(len(label))
    
    plt.bar(y_pos, fraction, align='center', alpha=0.5)
    plt.xticks(y_pos, label)
    plt.ylabel('Proportion')
    plt.title('Proportion of labels in ' + filename + '.txt')

    plt.show()

# Question 1.3
# Get in ref full dataset and noisy dataset, output graph showing proportion difference
def q3(datasetFull, datasetNoisy):
    flabel, ffraction = datasetFull.getLabelFractions()
    nlabel, nfraction = datasetNoisy.getLabelFractions()
    
    dfraction = [] # diff in fractions
    
    for i in range(len(ffraction)):
        dfraction.append(ffraction[i] - nfraction[i])
    
    y_pos = np.arange(len(flabel))
    
    plt.bar(y_pos, dfraction, align='center', alpha=0.5)
    plt.xticks(y_pos, flabel)
    plt.ylabel('Proportion difference')
    plt.title('Proportion difference between train_full.txt and train_noisy.txt')

    plt.show()

