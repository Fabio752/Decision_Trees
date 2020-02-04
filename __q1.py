import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from dataset import ClassifierDataset
from eval import Evaluator


def plot_confusion_matrix(cm,target_names,title,cmap=None,normalize=False):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion matrix: " + title, pad = 10)
    plt.colorbar()
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = cm[j,i]
            if normalize:
                ax.text(i, j, "{:0.4f}".format(c),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(i, j, "{:,}".format(c),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            #ax.text(i, j, str(c), va='center', ha='center')

    plt.ylabel('Labels from train_full')
    ax.set_xlabel('Labels from train_noisy')
    ax.xaxis.set_label_position('top')

    text = "Accuracy={:0.4f}; Misclassification={:0.4f}".format(accuracy, misclass)
    plt.text(0.3,6.0, text)
    plt.tight_layout()
    plt.savefig("confusion_" + title + '.pdf', bbox_inches='tight')
    return

def plot_other_stats(perf_mat, plt_title):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Blues')
    im = ax.imshow(perf_mat, interpolation='nearest', cmap=cmap)
    plt.title("Performance metrics: " + plt_title, y=1.15)
    plt.colorbar(im, fraction = 0.02, pad = 0.04)
    ax.set_xticks(np.arange(7))
    ax.set_yticks(np.arange(3))
    x_labels = ["A", "C", "E", "G", "O", "Q", "Macro avg"]
    y_labels = ["Precision", "Recall", "F-1"]
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    for i in range(perf_mat.shape[1]):
        for j in range(perf_mat.shape[0]):
            c = perf_mat[j, i]
            ax.text(i, j, "{:0.4f}".format(c), va='center', ha='center')

    plt.savefig("performance_" + plt_title + '.pdf', bbox_inches='tight')
    return




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
    print(c_matrix)
    target_names = ["A", "C", "E", "G", "O", "Q"]
    plot_confusion_matrix(c_matrix, target_names, "Noisy vs Full")

    precision, macro_p = evaluator.precision(c_matrix)
    recall, macro_r = evaluator.recall(c_matrix)
    f1, macro_f1 = evaluator.f1_score(c_matrix)

    p = np.append(precision, macro_p)
    r = np.append(recall, macro_r)
    f1 = np.append(f1, macro_f1)

    performance_matrix = np.vstack((p, np.vstack((r, f1))))
    print(performance_matrix)
    plot_other_stats(performance_matrix, "Train_noisy")
    return




full_dat = ClassifierDataset()
full_dat.initFromFile('./data/train_full.txt')
noisy_dat = ClassifierDataset()
noisy_dat.initFromFile('./data/train_noisy.txt')

q4confmat(full_dat, noisy_dat)
