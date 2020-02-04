from eval import Evaluator
import numpy as np
from dataset import ClassifierDataset
from classification import DecisionTreeClassifier
from prune import Prune
import matplotlib.pyplot as plt


full_dataset_path = "./data/train_full.txt"
subset_path = "./data/train_sub.txt"
noisy_path = "./data/train_noisy.txt"
test_path = "./data/test.txt"
val_path = "./data/validation.txt"

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

    plt.ylabel('Ground Truth')
    ax.set_xlabel('Predicted labels')
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


class q3_1:
    def calc_stats(self, test_path, path_to_data, plt_title, prune, pruneAggressively):
        #load dataset, atttribs, labels
        d_subset = ClassifierDataset()
        d_subset.initFromFile(path_to_data)
        attribs = d_subset.attrib
        labels = d_subset.labels

        ds_test = ClassifierDataset()
        ds_test.initFromFile(test_path)
        test_attribs = ds_test.attrib
        test_labels = ds_test.labels

        #train and predict
        print("TRAINING")
        tree = DecisionTreeClassifier()
        tree.train(attribs, labels)

        print("FINISHED TRAINING")
        if prune == True:
            print("PRUNING")
            validationDataset = ClassifierDataset()
            validationDataset.initFromFile(val_path)

            Prune(tree, validationDataset.attrib, validationDataset.labels, pruneAggressively)

            print("FINISHED PRUNING")


        predictions = tree.predict(test_attribs)

        evaluator = Evaluator()
        c_matrix = evaluator.confusion_matrix(predictions, test_labels)
        print(c_matrix)

        a = ["A", "C", "E", "G", "O", "Q"]
        b = path_to_data[7:-4]
        if prune :
            if pruneAggressively:
                b = b + "_aggressively_pruned"
            else :
                b += "_pruned"

        else :
            b += "_not_pruned"

        plot_confusion_matrix(c_matrix, a, plt_title)
        print(" ")
        print("Accuracy: " + str(evaluator.accuracy(c_matrix)))
        print(" ")

        precision, macro_p = evaluator.precision(c_matrix)
        recall, macro_r = evaluator.recall(c_matrix)
        f1, macro_f1 = evaluator.f1_score(c_matrix)

        p = np.append(precision, macro_p)
        r = np.append(recall, macro_r)
        f1 = np.append(f1, macro_f1)

        performance_matrix = np.vstack((p, np.vstack((r, f1))))
        print(performance_matrix)
        plot_other_stats(performance_matrix, plt_title)


        '''
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1))'''

        print(" ")
        print("Macro avg recall:" + str(macro_r))
        print("Macro avg precision:" + str(macro_p))
        print("Macro avg f1:" + str(macro_f1))
        print(" ")


print("FULL DATASET: ")
full_3_1 = q3_1()
full_3_1.calc_stats(test_path, full_dataset_path, "train_full", False, False)
'''full_3_1.calc_stats(test_path, full_dataset_path, True, False)
full_3_1.calc_stats(test_path, full_dataset_path, True, True)'''

print("SUBSET: ")
sub_3_1 = q3_1()
sub_3_1.calc_stats(test_path, subset_path,"train_sub", False, False)
'''sub_3_1.calc_stats(test_path, subset_path, True, False)
sub_3_1.calc_stats(test_path, subset_path, True, True)'''

print("NOISY: ")
noisy_3_1 = q3_1()
noisy_3_1.calc_stats(test_path, noisy_path, "train_noisy", False, False)
'''noisy_3_1.calc_stats(test_path, noisy_path, True, False)
noisy_3_1.calc_stats(test_path, noisy_path, True, True)'''



#---Task 3.1
'''
d_subset = ClassifierDataset()
d_subset.initFromFile(subset_path)
attribs = d_subset.attrib
labels = d_subset.labels

size = len(labels)

chunk = int(size*0.8)


tree = DecisionTreeClassifier()
tree.train(attribs[:chunk], labels[:chunk])
predictions = tree.predict(attribs[chunk:])

test_labels = labels[chunk:]

evaluator = Evaluator()
c_matrix = evaluator.confusion_matrix(predictions, test_labels)
print(np.unique(test_labels, return_counts=True))
print(c_matrix)
print(np.sum(c_matrix))

plot_confusion_matrix(c_matrix, ["A", "C", "E", "G", "O", "Q"], "test_plot")

import pickle

with open("cm_3_1.pickle", "wb") as f:
    pickle.dump(c_matrix, f)
'''
