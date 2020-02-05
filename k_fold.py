import numpy as np
from dataset import ClassifierDataset
from eval import Evaluator
from classification import DecisionTreeClassifier
import matplotlib.pyplot as plt

class k_fold_validator:
    def __init__(self, k, path_to_dataset):
        self.dataset = ClassifierDataset()
        self.dataset.initFromFile(path_to_dataset)
        assert k == 0 or self.dataset.totalInstances % k == 0, \
            "k must be more than 0 and a factor of the length of dataset: ({})" \
            .format(int(dataset.totalInstances))
        self.k = k

    def split_dataset(self):
        n_rows = self.dataset.totalInstances
        rows_per_split = int(n_rows/self.k)
        indices = np.random.permutation(n_rows)
        self.test_indices = np.ndarray((self.k, rows_per_split), dtype=np.int32)
        for i in range(self.k):
            base_index = i*rows_per_split
            self.test_indices[i] = indices[base_index:base_index+rows_per_split]

        self.train_indices = np.ndarray((self.k, n_rows - rows_per_split), dtype=np.int32)

        for j in range(self.k):
            self.train_indices[j] = [i for i in indices if i not in self.test_indices[j]]

        #print(self.train_indices)
        #print(self.test_indices)

    def perform_validation(self):
        self.split_dataset()
        self.accuracy_scores = np.zeros(self.k, dtype=np.float64)
        all_class_labels = np.sort(np.unique(self.dataset.labels))

        for i in range(self.k):
            #load train and test splits
            train_attribs = self.dataset.attrib[self.train_indices[i]]
            train_labels = self.dataset.labels[self.train_indices[i]]
            test_attribs = self.dataset.attrib[self.test_indices[i]]
            test_labels = self.dataset.labels[self.test_indices[i]]

            tree = DecisionTreeClassifier()
            tree.train(train_attribs, train_labels)
            predictions = tree.predict(test_attribs)

            print("i = {}".format(int(i)))
            print(np.unique(predictions))
            print(np.unique(test_labels))

            output_path = "q3/model_{}.pickle".format(int(i))
            tree.writeToFile(output_path)

            e = Evaluator()
            c_matrix = e.confusion_matrix(predictions, test_labels, all_class_labels)

            self.accuracy_scores[i] = e.accuracy(c_matrix)

        avg_accuracy = np.average(self.accuracy_scores)
        std_dev = np.std(self.accuracy_scores)

        print(avg_accuracy, std_dev)

    def test_best_model(self, test_path):
        print(self.accuracy_scores)
        max_index = np.argmax(self.accuracy_scores)
        print("Index: {}".format(int(max_index)))

        model_path = "q3/model_" + str(max_index) + ".pickle"
        tree = DecisionTreeClassifier()
        tree.readFromFile(model_path)
        print(tree.__repr__(3))

        test_dataset = ClassifierDataset()
        test_dataset.initFromFile(test_path)

        predictions = tree.predict(test_dataset.attrib)
        evaluator = Evaluator()
        c_matrix = evaluator.confusion_matrix(predictions, test_dataset.labels)
        print(c_matrix.shape)

        a = ["A", "C", "E", "G", "O", "Q"]

        #return stats
        plot_confusion_matrix(c_matrix, a, "Most Accurate Model")
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
        plot_other_stats(performance_matrix, "Most Accurate Model")
        return



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
    plt.savefig("conf_" + title + '.pdf', bbox_inches='tight')
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

    plt.savefig(plt_title + '.pdf', bbox_inches='tight')
    return

def calc_mode(a, axis=0):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

def rand_forest(path_to_data):
    test_dataset = ClassifierDataset()
    test_dataset.initFromFile(path_to_data)
    all_predictions = np.ndarray((10,len(test_dataset.labels)), dtype=np.object)

    for i in range(10):
        model_path = "q3/model_" + str(i) + ".pickle"
        tree = DecisionTreeClassifier()
        tree.readFromFile(model_path)
        predictions = tree.predict(test_dataset.attrib)
        print(predictions)
        all_predictions[i]  = predictions

    mode = calc_mode(all_predictions)[0]
    mode = mode.flatten()
    evaluator = Evaluator()
    c_matrix = evaluator.confusion_matrix(mode, test_dataset.labels)
    print(c_matrix.shape)
    a = ["A", "C", "E", "G", "O", "Q"]
    #return stats
    plot_confusion_matrix(c_matrix, a, "Random Forest")
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
    plot_other_stats(performance_matrix, "Random Forest")
    return




path_to_data = "./data/train_full.txt"
test_path = "./data/test.txt"

kf = k_fold_validator(10, path_to_data)
kf.perform_validation()
kf.test_best_model(test_path)
rand_forest(test_path)
