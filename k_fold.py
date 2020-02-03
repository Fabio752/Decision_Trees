import numpy as np
from dataset import ClassifierDataset
from eval import Evaluator
from classification import DecisionTreeClassifier

class k_fold_validator:
    def __init__(self, k, path_to_dataset):
        self.dataset = ClassifierDataset()
        self.dataset.initFromFile(path_to_dataset)
        assert k == 0 or self.dataset.totalInstances % k == 0, \
            "k must be more than 0 and a factor of the length of dataset: ({})" \
            .format(int(dataset.totalInstances))
        self.k = k
        self.models = []

    def split_dataset(self):
        n_rows = self.dataset.totalInstances
        rows_per_split = int(n_rows/self.k)
        indices = np.random.permutation(n_rows)
        subsets = np.ndarray((self.k, rows_per_split), dtype=np.int32)
        for i in range(self.k):
            base_index = i*rows_per_split
            subsets[i] = indices[base_index:base_index+rows_per_split]

        self.train_indices = np.ndarray((self.k, n_rows - rows_per_split), dtype=np.int32)
        self.test_indices = np.ndarray((self.k, rows_per_split), dtype=np.int32)

        for j in range(self.k):
            self.test_indices[j] = subsets[j]
            self.train_indices[j] = [i for i in indices if i not in subsets[j]]

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

            '''
            output_path = "model_{}.pickle".format(int(i))
            tree.writeToFile(output_path)'''

            self.models.append(tree)

            e = Evaluator()
            c_matrix = e.confusion_matrix(predictions, test_labels, all_class_labels)

            self.accuracy_scores[i] = e.accuracy(c_matrix)

        avg_accuracy = np.average(self.accuracy_scores)
        std_dev = np.std(self.accuracy_scores)

        print(avg_accuracy, std_dev)

    def plot_confusion_matrix(cm,
                              target_names,
                              name,
                              title='Confusion matrix',
                              cmap=None,
                              normalize=False):


        import matplotlib.pyplot as plt
        import itertools

        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy


        if cmap is None:
            cmap = plt.get_cmap('Blues')

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('Predicted label')
        plt.xlabel('True label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(name + '.pdf')
        return


    def test_best_model(self, test_path):
        print(self.accuracy_scores)
        max_index = np.argmax(self.accuracy_scores)
        print("Index: {}".format(int(max_index)))

        '''
        model_path = "model_" + str(max_index) + ".pickle"
        tree = DecisionTreeClassifier()
        tree.readFromFile(model_path)
        tree.is_trained = True'''

        tree = self.models[max_index]
        print(tree.__repr__)

        test_dataset = ClassifierDataset()
        test_dataset.initFromFile(test_path)

        predictions = tree.predict(test_dataset.attrib)
        evaluator = Evaluator()
        c_matrix = evaluator.confusion_matrix(predictions, test_dataset.labels)

        a = ["A", "C", "E", "G", "O", "Q"]
        b = "q3/best_model_eval_" + test_path[7:-4]

        #return stats
        plot_confusion_matrix(c_matrix, a, b)
        print(" ")
        print("Accuracy: " + str(evaluator.accuracy(c_matrix)))
        print(" ")
        precision, macro_p = evaluator.precision(c_matrix)
        recall, macro_r = evaluator.recall(c_matrix)
        f1, macro_f1 = evaluator.f1_score(c_matrix)
        print(" ")
        print("Macro avg recall:" + str(macro_r))
        print("Macro avg precision:" + str(macro_p))
        print("Macro avg f1:" + str(macro_f1))
        print(" ")
        return





path_to_data = "./data/train_full.txt"

kf = k_fold_validator(10, path_to_data)
kf.perform_validation()

test_path = "./data/test.txt"
kf.test_best_model(test_path)
