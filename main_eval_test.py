from eval import Evaluator
import numpy as np
from dataset import Dataset
from classification import DecisionTreeClassifier

full_dataset_path = "./data/train_full.txt"
subset_path = "./data/train_sub.txt"
noisy_path = "./data/train_noisy.txt"
test_path = "./data/test.txt"
val_path = "./data/validation.txt"

class q3_1:
    def calc_stats(self, test_path, path_to_data):
        #load dataset, atttribs, labels
        d_subset = Dataset()
        d_subset.initFromFile(path_to_data)
        attribs = d_subset.attrib
        labels = d_subset.labels

        ds_test = Dataset()
        ds_test.initFromFile(test_path)
        test_attribs = ds_test.attrib
        test_labels = ds_test.labels

        #train and predict
        tree = DecisionTreeClassifier()
        tree.train(attribs, labels)
        predictions = tree.predict(test_attribs)

        evaluator = Evaluator()
        c_matrix = evaluator.confusion_matrix(predictions, test_labels)

        print(" ")
        print("Accuracy: " + str(evaluator.accuracy(c_matrix)))
        print(" ")

        precision, macro_p = evaluator.precision(c_matrix)
        recall, macro_r = evaluator.recall(c_matrix)
        f1, macro_f1 = evaluator.f1_score(c_matrix)

        unique_lables = np.sort(np.unique(test_labels))

        total = np.column_stack((unique_lables,precision, recall, f1))
        header_row = np.array([" ", "Prec.", "Recall", "F1"])
        final_table = np.vstack((header_row, total))

        print(final_table)

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
full_3_1.calc_stats(test_path, full_dataset_path)

print("SUBSET: ")
sub_3_1 = q3_1()
sub_3_1.calc_stats(test_path, subset_path)

print("NOISY: ")
noisy_3_1 = q3_1()
noisy_3_1.calc_stats(test_path, noisy_path)
