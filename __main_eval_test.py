from eval import Evaluator
import numpy as np
from dataset import ClassifierDataset
from classification import DecisionTreeClassifier
from prune import Prune


full_dataset_path = "./data/train_full.txt"
subset_path = "./data/train_sub.txt"
noisy_path = "./data/train_noisy.txt"
test_path = "./data/test.txt"
val_path = "./data/validation.txt"




def plot_confusion_matrix(cm,
                          target_names,
                          name,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):


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


class q3_1:
    def calc_stats(self, test_path, path_to_data, prune, pruneAggressively):
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
        
        a = ["A", "C", "E", "G", "O", "Q"]
        b = path_to_data[7:-4]
        if prune :
            if pruneAggressively:
                b = b + "_aggressively_pruned"            
            else :
                b += "_pruned"

        else :
            b += "_not_pruned"
        
        plot_confusion_matrix(c_matrix, a, b)
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
full_3_1.calc_stats(test_path, full_dataset_path, False, False)
full_3_1.calc_stats(test_path, full_dataset_path, True, False) 
full_3_1.calc_stats(test_path, full_dataset_path, True, True)

print("SUBSET: ")
sub_3_1 = q3_1()
sub_3_1.calc_stats(test_path, subset_path, False, False)
sub_3_1.calc_stats(test_path, subset_path, True, False)
sub_3_1.calc_stats(test_path, subset_path, True, True)

print("NOISY: ")
noisy_3_1 = q3_1()
noisy_3_1.calc_stats(test_path, noisy_path, False, False)
noisy_3_1.calc_stats(test_path, noisy_path, True, False)
noisy_3_1.calc_stats(test_path, noisy_path, True, True)

