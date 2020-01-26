from eval import Evaluator
import numpy as np
from dataset import Dataset
from classification import DecisionTreeClassifier

'''
p = np.array(["A","B","A","A","B","C","A","C","C","A","B","C"])
a = np.array(["B","B","C","B","A","A","B","A","C","B","C","C"])

sort = ["C", "B", "A"]

c = e.confusion_matrix(p,a)
print(e.accuracy(c))
print(e.precision(c))
print(e.recall(c))
print(e.f1_score(c))
'''

full_dataset_path = "./data/train_full.txt"
subset_path = "./data/train_sub.txt"
noisy_path = "./data/train_noisy.txt"
test_path = "./data/test.txt"

#load dataset, atttribs, labels
d_subset = Dataset()
d_subset.initFromFile(subset_path)
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

print("Accuracy: " + str(evaluator.accuracy(c_matrix)))


precision, macro_p = evaluator.precision(c_matrix)
recall, macro_r = evaluator.recall(c_matrix)
f1, macro_f1 = evaluator.f1_score(c_matrix)

total = np.column_stack((precision, recall, f1))
print(total)


print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1 Score: " + str(f1))

print("Macro avg recall:" + str(macro_r))
print("Macro avg precision:" + str(macro_p))
print("Macro avg f1:" + str(macro_f1))


e = Evaluator()
