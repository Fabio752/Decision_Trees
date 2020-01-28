import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as pyplot
import sys

sys.path.insert(0, '../../')
import classification as cs
import dataset as ds

pathToFull = '../../data/train_full.txt'

dataset = ds.ClassifierDataset()
dataset.initFromFile(pathToFull)

dtc = cs.DecisionTreeClassifier()
dtc.train(dataset.attrib, dataset.labels)

label = ['A', 'C', 'E', 'G', 'O', 'Q']
diction = {}
means = []
counts = []

def recurse(tree):
    if not tree.label is None:
        lbl = tree.label
        
        if not lbl in diction:
            diction[lbl] = []
        
        diction[lbl].append(tree.depth)
    else:
        recurse(tree.left)
        recurse(tree.right)

recurse(dtc.classifierTree)

for lbl in label:
    means.append(np.average(diction[lbl]))
    counts.append(len(diction[lbl]))
print(means)


y_pos = np.arange(len(label))
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, label)
plt.ylabel('Count of leaf nodes')
plt.xlabel('Label')
plt.title('Count of leaf nodes against label')

plt.show()