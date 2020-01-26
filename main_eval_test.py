from eval import Evaluator
import numpy as np

p = np.array(["A","B","A","A","B","C","A","C","C","A","B","C"])
a = np.array(["B","B","C","B","A","A","B","A","C","B","C","C"])

sort = ["C", "B", "A"]


e = Evaluator()
c = e.confusion_matrix(p,a)

print(e.accuracy(c))
print(e.precision(c))
print(e.recall(c))
print(e.f1_score(c))
