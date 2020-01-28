import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# distribution of leaves
labels = range(22)
leaves = [0,0,0,3, 1, 2, 7, 26, 19, 26, 36, 32, 27, 28, 19, 8, 8, 17, 18, 0, 3, 2]
    
cumSum = 0

for i in range(22):
    cumSum += i * leaves[i]

mean = cumSum / np.sum(leaves)
print(np.sum(leaves))
print(mean)

y_pos = np.arange(len(labels))
plt.bar(y_pos, leaves, align='center', alpha=0.5)
plt.xticks(y_pos, labels)
plt.ylabel('Number of leaf nodes')
plt.xlabel('Depth')
plt.title('Number of leaf nodes against depth of decision tree')

plt.show()