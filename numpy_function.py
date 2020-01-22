import numpy as np

filename = "train_sub.txt"

with open(filename, 'r') as file:
     first_line = file.readline()
     your_data = file.readlines()
ncol = first_line.count(',') + 1


my_data = np.genfromtxt(filename, delimiter=',', dtype=np.int32, usecols=range(0,ncol-1))
print(my_data)

labels = np.genfromtxt(filename, delimiter=',', dtype=np.unicode_, usecols=(ncol-1))
print(labels)
