# import numpy as np
# import dataset as ds
# import classification as cs
# import prune
# import cProfile

# pathToSimple1 = './data/simple1.txt'
# pathToSimple2 = './data/simple2.txt'
# pathToTest = './data/test.txt'
# pathToToy = './data/toy.txt'
# pathToToy2 = './data/toy2.txt'
# pathToFull = './data/train_full.txt'
# pathToNoisy = './data/train_noisy.txt'
# pathToSub = './data/train_sub.txt'
# pathToValid = './data/validation.txt'

# dataset = ds.ClassifierDataset()
# dataset.initFromFile(pathToFull)

# dtc = cs.DecisionTreeClassifier()
# print("FULL")
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')

# dataset.initFromFile(pathToSub)
# print("=====")
# print("SUB")
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')

# dataset.initFromFile(pathToNoisy)
# print("=====")
# print("NOISY")
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')
# cProfile.run('dtc.train(dataset.attrib, dataset.labels)', None, 'time')