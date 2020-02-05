## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the skeleton code for the ``DecisionTreeClassifier`` class. Your task 
is to implement the ``train()`` and ``predict()`` methods.


- ``eval.py``

	* Contains the skeleton code for the ``Evaluator`` class. Your task is to 
implement the ``confusion_matrix()``, ``accuracy()``, ``precision()``, 
``recall()``, and ``f1_score()`` methods.


- ``example_main.py``

	* Contains an example of how the evaluation script on LabTS might use the classes
and invoke the methods defined in ``classification.py`` and ``eval.py``.


### Instructions
- The project contains some files for visualisation and data analysis purposes as well as the required ones.
    - Non-required files 
        - `__main_eval_test.py` :
        	- run this file with the command: <strong>`python3 __main_eval_test.py`</strong>
        	- The purpose of this file is to generate the confusion matrix, accuracy and calculate macro average recall, precision and f1 for each training.set. 

	    - `__main_prune.py` :
    	    - run this file with the command: <strong>python3 __main_prune.py</strong>
    	    -The purpose of this file is to determine unpruned and pruned accuracy 	on input datasets, number of nodes pruned (as well as number of parent leaves) and decreasing in the tree's max depth.

	    - `__main_draw.py` :
    	    - run this file with the command: <strong>python3 __main_draw.py</strong>
    	    - The purpose of this file is to generate a pdf file to visualise the tree (pruned or unpruned).
	
	    -  `__profiler.py`:
      	    - run this file with the command: <strong>python3 __profiler.py</strong>
      	    - The purpose of this file is to generate a table to assess the execution time for training different datasets.
	
	    -  __q1.py:
      	    - run this file with the command: <strong>python3 __profiler.py</strong>
      	    - The purpose of this file is to generate a table to assess the execution time for training different datasets.

    - Required files 
        -  classification.py:
            - ClassifierTreeStats: a class storing statistics of the ClassifierTree (nodes, leaves and maxDepth)
            - ClassifierTree: a class storing the decision tree with member functions: 
            
			    - init: initialise it by passing: 
                    - dataset: the dataset to classify.
                    - splitObject: a splitObject object.
                    - treeStats: a TreeStats object.
                    - depth (optional, default = 0):
                    - parent (def = None):
     		
     		    - buildTree: (no arguments taken), just builds the structure
			        - predict: takes one argument, returns prediction:
    			    - attrib: attributes to predict.
  			    
  			    - __repr__: takes two arguments, returns string-version of the tree
    			    - maxDepth (def = None): specify a maxDepth for the tree.
    			    - pre (def = ""): indentation. 
     	
		 
		    - DecisionTreeClassifier:  A class for the making a decision tree classifier object. Has an attribute <em> is_trained: bool </em> that keeps track of whether the classifier has been trained. It has also the following methods:
         	    
         	    - train: this method constructs the classifier from the data. It returns a copy of the classifier tree instance.
				    - x: numpy.array (N x K) where N is the number of instances and K the number of attributes.
				    - y: numpy.array (N x 1) storing the outcomes.   
			
			    - predict: this method takes one argument as input and predicts the outcomes from the given samples returning a (N x 1) numpy.array. It assumes the classifier has already been trained.
             	    - x: numpy.array (N x K) where N is the number of instances and K the number of attributes.
  			
			    - __repr__: takes two arguments, returns string-version of the tree
    			    - maxDepth (def = None): specify a maxDepth for the tree.
    
	    -  dataset.py: 
            - ClassifierDataset: a class that has member functions to calculate best split for a given dataset and range.  
                
                - initFromFile: a function that takes as input a path to a file(pathToFile) and returns the classifier.
                
                - initFromData: takes two parameters (attrib and labels) that allow it to instantiate the object from the given data.  
    
	    -  eval.py:
      	    - Evaluator: this class has several methods that can be called to evaluate the classifier predicitions based on various metrics.
        	    - confusion_matrix: Computes the confusion matrix on the given classifier. Has the following parameters.
            	- prediction: np.array containing the predicted class labels
            	- annotation: np.array containing the ground truths
            	- class_labels: np.array containing the ordered set of class labels. If not provided, default value will be the unique values in annotation. 
        	    - accuracy: calculates accuracy given the matrix
        	    - precision: calculates precision given the matrix
            	- recall: calculates recall given the matrix
        		- f1_score: calculates f1_score given the matrix
    
	    -  prune.py:
            -  Prune: a class that can be initialised passing the following parameters and will prune the tree and print the metrics analysis on the pruned tree.
                - decisionTreeClassifier: an object representing the tree to prune
                - validationAttrib: the attributes for the validation set.  
                - validationLabel: the labels for the validation set.
                - aggressive: boolean (def = false). Pruning aggressively means prune even when the accuracy after pruning stays the same.
    
        -  visualise.py:
            -  TreeVisualiser: class that initialises and plot the tree. Takes several parameters:
                - decisionTreeClassifier: the object to print.
                - maxPlotDepth: int value to indicate the depth level on which to stop the printing (def = None).
                - compact: boolean (def = false) that enables compact mode.
                - filename: the name of the output file (def = visualiser_output).
                - format: the format of the output file, supported svg, jpg, png or pdf. (def = svg).   
    
	-  k_fold.py:
       -   
  


