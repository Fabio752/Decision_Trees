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

    - #### Required files 
        -  `classification.py`:
            - <em>ClassifierTreeStats</em>: a class storing statistics of the ClassifierTree (node count, leaf count, etc.)
            - <em>ClassifierTree</em>: a class storing the decision tree 		with member functions:
             	- **\_\_init__**: initialise it by passing: 
                    - dataset: the dataset to classify.
                    - splitObject: a splitObject object.
                    - treeStats: a TreeStats object.
                    - depth (optional, default = 0):
                    - parent (optional, default = None):
          		- **buildTree**: (no arguments taken), just builds the structure
			    - **predict**: takes one argument, returns prediction:
    			    - attrib: one set of attributes to predict.
  			    - **\_\_repr__**: takes one optional argument, returns text-based visualisation of the tree
    			    - maxDepth (optional, default = None): max depth for the visualisation
     	
		 
		    - <em>DecisionTreeClassifier</em>:  A class for the making a decision tree classifier object. Has an attribute <em> is_trained: bool </em> that keeps track of whether the classifier has been trained. It has also the following methods:
         	    
         	    - **train**: this method constructs the classifier from the data. It takes in arguments:
				    - x: numpy.array (N x K) where N is the number of instances and K the number of attributes.
				    - y: numpy.array (N x 1) storing the outcomes.   
    			- **predict**: this method takes one argument as input and predicts the outcomes from the given samples returning a (N x 1) numpy.array. It assumes the classifier has already been trained. It takes in an argument:
             	    - x: numpy.array (N x K) where N is the number of instances and K the number of attributes.
    			- **\_\_repr__**: takes one optional argument, returns a text-based visualisation of the tree
    			    - maxDepth (optional, default = None): max depth for the visualisation
    
	    -  `dataset.py`: 
            - <em>ClassifierDataset</em>: a class that contains the dataset and has member functions to calculate best split for a given range of data.
                - **initFromFile**: a function that takes as input a path to a file (pathToFile) and reads the file.
                - **initFromData**: takes two parameters (attrib and labels) that allow it to instantiate the object from the given data.  
                - Other functions for computing the best split while building the tree are included in the class.
	    -  `eval.py`:
      	    - <em>Evaluator</em>: this class has several methods that can be called to evaluate the classifier predicitions based on various metrics.
        	    - **confusion_matrix**: Computes the confusion matrix on the given classifier. Has the following parameters:
            	    - prediction: np.array containing the predicted class labels
            	    - annotation: np.array containing the ground truths
            	- **class_labels**: np.array containing the ordered set of class labels. If not provided, default value will be the unique values in annotation. 
        	    - **accuracy**: calculates accuracy given the confusion matrix.
        	    - **precision**: calculates precision given the confusion matrix.
            	- **recall**: calculates recall given the confusion matrix.
        		- **f1_score**: calculates f1_score given the confusion matrix.
    
	    -  `prune.py`:
            -  <em>Prune</em>: a class prunes the decision tree upon initialisation.
                - **\_\_init__**:
                    - decisionTreeClassifier: an object representing the tree to prune
                    - validationAttrib: the attributes for the validation set.  
                    - validationLabel: the labels for the validation set.
                    - aggressive: boolean (optional, default = false). Pruning aggressively means prune even when the accuracy after pruning stays the same.
    
        -  `visualise.py`:
            -  <em>TreeVisualiser</em>: class that, upon initialisation, plots a image-based visualisation of a tree.
                - **\_\_init__**:
                    - decisionTreeClassifier: the decision tree to print.
                    - maxPlotDepth: int value to indicate the depth level on which to stop the printing (optional, default = None).
                    - compact: boolean (optional, default = false) that enables compact mode.
                    - filename: the name of the output file (optional, default = visualiser_output).
                    - format: the format of the output file (supports svg, jpg, png or pdf). (optional, default = svg).   
    
    	-  `k_fold.py`:   
  


    - #### Non-required files 
		- `__main_eval_test.py` :
			- The purpose of this file is to generate the confusion matrix, accuracy and calculate macro average recall, precision and f1 for each training.set. 

		- `__main_prune.py` :
			- The purpose of this file is to determine unpruned and pruned accuracy 	on input datasets, number of nodes pruned (as well as number of parent leaves) and decreasing in the tree's max depth.

		- `__main_draw.py` :
			- The purpose of this file is to generate a pdf file to visualise the tree (pruned or unpruned).

		-  `__profiler.py` :
			- The purpose of this file is to generate a table to assess the 	execution time for training different datasets.
			- The number of test samples can be changed in the file.

		-  `__main_analysis.py`:
			- The purpose of this file is to generate a table to assess the execution time for training different datasets.
