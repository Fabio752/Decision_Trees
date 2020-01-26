##############################################################################
# CO395: Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks:
# Complete the following methods of Evaluator:
# - confusion_matrix()
# - accuracy()
# - precision()
# - recall()
# - f1_score()
##############################################################################

import numpy as np
debug = False

class Evaluator(object):
    """ Class to perform evaluation
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        '''
        print("Predicted: " + str(prediction))
        print("Annotated: " + str(annotation))
        print("-------")
        '''

        if not class_labels:
            class_labels = np.unique(annotation)
            class_labels = np.sort(class_labels)


        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        elements_dict = {}

        for i in range(len(class_labels)):
            elements_dict[class_labels[i]] = i

        #print(elements_dict)

        for i in range(len(annotation)):
            a_character = annotation[i]
            p_character = prediction[i]
            confusion[elements_dict[a_character]][elements_dict[p_character]]+=1

        print("Confusion matrix: ")

        '''classes_transposed = np.transpose(class_labels)
        print(classes_transposed)
        full_confusion_matrix = np.hstack((classes_transposed, confusion))
        print(full_confusion_matrix)'''

        return confusion


    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        accurate = confusion.trace()
        '''for i in range(len(confusion)):
            accurate += confusion[i][i]'''

        accuracy = float(str(float(accurate/np.sum(confusion)))[0:5])

        return accuracy


    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros(len(confusion), dtype=np.float)

        for i in range(len(p)):
            numerator=confusion[i][i]
            denominator = np.sum(confusion, axis = 0)[i]
            if debug: print(numerator, denominator)
            p[i] = float(str(numerator/denominator)[0:5]) if denominator != 0 else 0


        macro_p = float(str(np.average(p))[0:5])

        return (p, macro_p)


    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros(len(confusion), dtype=np.float)

        for i in range(len(r)):
            numerator=confusion[i][i]
            denominator = np.sum(confusion, axis = 1)[i]
            if debug: print(numerator, denominator)
            r[i] = float(str(numerator/denominator)[0:5]) if denominator != 0 else 0

        macro_r = float(str(np.average(r))[0:5])

        return (r, macro_r)


    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros(len(confusion), dtype=np.float)

        precision, _ = self.precision(confusion)
        recall, _ = self.recall(confusion)

        for i in range(len(f)):
            f[i] = float(str(2*recall[i]*precision[i]/(recall[i]+precision[i]))[0:5]) if (recall[i]+precision[i]) != 0 else 0

        macro_f = float(str(np.average(f))[0:5])

        return (f, macro_f)
