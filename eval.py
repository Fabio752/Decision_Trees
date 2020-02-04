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

        if class_labels is None:
            class_labels = np.unique(annotation)
            class_labels = np.sort(class_labels)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        elementsDict = {}

        for i in range(len(class_labels)):
            elementsDict[class_labels[i]] = i

        for i in range(len(annotation)):
            aChar = annotation[i]
            pChar = prediction[i]
            confusion[elementsDict[aChar]][elementsDict[pChar]] += 1

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

        accuracy = float(confusion.trace()/np.sum(confusion))

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
            p[i] = float(numerator/denominator) if denominator != 0 else 0

        macro_p = np.average(p)

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
            r[i] = float(numerator/denominator) if denominator != 0 else 0

        macro_r = np.average(r)

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
        f = np.zeros((len(confusion), ))

        #######################################################################
        #                 ** YOUR TASK: COMPLETE THIS METHOD **
        #######################################################################

        # precision
        prec, _ = self.precision(confusion)
        # recall
        rec, _ = self.recall(confusion)

        for i in range(len(confusion)):
            f[i] = 2 * rec[i] * prec[i] / (rec[i] + prec[i])

        macro_f = np.average(f)

        return (f, macro_f)
