#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# import warnings

from abc import ABCMeta,abstractmethod
import numpy as np
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import ClassifierMixin, is_regressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DTYPE
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fminbound, minimize

from sklearn.externals import six
from sklearn.ensemble.forest import BaseForest
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_array, check_X_y,check_random_state
from sklearn.utils.validation import check_is_fitted
import math




# from skboost.milboost.softmax import SoftmaxFunction

class BaseWeightBoosting(six.with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for AdaBoost estimators.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.0,
                 random_state=None):

        super(BaseWeightBoosting, self).__init__(	
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
    	print ('000000000')

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if (self.base_estimator is None or
                isinstance(self.base_estimator, (BaseDecisionTree,
                                                 BaseForest))):
            dtype = DTYPE
            accept_sparse = 'csc'
        else:
            dtype = None
            accept_sparse = ['csr', 'csc']

        X, y = check_X_y(X, y, accept_sparse=accept_sparse, dtype=dtype, 
        	y_numeric=is_regressor(self))

        if (sample_weight is None):
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                    "Attempting to fit with a non-positive "
                    "weighted number of samples.")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            # sample_weight, estimator_weight_cm, estimator_error
            sample_weight, estimator_weight, estimator_error = self._boost(iboost, X, y, sample_weight, random_state)

            # Early termination
            if sample_weight is None:
                break
            print ('iboost')
            print (estimator_error)
            self.estimator_weights_[iboost] = estimator_weight
            # self.estimator_errors_[iboost] = estimator_error

			# count = 0
			# for i in range(0,len(estimator_error)):
			# 	if estimator_error[i]<= 0.5:
			# 		count+=1
			# print (count)
			# if count==len(estimator_error):
			# 	break

			# if estimator_weight <= 0:
			# 	break


        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        pass

    @property
    def feature_importances_(self):

        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")

    def _validate_X_predict(self, X):
        """Ensure that X is in the proper format"""
        if (self.base_estimator is None or
                isinstance(self.base_estimator,
                           (BaseDecisionTree, BaseForest))):
            X = check_array(X, accept_sparse='csr', dtype=DTYPE)

        else:
            X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        return X



class MILBoostClassifier(ClassifierMixin, BaseWeightBoosting):

	def __init__(self,
				base_estimator=DecisionTreeClassifier(max_depth=10),
				# softmax=None,
				n_estimators=50,
				learning_rate=1.0,
				random_state=None,
				verbose=False):

		super(MILBoostClassifier, self).__init__(base_estimator=base_estimator,
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			random_state=random_state)

		# self.softmax_fcn = softmax
		self._verbose = verbose

		self._bag_labels = None
		self._inferred_y = None
		self._bag_partitioning = None

	def __str__(self):
		return "{0}, with {1} {2} classifiers".format(
			self.__class__.__name__, len(self.estimators_), self.estimators_[0])

	def fit(self, X, y, y_o, X_bag, y_bag, partition, sample_weight=None):
		print ('aaaaaaaaa')
		# !!!!
		unique_bag_ids = np.unique(y_o)
		self._bag_nums = len(X_bag)
		print (self._bag_nums)
		# self._bag_labels = np.zeros((max(np.abs(unique_bag_ids)) + 1, ), 'int')
		# self._bag_labels[np.abs(unique_bag_ids)] = np.sign(unique_bag_ids)
		# self._bag_labels = self._bag_labels[1:]
		self._bag_labels = y_bag
		self._inferred_y = np.sign(y)


		# self._bag_partitioning = np.cumsum(np.bincount(np.abs(y_o))[1:])
		self._bag_partitioning = partition

		print ('partition')
		print (self._bag_partitioning)

		self.bag_sample_num = self.ins_in_bag()

		out = super(MILBoostClassifier, self).fit(X, y, sample_weight)

		# self._bag_labels = None
		# self._inferred_y = None
		# self._bag_partitioning = None

		return out

	# def y_instance(self, y):
	# 	length = self._bag_nums
	# 	y_instance = []
	# 	for i in range(0,length):
	# 		if i == 0 :
	# 			j = self._bag_partitioning[0]
	# 		else:
	# 			j = self._bag_partitioning[i]-self._bag_partitioning[i-1]
	# 		y_label = self._inferred_y[i]
	# 		for m in range(0,j):
	# 			y_instance.append(y_label)
	# 	y = np.array(y_instance)
	# 	return y

	def ins_in_bag(self):
		bag_sample_num = []
		for i in range(0, self._bag_nums):
			if i == 0:
				j = self._bag_partitioning[0]
			else:
				j = self._bag_partitioning[i]-self._bag_partitioning[i-1]
			bag_sample_num.append(j)
		print ('bag_sample_num')
		print (bag_sample_num)

		return bag_sample_num


	def _boost(self, iboost, X, y, sample_weight, random_state):
		print ('bbbbbbbb')
		# print (len(X))
		length = self._bag_nums
		# print (length)
		if iboost == 0:
			bag_weight = []
			sample_weight = []
			for i in range(0,length):
				bag_weight.append(1/length)
			bag_weight = np.array(bag_weight)
			for i in range(0,length):
				j = self.bag_sample_num[i]
				w2 = bag_weight[i]/j
				for m in range(0,j):
					sample_weight.append(w2)
			sample_weight = np.array(sample_weight)

		estimator = self._make_estimator(random_state = random_state)
		sample_weight = np.abs(sample_weight)
		estimator.fit(X, self._inferred_y, sample_weight= sample_weight)

		if iboost == 0:
			self.classes_ = getattr(estimator, 'classes_', None)
			self.n_classes_ = len(self.classes_)

		y_predict = estimator.predict(X)

		estimator_error = self.find_estimator_error(iboost, y_predict,y)

		if iboost>0:
			bags = self._bag_split(sample_weight)
			bag_weight = []
			for x in bags:
				bag_w = np.sum(x)
				bag_weight.append(bag_w)
			bag_weight = np.array(bag_weight)

		estimator_weight_cm = self.find_cm(bag_weight, estimator_error)

		bag_weight, sample_weight = self.new_weight_data(iboost, estimator_weight_cm, bag_weight, estimator_error)

		return sample_weight, estimator_weight_cm, estimator_error


	def _bag_split(self, x):
		return np.split(x, self._bag_partitioning)[:-1]


	def find_estimator_error(self, iboost, y_predict, y):
		print ('cccccccccc')
		estimator_error = np.zeros((self._bag_nums,), 'float')
		# print (y_predict[400:410])
		# print (self.y_instance(y)[400:410])
		incorrect = abs(y_predict - self._inferred_y)
		# print (len(incorrect))
		incorrect_bags = self._bag_split(incorrect)
		print (incorrect_bags[0:2])	
		for each in range(0, self._bag_nums):
			count = 0
			for zz in range(0, len(incorrect_bags[each])):
				if(incorrect_bags[each][zz] == 1):
					count += 1
			incorrect_bags[each] = count/len(incorrect_bags[each])
		# estimator_error = np.mean(incorrect_bags, axis=1)
		print ('estimator_error')
		print (estimator_error)
		return estimator_error



	def find_cm(self,bag_weight,estimator_error):
		print ('dddddddddd ')

		def optimization_fun(m):
			length = len(bag_weight)
			sumup_array = []
			for i in range(0,length):
				sumup = (bag_weight[i]) * (math.exp(((2 * estimator_error[i]) - 1)*m))
				sumup_array.append(sumup)
			sumup_array = np.array(sumup_array)
			return np.sum(sumup_array)
		m = minimize(optimization_fun, 0.0, method = 'BFGS')
		print (m['x'][0])
		return m['x'][0]



	def new_weight_data(self, iboost, estimator_weight_cm, bag_weight, estimator_error):
		length = self._bag_nums
		sample_weight = []
		bag_weight_update = []

		for i in range(0,length):

			weight_i = (bag_weight[i]) * (math.exp(((2 * estimator_error[i]) - 1) * estimator_weight_cm))
			bag_weight_update.append(weight_i)	
		bag_weight = np.array(bag_weight_update)/np.sum(np.abs(bag_weight_update))
		for i in range(0, length):
			j = self.bag_sample_num[i]
			ww = bag_weight[i]/j
			for m in range(0,j):
				sample_weight.append(ww)
		sample_weight = np.array(sample_weight)

		return bag_weight,sample_weight

	def decision_function(self, X):

		check_is_fitted(self, "n_classes_")
		X = self._validate_X_predict(X)

		classes = self.classes_[:, np.newaxis]
		pred = sum((estimator.predict(X) == classes).T * weights
                       for estimator, weights in zip(self.estimators_,
                                               self.estimator_weights_))
		
		# bag_pred = self._bag_split(pred)
		# print (len(bag_pred))
		pred /=self.estimator_weights_.sum()
		pred[:, 0] *= -1
		return pred.sum(axis=1)


	# def predict(self, X):

	# 	pred = self.decision_function(X)
	# 	pred = self._bag_split(pred)
	# 	prediction = pred.sum(axis = 1)
	# 	res = np.sign(prediction)

	# 	# res = self.classes_.take(prediction > 0, axis=0)
	# 	return res

	def score(self,X, y, X_bag, y_bag, partition):
		_bag_partitioning = partition
		print (_bag_partitioning)
		_inferred_y = np.sign(y)
		label_bags = []
		# a = []
		# for i in range(0, len(_bag_partitioning)):
		# 	m = _inferred_y_o[_bag_partitioning[i]-1]
		# 	a.append(m)
		# label_bags = np.array(a)
		print (y_bag[0:2])
		for each in y_bag:
			label = each[0]
			label_bags.append(label)
		label_bags = np.array(label_bags)
		label_bags = np.sign(label_bags)
		# label_bags = y_bag[:,0]
		print ('label_bags')
		print (label_bags)
		# print (label_bags)
		pred = self.decision_function(X)
		print ('pred')
		print (pred)
		pred = np.split(pred, _bag_partitioning)[:-1]

		prediction = []
		for each in pred:
			bag_pred = np.sum(each)
			prediction.append(bag_pred)
		prediction = np.array(prediction)
		res = np.sign(prediction)
		print (res)
		count =0
		for i in range(0,len(_bag_partitioning)):
			if res[i] == label_bags[i]:
				count += 1
		score = count/self._bag_nums

		return score





