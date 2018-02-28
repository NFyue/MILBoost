#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# import warnings

import numpy as np
from sklearn.ensemble.weight_boosting import ClassifierMixin, BaseWeightBoosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fminbound, minimize

# from skboost.milboost.softmax import SoftmaxFunction


class MILBoostClassifier(ClassifierMixin, BaseWeightBoosting):

	def __init__(self,
				base_estimator=DecisionTreeClassifier(max_depth=10),
				softmax=None,
				n_estimators=50,
				learning_rate=1.0,
				random_state=None,
				verbose=False):

	super(MILBoostClassifier, self).__init__(
		base_estimator=base_estimator,
		n_estimators=n_estimators,
		learning_rate=learning_rate,
		random_state=random_state)

	self.softmax_fcn = softmax
	self._verbose = verbose

	self._bag_labels = None
	self._inferred_y = None
	self._bag_partitioning = None

	def __str__(self):
		return "{0}, with {1} {2} classifiers".format(
			self.__class__.__name__, len(self.estimators_), self.estimators_[0])

	def fit(self, X, y, sample_weight=None):
		unique_bag_ids = np.unique(y)
		self._bag_nums = len(unique_bag_ids)
		self._bag_labels = np.zeros((max(np.abs(unique_bag_ids)) + 1, ), 'int')
		self._bag_labels[np.abs(unique_bag_ids)] = np.sign(unique_bag_ids)
		self._bag_labels = self._bag_labels[1:]
		self._inferred_y = np.sign(y)
		self._bag_partitioning = np.cumsum(np.bincount(np.abs(y))[1:])

		out = super(MILBoostClassifier, self).fit(X, y, sample_weight)

	    self._bag_labels = None
	    self._inferred_y = None
	    self._bag_partitioning = None

		return out


	def _boost(self, iboost, X, y, sample_weight, random_state):
		if iboost == 0:
			bag_weight = []
			sample_weight = []
			length = self._bag_nums
			for i in range(0,length):
				bag_weight.append(1/length)
			bag_weight = np.array(bag_weight)
			for i in range(0,length):
				j = self._bag_partitioning[i]
				w2 = bag_weight[i]/j
				for m in j:
					sample_weight.append(w2)
			sample_weight = np.array(sample_weight)

			self.classes_ = getattr(estimator, 'classes_', None)
			self.n_classes_ = len(self.classes_)
			cm = []

		estimator = self._make_estimator(random_state = random_state)
		sample_weight = np.abs(sample_weight)
		estimator.fit(X, self._inferred_y, sample_weight= sample_weight)
		y_predict = estimator.y_predict(X)

		estimator_error = self.find_estimator_error(iboost, y_predict)

		bags = self._bag_split(sample_weight)
		bag_weight = []
		for x in bags:
			bag_w = np.sum(x)
			bag_weight.append(bag_w)
		bag_weight = np.array(bag_weight)

		cm, estimator_weight_cm = self.find_cm(bag_weight, estimator_error, cm)

		bag_weight, sample_weight = self.new_weight_data(estimator_weight_cm, bag_weight, estimator_error)

		return sample_weight, estimator_weight_cm, estimator_error



	def find_estimator_error(self, iboost, y_predict):
		if iboost == 0:
			estimator_error = np.zeros((self._bag_nums.shape[0],), 'float')
		else:
			incorrect = y_predict != self._inferred_y
			incorrect_bags = self._bag_split(incorrect)
			estimator_error = np.mean(incorrect_bags, axis= 1)
		return estimator_error



	def find_cm(self,bag_weight,estimator_error, cm):

		def optimization_fun(m):
			length = len(bag_weight)
			sumup_array = []
			for i in range(0,length):
				sumup = bag_weight[i]* exp((2 * estimator_error[i] - 1)*m)
				sumup_array.append(sumup)
			sumup_array = np.array(sumup_array)
			return np.sum(sumup_array)
		m = minimize(optimization_fun, 0.0, method = 'BFGS')
		cm.append(m)
		return cm, m



	def new_weight_data(iboost, estimator_weight_cm, bag_weight, estimator_error):
		length = self._bag_nums
		sample_weight = []
		for i in range(0,length):
			bag_weight[i] = bag_weight[i] * exp((2* estimator_error[i] - 1) * estimator_weight_cm)
		bag_weight = bag_weight/np.sum(np.abs(bag_weight))
		for i in range(0, length):
			j = self._bag_partitioning[i]
			ww = bag_weight[i]/j
			for m in j:
				sample_weight.append(ww)
		sample_weight = np.array(sample_weight)

		return bag_weight,sample_weight

	def decision_function(self, X):

		check_is_fitted(self, "n_classes_")
		X = self._validate_X_predict(X)

		classes = self.classes_[:, np.newaxis]
		pred = sum((estimator.predict(X) == classes).T * w
		           for estimator, w in zip(self.estimators_,
		                                   self.estimator_weights_))
		pred[:, 0] *= -1
		return pred.sum(axis=1)


	def predict(self, X):

		pred = self.decision_function(X)

		return self.classes_.take(pred > 0, axis=0)




