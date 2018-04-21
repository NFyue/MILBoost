# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_less

from MUSK import MUSK1,MUSK2
import milboost_2
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
import random
from numpy import linalg

import matplotlib.pyplot as plt


# np.norm(x1,x1')

class test_milboost(object):


    def bag_distance(self,X,y):
        self._bag_partitioning = np.cumsum(np.bincount(np.abs(y))[1:])
        self.data_bag = np.split(X, self._bag_partitioning)[:-1]
        self.label_bag = np.split(y, self._bag_partitioning)[:-1]
        length = len(self.data_bag)
        max_norms = []
        min_norms = []
        bag_center = []
        for i in range(0,length):
            average_bag = np.average(self.data_bag[i],axis=0)
            dist = []
            for each in self.data_bag[i]:
                dist.append(linalg.norm(each - average_bag))
            max_norms.append(max(dist))
            min_norms.append(min(dist))

            bag_center.append(average_bag)

        self.bag_center = bag_center
        self.max_norms = max_norms
        self.min_norms = min_norms


    def change(self, data, label, m):
        self.bag_distance(data,label)
        data_bag_ = self.data_bag
        label_bag_ = self.label_bag

        max_val_list = []
        min_val_list = []
        length = len(data_bag_)
        length_feature = len(data[1])
        for i in range(0, length_feature):
            array = data[:,i]
            max_val = np.amax(array)
            min_val = np.amin(array)
            max_val_list.append(max_val)
            min_val_list.append(min_val)

        for i in range(0, length):  #for every bag
            for each in range(0,len(data_bag_[i])): #for every instance in bag
                data2 = data_bag_[i][each].copy() #copy every instance
                for j in range(0,m): #number of changed features
                    x = random.randint(0, length_feature-1) #which feature will change
                    data2[x] = random.uniform(min_val_list[x], max_val_list[x]) #change
                data_bag_[i] = np.concatenate((data_bag_[i], [data2]), axis = 0)

                dist_new = linalg.norm(data2 - self.bag_center[i])
                if dist_new <= self.max_norms[i] and dist_new >= self.min_norms[i]:
                    new_label = label_bag_[i][each]
                else:
                    new_label = - label_bag_[i][each]
                label_bag_[i] = np.concatenate((label_bag_[i],[new_label]), axis=0)


        data_ins_ = []
        label_ins_ = []
        bag_partitioning_new = []

        count = 0
        for i in range(0, length):  #for every bag
            for each in range(0,len(data_bag_[i])):
                data_ins_.append(data_bag_[i][each])
                label_ins_.append(label_bag_[i][each])
            count +=len(data_bag_[i])

            bag_partitioning_new.append(count)

        data_ins_ = np.array(data_ins_)
        label_ins_ = np.array(label_ins_)
        bag_partitioning_new = np.array(bag_partitioning_new)

        return data_bag_,label_bag_,data_ins_,label_ins_,bag_partitioning_new

    def test_milboost_musk12_(self):


        clf = milboost_2.MILBoostClassifier(
            # base_estimator=DecisionTreeClassifier(max_depth=10),
            # softmax=LogSumExponential(5.0),
            n_estimators=50,
            learning_rate=1.0,
            random_state = None
        )

        
        pred = []
        m_change = []
        m = 0
        data1 = MUSK1()
        data2 = MUSK2()
        label_o = np.copy(data2.labels)
        data_bag, label_bag, data, label, partition = self.change(data2.data, data2.labels, m)        
            
        clf.fit(data, label, label_o, data_bag, label_bag, partition)
        data_bag1, label_bag1, data1, label1, partition1 = self.change(data1.data, data1.labels, m)
        pred_result = clf.score(data1, label1, data_bag1, label_bag1, partition1)
        pred.append(pred_result)
        m_change.append(m)
        m += 1
        while m<=166:
            data1 = MUSK1()
            data2 = MUSK2()
            label_o = np.copy(data2.labels)
            data_bag, label_bag, data, label, partition = self.change(data2.data, data2.labels, m)        
            
            clf.fit(data, label, label_o, data_bag, label_bag, partition)
            data_bag1, label_bag1, data1, label1, partition1 = self.change(data1.data, data1.labels, m)
            pred_result = clf.score(data1, label1, data_bag1, label_bag1, partition1)
            print (pred_result)
            pred.append(pred_result)
            m_change.append(m)
            m *= 2
        print (m_change)
        print (pred)
        return m_change, pred

if __name__ == '__main__':
    print (111111111)
    m, pred = test_milboost().test_milboost_musk12_()
    plt.plot(m2, pred2)
    plt.ylabel('prediction accuracy')
    plt.xlabel('m=numbers of changed features')
    plt.savefig('prediction accuracy2.png')
    print ("successful")



