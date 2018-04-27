import csv
import sys
import numpy as np
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from scipy.stats import entropy
from numpy.linalg import norm


class KNNClassifier(object):
    def __init(self):
        return

    def getJensenShannonDistance(self, P, Q):
        # print "test-white:%s--train_white:%s" %(str(P),str(Q))
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def classifyKNN(self, CSVFilePath):
        knn2=KNeighborsClassifier(n_neighbors=1,algorithm="ball_tree",metric=self.getJensenShannonDistance)
        hist=[]
        labels=[]
        tests=[]
        p=re.compile("\d+")
        with open(CSVFilePath,"rb") as myCsv:
            reader=csv.reader(myCsv)
            c=0
            for r in reader:
                if c==0: #if header, skip it
                    c = c + 1
                    continue
                i=[int(i) for i in p.findall(r[0])] #convert histogram from string to array of int
                hist.append(i)
                labels.append(r[1])
                c=c+1
            myCsv.close()


        tests=[]
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        kfold=KFold(n_splits=7)
        true=0
        false=0
        train_inc=1
        test_inc=1

        for train_idx, test_idx in kfold.split(hist):
            train_fold=[]
            train_labels=[]
            test_fold=[]
            test_labels=[]
            for tr in train_idx:
                train_fold.append(hist[tr])
                train_labels.append(labels[tr])
            for te in test_idx:
                test_fold.append(hist[te])
                test_labels.append(labels[te])

            print "length TRAIN(%d)" %(len(train_fold))
            print "length TEST(%d)"  %(len(test_fold))
            knn2.fit(train_fold,train_labels)
            for i in range(0,len(test_fold)):
                pred=knn2.predict([test_fold[i]])
                if (pred==test_labels[i]):
                    #score=get_distance(hist[test],hist[test])
                    true=true+1
                    print "TEST idx:%d/%d:TRUE" % (i+1, len(test_fold))
                else:
                    false=false+1
                    print "TEST idx:%d/%d:(%s):FALSE" % (i+1, len(test_fold),pred)
                test_inc=test_inc+1
            print "(%d/%d)true:%d, false:%d, Accuracy:%f" %(train_inc,len(train_fold),true,false,(1.0*true/(true+false)))
            train_inc=train_inc+1
            break
