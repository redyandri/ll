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

    def get_ecludian_of_JensenShannon(self, histA, histB):
        histogram_types_num=12
        distances = 0
        # print "histA:%s.......histB:%s" %(str(histA),str(histB))
        #hbx","hwx","hbwx","hby","hwy","hbwy","hbm","hwm","hbwm","hbs","hws","hbws", "class"
        arr_a=np.array(histA)
        arr_b = np.array(histB)
        hbx_a=arr_a[0:8]
        hwx_a = arr_a[8:16]
        hbwx_a = arr_a[16:24]
        hby_a = arr_a[24:32]
        hwy_a = arr_a[32:40]
        hbwy_a = arr_a[40:48]
        hbm_a = arr_a[48:56]
        hwm_a = arr_a[56:64]
        hbwm_a = arr_a[64:72]
        hbs_a = arr_a[72:80]
        hws_a = arr_a[80:88]
        hbws_a = arr_a[88:96]

        hbx_b = arr_b[0:8]
        hwx_b = arr_b[8:16]
        hbwx_b = arr_b[16:24]
        hby_b = arr_b[24:32]
        hwy_b = arr_b[32:40]
        hbwy_b = arr_b[40:48]
        hbm_b = arr_b[48:56]
        hwm_b = arr_b[56:64]
        hbwm_b = arr_b[64:72]
        hbs_b = arr_b[72:80]
        hws_b = arr_b[80:88]
        hbws_b = arr_b[88:96]

        hbx_d = self.getJensenShannonDistance(hbx_a,hbx_b)
        hwx_d = self.getJensenShannonDistance(hwx_a, hwx_b)
        hbwx_d = self.getJensenShannonDistance(hbwx_a, hbwx_b)
        hby_d = self.getJensenShannonDistance(hby_a, hby_b)
        hwy_d= self.getJensenShannonDistance(hwy_a, hwy_b)
        hbwy_d = self.getJensenShannonDistance(hbwy_a, hbwy_b)
        hbm_d = self.getJensenShannonDistance(hbm_a, hbm_b)
        hwm_d = self.getJensenShannonDistance(hwm_a, hwm_b)
        hbwm_d = self.getJensenShannonDistance(hbwm_a, hbwm_b)
        hbs_d = self.getJensenShannonDistance(hbs_a, hbs_b)
        hws_d = self.getJensenShannonDistance(hws_a, hws_b)
        hbws_d = self.getJensenShannonDistance(hbws_a, hbws_b)

        ecludian_d=1.0*((hbx_d**2+hwx_d**2+hbwx_d**2+hby_d**2+hwy_d**2+hbwy_d**2+hbm_d**2+hwm_d**2+hbwm_d**2+hbs_d**2+hws_d**2+hbws_d**2)**(0.5))
        #
        # for i in range(histogram_types_num):
        #     if i>=0 & i<=7:
        #
        #     distances += (self.getJensenShannonDistance(histA[i], histB[i])) ** 2
        # l2_distance = 1.0 * distances ** (0.5)
        return ecludian_d

    def getJensenShannonDistance(self, P, Q):
        # print "test-white:%s--train_white:%s" %(str(P),str(Q))
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def classifyKNN(self, CSVFilePath):
        knn2=KNeighborsClassifier(n_neighbors=1,algorithm="ball_tree",metric=self.get_ecludian_of_JensenShannon)
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
                hl = []
                for i in range(len(r) - 2): #excluse last idx=class name
                    h_temp = [int(i) for i in p.findall(r[i])]  # convert histogram from string to array of int
                    hl+=h_temp
                hist.append(hl)
                labels.append(r[len(r) - 1]) #get class name
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
                # print "%s:%s" %(str(hist[tr]),labels[tr])
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
