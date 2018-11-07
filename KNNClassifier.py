from __future__ import division
from numpy import median
import csv
import sys
import numpy as np
#import tensorflow as tf
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from victorinox import victorinox
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats import entropy
from numpy.linalg import norm
import pandas as pd
from my_tool import *
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

class KNNClassifier(object):
    def __init__(self):
        return

    def get_average_of_JensenShannon(self, histA, histB):
        histogram_types_num=12
        distances = 0
        arr_a=np.array(histA)
        arr_b = np.array(histB)
        hbx_a=arr_a[0:8]
        hwx_a = arr_a[8:16]
        hbwx_a = arr_a[16:24]
        hby_a = arr_a[24:32]
        hwy_a = arr_a[32:40]
        hbwy_a = arr_a[40:48]

        hbx_b = arr_b[0:8]
        hwx_b = arr_b[8:16]
        hbwx_b = arr_b[16:24]
        hby_b = arr_b[24:32]
        hwy_b = arr_b[32:40]
        hbwy_b = arr_b[40:48]

        hbx_d = self.getJensenShannonDistance(hbx_a,hbx_b)
        hwx_d = self.getJensenShannonDistance(hwx_a, hwx_b)
        hbwx_d = self.getJensenShannonDistance(hbwx_a, hbwx_b)
        hby_d = self.getJensenShannonDistance(hby_a, hby_b)
        hwy_d= self.getJensenShannonDistance(hwy_a, hwy_b)
        hbwy_d = self.getJensenShannonDistance(hbwy_a, hbwy_b)

        mean_d=1.0*np.mean([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d])
        return mean_d


    def get_weighted_JensenShannon(self, histA, histB): #combination of different features
        histogram_types_num=12
        distances = 0
        arr_a=np.array(histA)
        rl_a=arr_a[0:48]
        cc_a=arr_a[48:]
        arr_b = np.array(histB)
        rl_b = arr_b[0:48]
        cc_b = arr_b[48:]
        rl_distance=self.get_average_of_JensenShannon(rl_a,rl_b) #consists @8 bins of rlbxh,rlwxh,rlbwxh,rlbyh,rlwyh,rlbwyh
        cc_distance=self.getJensenShannonDistance(cc_a,cc_b)  #consists 64 bin of ccxyh
        min_weight=1.0*np.mean([rl_distance,cc_distance])
        rl_weight=0.5
        cc_weight=0.5
        # return 1.0*((rl_weight*rl_distance+cc_weight*cc_distance)/(rl_weight+cc_weight))
        return min_weight


    def get_ccxyh_distance_by_JensenShannon(self, histA, histB):
        histogram_types_num=12
        distances = 0
        arr_a=np.array(histA)
        arr_b = np.array(histB)
        ccxyh_a=arr_a[0:64]
        ccxyh_b = arr_b[0:64]
        ccxyh_d = self.getJensenShannonDistance(ccxyh_a,ccxyh_b)
        return ccxyh_d

    def getJensenShannonDistance(self, P, Q):
        # print "test-white:%s--train_white:%s" %(str(P),str(Q))
        _P = P / norm(P, ord=1)
        _Q = Q / norm(Q, ord=1)
        _M = 0.5 * (_P + _Q)
        return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def read_histogram_csv(self, CSVFilePath):
        print "Reading CSV...."
        #columns = ["bx", "wx", "bwx", "by", "wy", "bwy", "bm", "wm", "bwm","bs","ws","bws","class"]
        df = pd.read_csv(CSVFilePath, sep=",", header=None)
        X=np.array(df.ix[:,0:df.shape[1]-2])
        y = np.array(df.ix[:,df.shape[1]-1])
        return X,y

    def classifyKNN_by_rlbwxyh(self,
                               CSVFilePath,
                               class_number=16,
                               n_splits=5,
                               nearest_neighbors=1,
                               class_dict={},
                               result_csv="a.csv",
                               resume_test_idx=0):
        datas,labels=self.read_histogram_csv(CSVFilePath)
        datas=datas[:20000]
        labels=labels[:20000]
        knn2=KNeighborsClassifier(n_neighbors=nearest_neighbors,
                                  algorithm="ball_tree",
                                  metric=self.get_average_of_JensenShannon)
        accuracies=[]
        median_acc=0.0
        avg_acc=0.0
        hist=[]
        tests=[]
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        # n_split=class_number
        kfold = KFold(n_splits=n_splits)   #same number as #class: "text","math","logo","ta1ble","drawing","halftone","ruling"
        true = 0
        false = 0
        test_inc = 1
        train_time=[]
        test_time=[]
        with open(result_csv, "wb") as f:
            pass
        for train_idx, test_idx in kfold.split(labels):
            train_datas = datas[train_idx[:]]
            train_labels = labels[train_idx[:]]
            test_datas= datas[test_idx[:]]
            test_labels = labels[test_idx[:]]
            #print "shape train=(%d,%d),test(%d,%d)"%(train_datas.shape[0],train_datas.shape[1],test_datas.shape[0],test_datas.shape[1])

            print "length TRAIN#%d:(%d)" % (test_inc,len(train_datas))
            print "length TEST#%d:(%d)" % (test_inc,len(test_datas))
            #start=time.time()
            knn2.fit(train_datas, train_labels)
            # end=time.time()
            # traintime=(end-start)
            # #print "TRAIN TIME:%s" %str(train_time)
            # train_time.append(traintime)
            preds = np.array((1,), dtype=str)
            gt = []
            predictions=[]
            test_len=len(test_datas)
            #tmp_res=[]
            for idx in range(0,test_len):
                if idx<resume_test_idx:
                    continue
                test=test_datas[idx].reshape(1, test_datas.shape[1])
                #start=time.time()
                pred = knn2.predict(test)
                #test2=np.array(test)
                gt.append(test_labels[idx])
                predictions.append(pred.tolist())
                #tmp_res.append([test_labels[idx],])
                # end=time.time()
                # testtime=(end-start)
                # exe_times+=testtime
                # test_time.append(testtime)
                #print "AVG EXE TIME:%.5f" % np.mean(test_time)

                if (pred == test_labels[idx]):
                    true = true + 1
                    print "\rTEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
                else:
                    false = false + 1
                    print "\rTEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
                pred = np.array(pred)
                preds = np.vstack((preds, pred))
                print "\r(%d/%d)true:%d, false:%d, Accuracy:%f" % (
                    idx, test_len, true, false, (1.0 * true / (true + false)))

                batch=idx+1
                if batch%100==0:
                    with open(result_csv, "a+") as f:
                        w = csv.writer(f)
                        gt_=gt[-100:]
                        predictions_=predictions[-100:]
                        for ii in range (100):
                            w.writerow([gt_[ii], predictions_[ii]])
                    #tmp_res=[]
                #tmp_res.append()
                #break
            # with open(result_csv, "a+") as f:
            #     w = csv.writer(f)
            #     for i in range(len(test_labels)):
            #         w.writerow([test_labels[i], preds[i,0]])
                # head,file=os.path.split(result_csv)
                # time_fn=str(file).replace(".csv","_time.csv")
                # time_csv=os.path.join(head,time_fn)accuracy=accuracy_score(GT, prediction)
            accuracy = accuracy_score(gt, predictions)
            precision=precision_score(gt, predictions, average="macro")
            recall=recall_score(gt, predictions, average="macro")
            f1measure=f1_score(gt, predictions, average="macro")
            print "ACCURACY: %.5f"%accuracy
            print "PRECISION: %.5f"%precision
            print "RECALL: %.5f"%recall
            print "F1Measure: %.5f"%f1measure
                # with open(time_csv, "a+") as f:
                #     w = csv.writer(f)
                #     for t in test_time:
                #         w.writerow([t])

            acc = true / (true + false)
            accuracies.append(acc)
            print "CROSSVAL(%d/%d)------>true:%d, false:%d, Accuracy:%f" % (
            test_inc, n_splits, true, false, acc)
            preds = np.delete(preds, 0)
            t=my_tool()
            t.print_confusion_matrix(test_labels,preds,class_dict=class_dict)
            test_inc = test_inc + 1
            gtpred=zip(test_labels,preds)
            np.savetxt(result_csv,gtpred,fmt="%s",delimiter=" ")

            break
        median_acc=median(accuracies)
        avg_acc=sum(accuracies)/len(accuracies)
        # avg_train_time=sum(train_time)/len(train_time)
        # avg_test_time = sum(test_time) / len(test_time)
        print "MEDIAN ACCURACY:%.2f"%median_acc
        print "AVERAGE ACCURACY:%.2f" % avg_acc
        # print "AVERAGE TRAIN TIME:%.2f"%avg_train_time
        # print "AVERAGE TEST TIME:%.2f" % avg_test_time




    def classifyKNN_by_ccxyh(self, CSVFilePath):
        datas,labels=self.read_histogram_csv(CSVFilePath)
        knn2=KNeighborsClassifier(n_neighbors=1,algorithm="ball_tree",metric=self.get_ccxyh_distance_by_JensenShannon)
        hist=[]
        tests=[]
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        n_split=7
        kfold = KFold(n_splits=n_split)   #same number as #class: "text","math","logo","table","drawing","halftone","ruling"
        true = 0
        false = 0
        test_inc = 1
        for train_idx, test_idx in kfold.split(labels):
            train_datas = datas[train_idx]
            train_labels = labels[train_idx]
            test_datas= datas[test_idx]
            test_labels = labels[test_idx]
            #print "shape train=(%d,%d),test(%d,%d)"%(train_datas.shape[0],train_datas.shape[1],test_datas.shape[0],test_datas.shape[1])

            print "length TRAIN(%d)" % (len(train_datas))
            print "length TEST(%d)" % (len(test_datas))
            knn2.fit(train_datas, train_labels)
            for idx in range(0,len(test_datas)):
                test=test_datas[idx].reshape(1, test_datas.shape[1])
                pred = knn2.predict(test)
                if (pred == test_labels[idx]):
                    true = true + 1
                    print "TEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
                else:
                    false = false + 1
                    print "TEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
            print "(%d/%d)true:%d, false:%d, Accuracy:%f" % (test_inc,n_split, true, false, (1.0 * true / (true + false)))
            test_inc = test_inc + 1
            break



    def classifyKNN_by_rlbwxyh2(self, CSVFilePath,test_percentage=0.2):
        datas,labels=self.read_histogram_csv(CSVFilePath)
        knn2=KNeighborsClassifier(n_neighbors=1,algorithm="ball_tree",metric=self.get_average_of_JensenShannon)
        hist=[]
        tests=[]
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        n_split=7
        kfold = KFold(n_splits=n_split)   #same number as #class: "text","math","logo","ta1ble","drawing","halftone","ruling"
        true = 0
        false = 0
        test_inc = 1
        for train_datas, test_datas,train_labels, test_labels in train_test_split(datas,labels,test_size=test_percentage,random_state=16,stratify=None):
            print "length TRAIN(%d)" % (len(train_datas))
            print "length TEST(%d)" % (len(test_datas))
            knn2.fit(train_datas, train_labels)
            preds = np.array((1,), dtype=str)
            for idx in range(0,len(test_datas)):
                test=test_datas[idx].reshape(1, test_datas.shape[1])
                pred = knn2.predict(test)
                if (pred == test_labels[idx]):
                    true = true + 1
                    print "TEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
                else:
                    false = false + 1
                    print "TEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
                pred = np.array(pred)
                preds = np.vstack((preds, pred))
            print "(%d/%d)true:%d, false:%d, Accuracy:%f" % (
            test_inc, n_split, true, false, (1.0 * true / (true + false)))
            preds = np.delete(preds, 0)
            t=my_tool()
            t.print_confusion_matrix(test_labels,preds)
            test_inc = test_inc + 1



    def classifyKNN_by_weighted_distance(self, CSVFilePaths=[]):
        datas_list=[]
        labels_list=[]
        tool=my_tool()
        fuse_feature=tool.fuse_features(CSVFilePaths)
        datas=fuse_feature[:,0:fuse_feature.shape[1]-1]
        labels=fuse_feature[:,-1]
        knn2 = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree",
                                    metric=self.get_weighted_JensenShannon)
        hist = []
        tests = []
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        n_split = 7
        kfold = KFold(
            n_splits=n_split)  # same number as #class: "text","math","logo","table","drawing","halftone","ruling"
        true = 0
        false = 0
        test_inc = 1
        for train_idx, test_idx in kfold.split(labels):
            train_datas = datas[train_idx]
            train_labels = labels[train_idx]
            test_datas = datas[test_idx]
            test_labels = labels[test_idx]
            # print "shape train=(%d,%d),test(%d,%d)"%(train_datas.shape[0],train_datas.shape[1],test_datas.shape[0],test_datas.shape[1])

            print "length TRAIN(%d)" % (len(train_datas))
            print "length TEST(%d)" % (len(test_datas))
            knn2.fit(train_datas, train_labels)
            for idx in range(0, len(test_datas)):
                test = test_datas[idx].reshape(1, test_datas.shape[1])
                pred = knn2.predict(test)
                if (pred == test_labels[idx]):
                    true = true + 1
                    print "TEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (
                    idx + 1, len(test_datas), pred, test_labels[idx])
                else:
                    false = false + 1
                    print "TEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (
                    idx + 1, len(test_datas), pred, test_labels[idx])
            print "(%d/%d)true:%d, false:%d, Accuracy:%f" % (
            test_inc, n_split, true, false, (1.0 * true / (true + false)))
            test_inc = test_inc + 1
            break

    # def seek_optimum_knn(self, CSVFilePaths=[]):
    #     k=[i for i in range(0,50)]
    #     k=filter(lambda (x):x%2!=0,k) #find odd k between 1 to 50
    #     datas_list=[]
    #     labels_list=[]
    #     tool=my_tool()
    #     fuse_feature=tool.fuse_features(CSVFilePaths)
    #     datas=fuse_feature[:,0:fuse_feature.shape[1]-1]
    #     labels=fuse_feature[:,-1]
    #     knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree",metric=self.get_weighted_JensenShannon)
    #     hist = []
    #     tests = []
    #     n_split = 7
    #     kfold = KFold(
    #         n_splits=n_split)  # same number as #class: "text","math","logo","table","drawing","halftone","ruling"
    #     true = 0
    #     false = 0
    #     test_inc = 1
    #     for train_idx, test_idx in kfold.split(labels):
    #         train_datas = datas[train_idx]
    #         train_labels = labels[train_idx]
    #         test_datas = datas[test_idx]
    #         test_labels = labels[test_idx]
    #         # print "shape train=(%d,%d),test(%d,%d)"%(train_datas.shape[0],train_datas.shape[1],test_datas.shape[0],test_datas.shape[1])
    #
    #         print "length TRAIN(%d)" % (len(train_datas))
    #         print "length TEST(%d)" % (len(test_datas))
    #         knn2.fit(train_datas, train_labels)
    #         for idx in range(0, len(test_datas)):
    #             test = test_datas[idx].reshape(1, test_datas.shape[1])
    #             pred = knn2.predict(test)
    #             if (pred == test_labels[idx]):
    #                 true = true + 1
    #                 print "TEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (
    #                 idx + 1, len(test_datas), pred, test_labels[idx])
    #             else:
    #                 false = false + 1
    #                 print "TEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (
    #                 idx + 1, len(test_datas), pred, test_labels[idx])
    #         print "(%d/%d)true:%d, false:%d, Accuracy:%f" % (
    #         test_inc, n_split, true, false, (1.0 * true / (true + false)))
    #         test_inc = test_inc + 1
    #         break

    def classifyKNN_by_rlbwxyh_by_train_val_csv(self,
                               train_csv,
                               test_csv,
                               class_number=16,
                               #n_splits=5,
                               nearest_neighbors=1,
                               class_dict={},
                               result_csv="a.csv",
                               times_csv="a.csv",
                               resume_test_idx=0,
                                                save_batch=100):
       # datas,labels=self.read_histogram_csv(train_csv)
        knn2=KNeighborsClassifier(n_neighbors=nearest_neighbors,
                                  algorithm="ball_tree",
                                  metric=self.get_average_of_JensenShannon)
        with open(result_csv,"a+") as f:
           pass
        #times_csv=os.path.join(os.path.split(result_csv)[0],"times.csv")
        with open(times_csv,"a+") as f:
           pass
        accuracies=[]
        median_acc=0.0
        avg_acc=0.0
        hist=[]
        tests=[]
        # loo=LeaveOneOut()
        # #kfold=KFold(n_splits=50)
        # n_split=class_number
        #kfold = KFold(n_splits=n_splits)   #same number as #class: "text","math","logo","ta1ble","drawing","halftone","ruling"
        true = 0
        false = 0
        test_inc = 1
        train_time=[]
        test_time=[]
        #
        # for train_idx, test_idx in kfold.split(labels):
        #     train_datas = datas[train_idx[:]]
        #     train_labels = labels[train_idx[:]]
        #     test_datas= datas[test_idx[:]]
        #     test_labels = labels[test_idx[:]]
        #     #print "shape train=(%d,%d),test(%d,%d)"%(train_datas.shape[0],train_datas.shape[1],test_datas.shape[0],test_datas.shape[1])
        #
        #     print "length TRAIN#%d:(%d)" % (test_inc,len(train_datas))
        #     print "length TEST#%d:(%d)" % (test_inc,len(test_datas))
            #start=time.time()
        rev_dct={v:k for k,v in class_dict.items()}
        dftrain=pd.read_csv(train_csv,sep=",",header=None)
        npdtrain=np.array(dftrain)
        train_datas=npdtrain[:,:-2]
        train_labels=npdtrain[:,-1]
        train_labels=np.array([class_dict[a] for a in train_labels])
        #train_labels=[rev_dct[k] for k in train_labels]
        print "FITTING DATA ....."
        start=time.time()
        knn2.fit(train_datas, train_labels)
        end=time.time()
        elapsed=(end-start)
        print "FITTING TIME=%f" %elapsed
        # end=time.time()
        # traintime=(end-start)
        # #print "TRAIN TIME:%s" %str(train_time)
        # train_time.append(traintime)


        dftest = pd.read_csv(test_csv, sep=",", header=None)
        npdtest = np.array(dftest)
        test_datas = npdtest[:,: -2]
        test_labels = npdtest[:, -1]

        test_labels = np.array([class_dict[a] for a in test_labels])
        #test_labels=[rev_dct[k] for k in test_labels]
        preds = np.array((1,), dtype=str)

        labels = np.array((1,), dtype=str)
        test_len=len(test_datas)
        for idx in range(0,test_len):
            if idx<resume_test_idx:
                continue

            test=np.reshape(test_datas[idx],[1, test_datas.shape[1]])
            #start=time.time()
            start=time.time()
            pred = knn2.predict(test)
            end=time.time()
            elapsed=(end-start)
            print "PREDICT TIME=%f" %elapsed
            test_time.append(elapsed)
            # end=time.time()
            # testtime=(end-start)
            # #print "TEST TIME:%s" %testtime
            # test_time.append(testtime)
            gt_label=test_labels[idx]
            if (pred == gt_label):
                true = true + 1
                print "\rTEST idx:%d/%d:TRUE  pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
            else:
                false = false + 1
                print "\rTEST idx:%d/%d:FALSE   pred:%s ---- gt:%s" % (idx + 1, len(test_datas), pred, test_labels[idx])
            l = np.array(gt_label)
            labels = np.vstack((labels, l))
            pred = np.array(pred)
            preds = np.vstack((preds, pred))
            print "\r(%d/%d)true:%d, false:%d, Accuracy:%f" % (
                idx, test_len, true, false, (1.0 * true / (true + false)))
            # with open(result_csv, "a+") as f:
            #     w = csv.writer(f)
            #
            #     w.writerow([test_labels[idx], pred])
            #break
            if (idx % save_batch == 0) and (idx !=0) and len(preds)>save_batch:
                with open(result_csv, "a+") as f:

                    w = csv.writer(f)
                    gt_ = labels[-save_batch:]
                    predictions_ = preds[-save_batch:]
                    for ii in range(save_batch):
                        w.writerow([gt_[ii,0], predictions_[ii,0]])
                    print ">>>>>>>>>>>>>>>>>>>>>>>>>>WRITTEN TO CSV."
                with open(times_csv, "a+") as f:
                    w = csv.writer(f)
                    tt = test_time[-save_batch:]
                    for ii in range(save_batch):
                        w.writerow([tt[ii]])
        acc = true / (true + false)
        accuracies.append(acc)
        print "Epoch(%d)------>true:%d, false:%d, Accuracy:%f" % (
        test_inc, true, false, acc)
        preds = np.delete(preds, 0)
        t=my_tool()

        # with open(result_csv, "a+") as f:
        #     w = csv.writer(f)
        #     w.writerow([test_labels, preds])
        t.print_confusion_matrix(test_labels,preds,class_dict=class_dict)
        test_inc = test_inc + 1
        gtpred=zip(test_labels,preds)
        #np.savetxt(result_csv,gtpred,fmt="%s",delimiter=" ")


            #break
        median_acc=median(accuracies)
        avg_acc=sum(accuracies)/len(accuracies)
        # avg_train_time=sum(train_time)/len(train_time)
        # avg_test_time = sum(test_time) / len(test_time)
        print "MEDIAN ACCURACY:%.2f"%median_acc
        print "AVERAGE ACCURACY:%.2f" % avg_acc
        # print "AVERAGE TRAIN TIME:%.2f"%avg_train_time
        # print "AVERAGE TEST TIME:%.2f" % avg_test_time

    # def get_average_of_JensenShannon_using_tensorflow(self, histA, histB):
    #     histogram_types_num=12
    #     distances = 0
    #     arr_a=np.array(histA)
    #     arr_b = np.array(histB)
    #     hbx_a=tf.Variable(arr_a[0:8],dtype=tf.float32,trainable=False,name="hbx_a",)
    #     hwx_a = tf.Variable(arr_a[8:16],dtype=tf.float32,trainable=False,name="hwx_a")
    #     hbwx_a = tf.Variable(arr_a[16:24],dtype=tf.float32,trainable=False,name="hbwx_a")
    #     hby_a = tf.Variable(arr_a[24:32],dtype=tf.float32,trainable=False,name="hby_a")
    #     hwy_a = tf.Variable(arr_a[32:40],dtype=tf.float32,trainable=False,name="hwy_a")
    #     hbwy_a = tf.Variable(arr_a[40:48],dtype=tf.float32,trainable=False,name="hbwy_a")
    #
    #     hbx_b = tf.Variable(arr_b[0:8],dtype=tf.float32,trainable=False,name="hbx_b")
    #     hwx_b = tf.Variable(arr_b[8:16],dtype=tf.float32,trainable=False,name="hwx_b")
    #     hbwx_b = tf.Variable(arr_b[16:24],dtype=tf.float32,trainable=False,name="hbwx_b")
    #     hby_b = tf.Variable(arr_b[24:32],dtype=tf.float32,trainable=False,name="hby_b")
    #     hwy_b = tf.Variable(arr_b[32:40],dtype=tf.float32,trainable=False,name="hwy_b")
    #     hbwy_b = tf.Variable(arr_b[40:48],dtype=tf.float32,trainable=False,name="hbwy_b")
    #
    #     hbx_d = self.getJensenShannonDistance_using_tensorflow(hbx_a,hbx_b)
    #     hwx_d = self.getJensenShannonDistance_using_tensorflow(hwx_a, hwx_b)
    #     hbwx_d = self.getJensenShannonDistance_using_tensorflow(hbwx_a, hbwx_b)
    #     hby_d = self.getJensenShannonDistance_using_tensorflow(hby_a, hby_b)
    #     hwy_d= self.getJensenShannonDistance_using_tensorflow(hwy_a, hwy_b)
    #     hbwy_d = self.getJensenShannonDistance_using_tensorflow(hbwy_a, hbwy_b)
    #
    #     mean_d=1.0*np.mean([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d])
    #     arr_result=tf.Variable([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d],trainable=False,dtype=tf.float32)
    #     mean_d=tf.reduce_mean(arr_result)
    #     return mean_d
    #
    # def getJensenShannonDistance_using_tensorflow(self, P, Q):
    #     # print "test-white:%s--train_white:%s" %(str(P),str(Q))
    #     half=tf.Constant(0.5,dtype=tf.float32)
    #     _P = tf.div(P,tf.norm(P,ord=1))#P / norm(P, ord=1)
    #     _Q = tf.div(Q,tf.norm(Q,ord=1))#Q / norm(Q, ord=1)
    #     _M = tf.multiply (half, (tf.add(_P,_Q)))
    #
    #     ent1=tf.add_n(tf.multiply(_P,(tf.log(tf.div(_P,_M)))))
    #     ent2 = tf.add_n(tf.multiply(_Q, (tf.log(tf.div(_Q, _M)))))
    #     ent_add=tf.add(ent1,ent2)
    #     ret=tf.multiply(half,ent_add)
    #     return ret#0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def get_average_of_JensenShannon_using_tensorflow(self,batch, Xtrains, Xtest):
        histogram_types_num=12
        distances = []
        npd_ha_shape=np.shape(Xtrains)
        npd_hb_shape = np.shape(Xtest)
        i=0
        with tf.variable_scope("jsd",reuse=tf.AUTO_REUSE):
            arr_aa=tf.get_variable("histA",dtype=tf.float32,trainable=False,shape=[npd_ha_shape[0],npd_ha_shape[1]])#np.array(histA)
            arr_aa=tf.assign(arr_aa,Xtrains)
            arr_b = tf.get_variable("histB",dtype=tf.float32,trainable=False,shape=[npd_hb_shape[0]])#,npd_hb_shape[1]])#np.array(histB)
            arr_b=tf.assign(arr_b,Xtest)

            #for arr_b in tf.unstack(arr_bb):
            for arr_a in tf.unstack(arr_aa):
                #print "CALCULATE DISTANCE TO NEIGHBOIRS of TEST %d"%i
                with tf.variable_scope("rlbwxyh",reuse=tf.AUTO_REUSE):
                    arr_a=tf.reshape(arr_a,shape=[48])
                    hbx_a=tf.get_variable(name="hbx_a",dtype=tf.float32,trainable=False,shape=[8])#shape=[1,8])#
                    hbx_a=tf.assign(hbx_a,arr_a[0:8],validate_shape=False)
                    hwx_a = tf.get_variable(name="hwx_a", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hwx_a = tf.assign(hwx_a, arr_a[8:16], validate_shape=False)
                    hbwx_a = tf.get_variable(name="hbwx_a", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hbwx_a = tf.assign(hbwx_a, arr_a[16:24], validate_shape=False)
                    hby_a = tf.get_variable(name="hby_a", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hby_a = tf.assign(hby_a, arr_a[24:32], validate_shape=False)
                    hwy_a = tf.get_variable(name="hwy_a", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hwy_a = tf.assign(hwy_a, arr_a[32:40], validate_shape=False)
                    hbwy_a = tf.get_variable(name="hbwy_a", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hbwy_a = tf.assign(hbwy_a, arr_a[40:48], validate_shape=False)

                    hbx_b = tf.get_variable(name="hbx_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hbx_b = tf.assign(hbx_b, arr_b[0:8], validate_shape=False)
                    hwx_b = tf.get_variable(name="hwx_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hwx_b = tf.assign(hwx_b, arr_b[8:16], validate_shape=False)
                    hbwx_b = tf.get_variable(name="hbwx_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hbwx_b = tf.assign(hbwx_b, arr_b[16:24], validate_shape=False)
                    hby_b = tf.get_variable(name="hby_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hby_b = tf.assign(hby_b, arr_b[24:32], validate_shape=False)
                    hwy_b = tf.get_variable(name="hwy_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hwy_b = tf.assign(hwy_b, arr_b[32:40], validate_shape=False)
                    hbwy_b = tf.get_variable(name="hbwy_b", dtype=tf.float32, trainable=False, shape=[8])#shape=[1,8])
                    hbwy_b = tf.assign(hbwy_b, arr_b[40:48], validate_shape=False)

                    hbx_d = self.getJensenShannonDistance_using_tensorflow(hbx_a,hbx_b)
                    hwx_d = self.getJensenShannonDistance_using_tensorflow(hwx_a, hwx_b)
                    hbwx_d = self.getJensenShannonDistance_using_tensorflow(hbwx_a, hbwx_b)
                    hby_d = self.getJensenShannonDistance_using_tensorflow(hby_a, hby_b)
                    hwy_d= self.getJensenShannonDistance_using_tensorflow(hwy_a, hwy_b)
                    hbwy_d = self.getJensenShannonDistance_using_tensorflow(hbwy_a, hbwy_b)

                    #mean_d=1.0*np.mean([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d])
                    arr_result=tf.get_variable(name="rlbwxyh_result", dtype=tf.float32, trainable=False, shape=[6])
                    arr_result=tf.assign(arr_result,tf.reshape([hbx_d,hwx_d,hbwx_d,hby_d,hwy_d,hbwy_d],[6]))
                    mean_d=tf.reduce_mean(arr_result)
                    distances.append(mean_d)

                    i+=1
        return distances#mean_d

    def getJensenShannonDistance_using_tensorflow(self, P, Q):
        # print "test-white:%s--train_white:%s" %(str(P),str(Q))
        half=tf.constant(0.5,dtype=tf.float32)
        _P = tf.div(P,tf.norm(P,ord=1))#P / norm(P, ord=1)
        _Q = tf.div(Q,tf.norm(Q,ord=1))#Q / norm(Q, ord=1)
        _M = tf.multiply (half, (tf.add(_P,_Q)))

        ent1=tf.add_n([tf.multiply(_P,(tf.log(tf.div(_P,_M))))])
        ent2 = tf.add_n([tf.multiply(_Q, (tf.log(tf.div(_Q, _M))))])
        ent_add=tf.add(ent1,ent2)
        ret=tf.multiply(half,ent_add)
        ret = tf.norm(ret, ord='euclidean')
        ret=tf.reshape(ret,[1])
        return ret#0.5 * (entropy(_P, _M) + entropy(_Q, _M))

    def classify_by_knn_tensorflow(self,csv_train,csv_test,k,batch,result_csv):#X_t, y_t, x_t, k_t):
        dftrain=pd.read_csv(csv_train,sep=",",header=None)
        dftest = pd.read_csv(csv_test, sep=",", header=None)
        npdtrain=np.array(dftrain)#[:10])
        npdtest = np.array(dftest)#[:5])
        X_t=npdtrain[:,:-1]
        y_t = npdtrain[:, -1]
        x_t=npdtest[:,:-1]
        gt_label=npdtest[:,-1]
        k_t=k
        X_t_shape=np.shape(X_t)
        x_t_shape = np.shape(x_t)
        neg_one = tf.constant(-1.0, dtype=tf.float32)
        self.Xtrain_ph=tf.placeholder(dtype=tf.float32,shape=[X_t_shape[0], 48],name="Xtrain")#X_t_shape[1]
        self.Xtest_ph = tf.placeholder(dtype=tf.float32, shape=[48], name="Xtest")#x_t_shape[1]
        self.save_batch = tf.placeholder(dtype=tf.int32, name="save_batch")
        self.k_neighbor = tf.placeholder(dtype=tf.int32, name="k_neighbor")
        with open(result_csv,"a+") as f:
            pass
        results=[]
        times=[]
        distances = self.get_average_of_JensenShannon_using_tensorflow(
            batch=self.save_batch,
            Xtrains=self.Xtrain_ph,
            Xtest=self.Xtest_ph)
        # to find the nearest points, we find the farthest points based on negative distances
        # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
        neg_distances = tf.multiply(distances, neg_one)
        # get the indices
        vals, indx = tf.nn.top_k(neg_distances, self.k_neighbor)
        # slice the labels of these points
        y_s = tf.gather(y_t, indx)
        # we compute the L-1 distance
        #distances = tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
        for i in range(len(x_t)):
            x_test=x_t[i]
            gtlabel=gt_label[i]
            config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                #x_test=tf.reshape(x_test,shape=[48])
                begin=time.time()
                out = sess.run(y_s,feed_dict={self.Xtrain_ph:X_t,
                                              self.Xtest_ph:x_test,
                                              self.save_batch:batch,
                                              self.k_neighbor:k_t})
                end=time.time()
                elapsed=end-begin
                print "TIME=%f" %(elapsed)
                print "GT/predict = %s/%s %r" %(gtlabel,out,gtlabel==out)
                results.append([gtlabel,out[0]])
                times.append(elapsed)
                #print self.get_label(out)
                # config = tf.ConfigProto(allow_soft_placement=True)
                # with tf.Session(config=config) as sess:
                #     out=sess.run(y_s)
                #     print self.get_label(out)
        tool=victorinox()
        npres=np.array(results)
        tool.calculate_f1_measure(npres[:,0],npres[:,-1])
        head,tail=os.path.split(result_csv)
        times_csv=os.path.join(head,"times.csv")
        if len(results)>0:
            np.savetxt(result_csv,results,fmt="%s",delimiter=",")
        if len(times) > 0:
            np.savetxt(times_csv, times, fmt="%s", delimiter=",")
        return #y_s
        #return

    def get_label(self, preds):
        counts = np.bincount(preds.astype('int64'))
        return np.argmax(counts)