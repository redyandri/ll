import numpy as np
import re
import os
import sys
from Jansen_Shannon import JSD
from RLXYMSH import *
import csv
from collections import Counter

class Histogram_Classifier(object):
    jsd = JSD()
    rlxymsh = RLXYMSH()
    def __init__(self):
        return

    def isTenOccurencesAtLeast(self,filename):
        min_ten_occurances=['text_text-body','text_reference-list-item', 'text_section-heading',
                        'text_page-number','text-with-special-symbols_text-body', 'text_caption',
                        'text_list-item', 'text_page-header', 'math_non-text', 'drawing_non-text',
                        'text-with-special-symbols_list-item', 'text_page-footer', 'ruling_non-text',
                        'halftone_non-text', 'text_author', 'text_abstract-body', 'text_title',
                        'text_footnote', 'table_non-text', 'text_affiliation', 'text_reference-heading',
                        'text_abstract-heading', 'text_not-clear', 'text-with-special-symbols_page-footer',
                        'text_biography', 'text-with-special-symbols_caption', 'text_keyword-body',
                        'text-with-special-symbols_page-header', 'text-with-special-symbols_reference-list-item',
                        'text_article-submission-information', 'text-with-special-symbols_footnote',
                        'halftone-with-drawing_non-text', 'text_list', 'text_keyword-heading',
                        'text-with-special-symbols_list', 'text_reference-list', 'text_drop-cap',
                        'text-with-special-symbols_abstract-body', 'text-with-special-symbols_author',
                        'text-with-special-symbols_affiliation', 'text_definition', 'text_membership',
                        'text_synopsis',  'text-with-special-symbols_pseudo-code', 'map_non-text',
                        'logo_non-text', 'text-with-special-symbols_reference-list',
                        'text-with-special-symbols_biography','text_keyword_heading_and_body',
                        'advertisement_non-text']
        print "min 10 occurences:"+str(len(min_ten_occurances))
        p = re.compile("[a-zA-Z]+")
        filename=filename.split(".")[0] #get only name without ext
        fn_words=p.findall(filename)
        is_same=False
        for w in min_ten_occurances:
            words=p.findall(w)
            if fn_words == words:
                is_same=True
                break
        return is_same

    def compare_GT_Prediction(self,test,nearest):
        test_label=test[3]
        nearest_label=nearest[3]
        # p=re.compile("[a-zA-Z]+")
        # gt_class=p.findall(gt)
        # predict_class=p.findall(predict)
        return test_label==nearest_label

    def get_leave_one_out_fold(self,file_number,class_number):
        increment=0
        if file_number % class_number==0:
            increment=file_number/class_number
        else:
            increment = file_number / class_number +1
        tests=[]
        for i in range(0,file_number,increment):
            print "i:"+str(i)
            fold=[]
            start=i
            end=i+increment
            if(end > file_number):  # if end exceeds file number, make it equal to file number
                end = file_number
            fold.append(start)
            fold.append(end)
            tests.append(fold)


        return tests

    def get_nearest_neighbor(self,train_files,test_files,test_batch):
        test_results=[]
        count=1
        test_length=len(test_files)
        p=re.compile("\d+")
        for test in test_files:
            scores=[]
            for train in train_files:
                test_histo_white= [int(x) for x in p.findall(test[1])]
                test_label=test[3]
                train_histo_white = [int(x) for x in p.findall(train[1])]
                train_label=train[3]
                # hb_test,hw_test,hbw_test=self.rlxymsh.get_rlbwxh(test)
                # hb_train, hw_train, hbw_train = self.rlxymsh.get_rlbwxh(train)
                score=1
                try:
                    score=self.jsd.get_distance(test_histo_white,train_histo_white)
                except Exception,e:
                    print "Error on JSD:$s\ntest:$s---train:$s" %(str(e),str(test_histo_white),str(train_histo_white))
                scores.append(score)
            #print "test:"+str(test)
            least_score_index=np.argmin(scores)
            nearest=train_files[least_score_index]
            #print "nearest:"+str(nearest)
            test_results.append(self.compare_GT_Prediction(test,nearest))
            print "%d#(%d/%d)test - nearest:%s---%s" %(test_batch,count,test_length,test[3],nearest[3])#+str(test[3])+"-"+str(nearest[3]) %()
            count = count + 1
            #break

        return test_results

    def classifyNearestNeighbor(self,img_folder):
        dirs=os.listdir(img_folder)
        file_number=9987
        class_number=50
        tests=self.get_leave_one_out_fold(file_number,class_number)
        print "test conf:"+str(tests)
        test_files=[]
        train_files=[]
        test_results=[]
        for test in tests:
            start=test[0]
            print "start:"+str(start)
            end=test[1]
            print "end:"+str(end)
            counter=0
            for dir in dirs:
                dir=os.path.join(img_folder,dir)
                files =os.listdir(dir)
                for file in files:
                    file=os.path.join(dir,file)
                    if counter>=start and counter<end : #if test fold
                        test_files.append(file)
                        #print "add file to test:"+str(file)
                    else:
                        train_files.append(file)
                        #print "add file to train:"+str(file)
                        #break
                    counter = counter + 1
            print "test file length"+str(len(test_files))
            print "train file length" + str(len(train_files))
            test_result=self.get_nearest_neighbor(train_files,test_files)
            break


        increment_split=999 #cross validation contains 9x998 training samples,1x998 test
        return
    def get_true_false_count(self,test_results):
        c=Counter(test_results)
        true=c[True]
        false=c[False]
        print "True:%d,False:%d" %(true,false)
        return true, false

    def classify_1NN_with_csv_histogram(self,csv_path):
        #dirs=os.listdir(csv_path)
        file_number=9987
        class_number=50
        tests=self.get_leave_one_out_fold(file_number,class_number)
        #print "test conf:"+str(tests)
        test_files=[]
        train_files=[]
        test_results=[]
        test_inc=1
        total_true=0
        total_false=0
        for test in tests:
            test_files=[]       #empty test_files for new batch
            train_files=[]
            #print "TEST #"+str(test_inc)
            start=test[0]
            #print "start:"+str(start)
            end=test[1]
            #print "end:"+str(end)

            with open(csv_path,"rb") as histogram_csv:
                reader=csv.reader(histogram_csv)
                index = -1
                for line in reader:
                    if index==-1:   #meet header of csv
                        index = index + 1
                        continue
                    else:
                        # histo_black=line[0]
                        # histo_white = line[1]np.argmin(scores)
                        # histo_combi = linett[0],tt[1][2]
                        if index >= start and index < end:  # if test fold
                            test_files.append(line)
                            # print "add file to test:"+str(file)
                        else:
                            train_files.append(line)
                    index=index+1
                histogram_csv.close()
            #print "start:%d,end:%d,test-length:%d,train-length:%d" %(start,end,len(test_files),len(train_files))
            test_result=self.get_nearest_neighbor(train_files,test_files,test_inc)
            true,false=self.get_true_false_count(test_result)
            total_true=total_true+true
            total_false=total_false+false
            accuracy=1.0*true/(true+false)
            print "accuracy batch test#%d:%f" %(test_inc, accuracy)
            print "TOTAL TRUE:%d,TOTAL FALSE:%d,TOTAL DETECTION=%d,ACCURACY=%f" % (total_true, total_false, (total_true + total_false), 1.0 * total_true / (total_true + total_false))
            if test_inc==5:
                break
            test_inc = test_inc + 1
            #break

        return



hc=Histogram_Classifier()
hc.classify_1NN_with_csv_histogram(sys.argv[1])
# t=hc.get_leave_one_out_fold(104,5)
# print "the dtaset split:"+str(t)+"\n of length:"+str(len(t))
# print hc.isTenOccurencesAtLeast('1-2-3-4_advertisement_non-text.tif')
# print hc.compare_GT_Prediction('1-2-3-4_advertisement_text.tif','11-22-33-44_advertisement_non-text.tif')