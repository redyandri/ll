import numpy as np
import re
import os
import sys
from Jansen_Shannon import JSD
from RLXYMSH import *

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

    def compare_GT_Prediction(self,gt,predict):
        p=re.compile("[a-zA-Z]+")
        gt_class=p.findall(gt)
        predict_class=p.findall(predict)
        return gt_class==predict_class

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

    def get_nearest_neighbor(self,train_files,test_files):
        test_results=[]
        count=0
        for test in test_files:
            scores=[]
            for train in train_files:
                hb_test,hw_test,hbw_test=self.rlxymsh.get_rlbwxh(test)
                hb_train, hw_train, hbw_train = self.rlxymsh.get_rlbwxh(train)
                score=1.0*self.jsd.get_distance(hw_test[0],hw_train[0])
                scores.append(score)
            print "test:"+str(test)
            nearest=train_files[np.argmin(scores)]
            print "nearest:"+str(nearest)
            test_results.append(self.compare_GT_Prediction(test,nearest))
            if count == 29:
                print "results length:"+str(len(test_results))
                print "test_results:" + str(test_results)
                break
            count = count + 1


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



hc=Histogram_Classifier()
hc.classifyNearestNeighbor(sys.argv[1])
# t=hc.get_leave_one_out_fold(104,5)
# print "the dtaset split:"+str(t)+"\n of length:"+str(len(t))
# print hc.isTenOccurencesAtLeast('1-2-3-4_advertisement_non-text.tif')
# print hc.compare_GT_Prediction('1-2-3-4_advertisement_text.tif','11-22-33-44_advertisement_non-text.tif')