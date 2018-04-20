import csv
import sys
from collections import Counter
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
import re

occurances=['12-12-12-12_text_text-body','text_reference-list-item', 'text_section-heading',
                        'text_page-number','text-with-special-symbols_text-body', 'text_caption',
                        'text_list-item', 'text_page-header', 'math_non-text', 'drawing_non-text',
                        'text-with-special-symbols_list-item', 'text_page-footer', 'ruling_non-text',
                        'halftone_non-text', 'tehttps://stackoverflow.com/questions/7419665/python-move-and-overwrite-files-and-foldersxt_author', 'text_abstract-body', 'text_title',
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

def get_leave_one_out_fold(file_number, class_number):
    increment = 0
    if file_number % class_number == 0:
        increment = file_number / class_number
    else:
        increment = file_number / class_number + 1
    tests = []
    for i in range(0, file_number, increment):
        #print "i:" + str(i)
        fold = []
        start = i
        end = i + increment
        if (end > file_number):  # if end exceeds file number, make it equal to file number
            end = file_number
        fold.append(start)
        fold.append(end)
        tests.append(fold)

    return tests

def get_distance(P, Q):
    #print "test-white:%s--train_white:%s" %(str(P),str(Q))
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


t=get_leave_one_out_fold(9987,50)
f=sys.argv[1]
arr=[]
with open(sys.argv[1],"rb") as csvv:
    reader=csv.reader(csvv)
    #print str(t)
    i=0

    for r in reader:
        arr.append(r)
    #print "arr length:%d" %(len(arr))
    p=re.compile("\d+")
    for tt in t:
        print "start:%d,end:%d,content:%s"%(tt[0],tt[1],arr[tt[0]:tt[1]])
        a=p.findall(arr[tt[0]][1])
        b=p.findall(arr[tt[1]][1])
        h1=[int(x) for x in a]
        h2 = [int(x) for x in b]
        print "%s   vs %s"%(str(h1),str(h2))
        print get_distance(h1,h2)
        if(i>0):
            break
        i=i+1
# class JSD(object):
#     def __init__(self):
#         return
#
#     def get_distance(self,P, Q):
#         _P = P / norm(P, ord=1)
#         _Q = Q / norm(Q, ord=1)
#         _M = 0.5 * (_P + _Q)
#         return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))
# path=sys.argv[1]
# j=JSD()
# a=[9,7,6,5,4,3,2,1]
# print a
# import re
# p=re.compile("\d+")
# import pandas as pd
#
# with open(path,"rb") as f:
#     dataframe=pd.read_csv(f,header=None,sep=",")
#     i=0
#     import re
#     p=re.compile("\d+")
#     for index,row in dataframe.iterrows():
#         if index==0:
#             continue
#         if index==5:
#             break
#         print row[1]
#         ints=p.findall(row[1])
#         newrow=[int(x) for x in ints]
#         score=j.get_distance(newrow,a)
#         print "score:%f" %(score)
#
#     f.close()



# from mnist import MNIST
# import random
# import re
#

#
# p=re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
# for o in occurances:
#     print str(p.findall(o))
#print f