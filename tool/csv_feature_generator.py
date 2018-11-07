import sys
import csv
from RLXYMSH import RLXYMSH
import os
import re
import numpy as np

class csv_feature_generator(object):
    def __init__(self):
        #self.files=[]
        return

    def convertImageToCSV(self,img_folder,csv_path):
        try:
            #histogram_list=self.convert_images_to_histogram_list(img_folder)
            histogram_dictionary = self.convert_zone_to_histogram_list(img_folder)
            print "dictionary:%s" %str(histogram_dictionary)
            histogram_csv=[]
            histogram_csv=self.iterate_files(csv_path,histogram_csv)
            for hc in histogram_csv:
                head, tail = os.path.split(hc)
                if histogram_dictionary.has_key(tail):
                    rows=histogram_dictionary[tail]
                    with open(hc,"wb") as dest_file:
                        print "writing histogram:%s" %(tail)
                        writer=csv.writer(dest_file)
                        #writer.writerow(["hbx","hwx","hbwx","hby","hwy","hbwy","hbm","hwm","hbwm","hbs","hws","hbws", "class"])
                        #writer.writerow(["rlbxh","rlwxh","rlbwxh","class"])
                        for row in rows:
                            writer.writerow(row)
                        dest_file.close()
        except IOError,e:
            print "ERROR converting to CSV:"+str(e)

    # def convert_images_to_histogram_list(self,path):
    #     my_list=[]
    #     histogram_maker=RLXYMSH()
    #     try:
    #         images=os.listdir(path)
    #         p=re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
    #         for img in images:
    #             file_name = img.split(".")[0]
    #             file_name = p.findall(file_name)[0]
    #             img=os.path.join(path,img)
    #             hb,hw,hbw=histogram_maker.get_rlbwxh(img)
    #             row=[hb[0],hw[0],hbw[0],str(file_name)]
    #             my_list.append(row)
    #             print str(file_name)
    #             #break
    #     except Exception,e:
    #         print str(e)
    #     return my_list

    def convert_zone_to_histogram_list(self,path):
        nonTextClass=["math","logo","table","drawing","halftone","ruling"]
        textClass=['text_text-body', 'text_reference-list-item', 'text_section-heading', 'text_page-number', 'text_caption',
         'text_list-item', 'text_page-header', 'text_page-footer', 'text_abstract-body', 'text_title', 'text_footnote',
         'text_affiliation', 'text_reference-heading', 'text_abstract-heading', 'text_not-clear', 'text_biography',
         'text_keyword-body', 'text_article-submission-information', 'text_list', 'text_keyword-heading',
         'text_reference-list', 'text_drop-cap', 'text_definition', 'text_membership', 'text_synopsis',
         'text_keyword_heading_and_body']
        nonTextFound=[]
        textFound=[]
        my_list=[]
        hbx_arr=[]
        hwx_arr=[]
        hbwx_arr=[]
        hby_arr=[]
        hwy_arr=[]
        hbwy_arr=[]
        hbm_arr=[]
        hwm_arr=[]
        hbwm_arr=[]
        hbs_arr=[]
        hws_arr=[]
        hbws_arr=[]
        histogram_maker=RLXYMSH()
        try:
            images=[]
            images=self.iterate_files(path,images)    #get all files list withon folders
            #print "images:%s"%str(images)
            p=re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
            for img in images:
                print "processing image:%s" % str(img)
                head,tail=os.path.split(img)
                file_name = tail.split(".")[0]
                file_name = p.findall(file_name)[0]
                if file_name.startswith("text_") or file_name.startswith("text-"):
                    file_name="text"
                    # if file_name not in textClass:
                    #     textFound.append(file_name)
                    #     continue
                else:
                    # file_name = "non-text"
                    file_name=file_name[0:file_name.index("_non-text")]
                    # if file_name not in nonTextClass:
                    #     nonTextFound.append(file_name)
                    #     continue
                # img=os.path.join(path,img)
                # runX,lengthX=histogram_maker.getRunLengthX(img)
                # runY, lengthY = histogram_maker.getRunLengthY(img)
                # RunXY,LengthXY=histogram_maker.combineRunLength([runX,runY],[lengthX,lengthY])
                # histoXY=histogram_maker.get_histogram(LengthXY)
                # row= [histoXY[0],str(file_name)]
                hbx,hwx,hbwx=histogram_maker.get_rlbwxh(img)
                hby, hwy, hbwy = histogram_maker.get_rlbwyh(img)
                hbm, hwm, hbwm = histogram_maker.get_rlbwmh(img)
                hbs, hws, hbws = histogram_maker.get_rlbwsh(img)

                fn = [file_name]
                hbx_arr.append([i for i in hbx[0]]+fn) #get only histigram values, wothout bins
                hwx_arr.append([i for i in hwx[0]]+fn)
                hbwx_arr.append([i for i in hbwx[0]]+fn)
                hby_arr.append([i for i in hby[0]]+fn)
                hwy_arr.append([i for i in hwy[0]]+fn)
                hbwy_arr.append([i for i in hbwy[0]]+fn)
                hbm_arr.append([i for i in hbm[0]]+fn)
                hwm_arr.append([i for i in hwm[0]]+fn)
                hbwm_arr.append([i for i in hbwm[0]]+fn)
                hbs_arr.append([i for i in hbs[0]]+fn)
                hws_arr.append([i for i in hws[0]]+fn)
                hbws_arr.append([i for i in hbws[0]]+fn)

                #row=[hbx[0],hwx[0],hbwx[0],hby[0],hwy[0],hbwy[0],hbm[0],hwm[0],hbwm[0],hbs[0],hws[0],hbws[0],str(file_name)]
                #my_list.append(row)
                print str("converted "+file_name)
                #break
        except Exception,e:
            print str(e)
        #print "Non Text ignored(%d):%s " %(len(nonTextFound),set(nonTextFound))
        #print "Text ignored(%d):%s " % (len(textFound), set(textFound))
        #return my_list
        histogram_dictionary={}
        histogram_dictionary["rlbmh.csv"]=hbm_arr
        histogram_dictionary["rlbsh.csv"] = hbs_arr
        histogram_dictionary["rlbwmh.csv"] = hbwm_arr
        histogram_dictionary["rlbwsh.csv"] = hbws_arr
        histogram_dictionary["rlbwxh.csv"] = hbwx_arr
        histogram_dictionary["rlbwyh.csv"] = hbwy_arr
        histogram_dictionary["rlbxh.csv"] = hbx_arr
        histogram_dictionary["rlbyh.csv"] = hby_arr
        histogram_dictionary["rlwmh.csv"] = hwm_arr
        histogram_dictionary["rlwsh.csv"] = hws_arr
        histogram_dictionary["rlwxh.csv"] = hwx_arr
        histogram_dictionary["rlwyh.csv"] = hwy_arr
        print "histogram dictionary from fabrique:%s" %str(histogram_dictionary)
        return histogram_dictionary

    def iterate_files(self,folder,receiver):
        f=os.listdir(folder)
        for i in f:
            ff=os.path.join(folder,i)
            if os.path.isfile(ff):
                #print str(ff)
                receiver.append(ff)
            else:
                self.iterate_files(ff,receiver)
        return receiver