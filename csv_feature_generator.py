import sys
import csv
from RLXYMSH import RLXYMSH
import os
import re

class csv_feature_generator(object):
    def __init__(self):
        return

    def convertImageToCSV(self,img_folder,csv_path):
        try:
            #histogram_list=self.convert_images_to_histogram_list(img_folder)
            histogram_list = self.convertTextNonTextToHistogramList(img_folder)
            with open(csv_path,"wb") as dest_file:
                writer=csv.writer(dest_file)
                writer.writerow(["RLBWXYH", "class"])
                #writer.writerow(["rlbxh","rlwxh","rlbwxh","class"])
                for l in histogram_list:
                    writer.writerow(l)
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

    def convertTextNonTextToHistogramList(self,path):
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
        histogram_maker=RLXYMSH()
        try:
            images=os.listdir(path)
            p=re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
            for img in images:
                file_name = img.split(".")[0]
                file_name = p.findall(file_name)[0]
                if file_name.startswith("text_") or file_name.startswith("text-"):
                    file_name="text"
                    # if file_name not in textClass:
                    #     textFound.append(file_name)
                    #     continue
                else:
                    file_name = "non-text"
                    # file_name=file_name[0:file_name.index("_non-text")]
                    # if file_name not in nonTextClass:
                    #     nonTextFound.append(file_name)
                    #     continue
                img=os.path.join(path,img)
                runX,lengthX=histogram_maker.getRunLengthX(img)
                runY, lengthY = histogram_maker.getRunLengthY(img)
                RunXY,LengthXY=histogram_maker.combineRunLength([runX,runY],[lengthX,lengthY])
                histoXY=histogram_maker.get_histogram(LengthXY)
                row= [histoXY[0],str(file_name)]
                #hb,hw,hbw=histogram_maker.get_rlbwxh(img)
                #row=[hb[0],hw[0],hbw[0],str(file_name)]
                my_list.append(row)
                print str(file_name)
                #break
        except Exception,e:
            print str(e)
        print "Non Text ignored(%d):%s " %(len(nonTextFound),set(nonTextFound))
        print "Text ignored(%d):%s " % (len(textFound), set(textFound))
        return my_list

