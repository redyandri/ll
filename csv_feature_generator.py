import sys
import csv
from RLXYMSH import RLXYMSH
import os
import re

class csv_feature_generator(object):
    def __init__(self):
        return

    def convert_image_to_rlbwxymsh_csv(self,img_folder,csv_path):
        try:
            histogram_list=self.convert_images_to_histogram_list(img_folder)
            with open(csv_path,"wb") as dest_file:
                writer=csv.writer(dest_file)
                writer.writerow(["rlbxh","rlwxh","rlbwxh","class"])
                for l in histogram_list:
                    writer.writerow(l)
                dest_file.close()
        except IOError,e:
            print "ERROR converting to CSV:"+str(e)

    def convert_images_to_histogram_list(self,path):
        my_list=[]
        histogram_maker=RLXYMSH()
        try:
            images=os.listdir(path)
            p=re.compile("[a-zA-Z]+[a-zA-Z_\-]+[a-zA-Z]+")
            for img in images:
                file_name = img.split(".")[0]
                file_name = p.findall(file_name)[0]
                img=os.path.join(path,img)
                hb,hw,hbw=histogram_maker.get_rlbwxh(img)
                row=[hb[0],hw[0],hbw[0],str(file_name)]
                my_list.append(row)
                print str(file_name)
                #break
        except Exception,e:
            print str(e)
        return my_list

target_img=sys.argv[1]
target_csv=sys.argv[2]
c=csv_feature_generator()
c.convert_image_to_rlbwxymsh_csv(target_img,target_csv)