from PIL import Image
import sys
import re
import os
import time
from collections import Counter

unfound = []
zone_name = ""
zone_labels=[]
zone_count=0
try:
    pattern = re.compile("\d+")  # pattern to find x,y coordinate
    zone_folder = sys.argv[1]  # gt path
    resized_folder = sys.argv[2]  # image training path
    folders = os.listdir(zone_folder)
    for folder in folders:
        fp=os.path.join(zone_folder,folder) #source image
        fp32x32 = os.path.join(resized_folder, folder)  # 32 x 32 image path
        images = os.listdir(fp)
        os.mkdir(fp32x32)
        for image in images:
            fn = image.split(".")[0]
            try:
                source_img_path = os.path.join(fp, image)  # image to resize
                dest_img_path = os.path.join(fp32x32, image)  # resized image path
                img2resize = Image.open(source_img_path)
                img32x32=img2resize.resize((32,32),Image.ANTIALIAS)
                img32x32.save(dest_img_path)
                print dest_img_path
            except IOError, e:
                print "Resize exception:\n" + str(e)
                unfound.append(fn)
                #img2zone.close()
                # break#
                continue
except Exception, i:
    print "error io." + str(i)

print "Failed Resized:\n"+str(unfound)

