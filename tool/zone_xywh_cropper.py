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
    gt = sys.argv[1]  # gt path
    img = sys.argv[2]  # image training path
    files = os.listdir(gt)
    print ("NUM File:"+str(len(files)))
    # zone_path="/home/andri/Documents/5/thesis/dataset/UWIII/ZONE"
    zone_path = sys.argv[3]
    for file in files:
        try:
            if (file.endswith(".TXT")):
                print file
                fp = os.path.join(gt, file)  # traget groundtruth file
                fn = file.split(".")[0]
                img_path = os.path.join(img, fn + "BIN.TIF")  # target trainng image
                img2zone = Image.open(img_path)
                zdir = os.path.join(zone_path, fn)
                os.mkdir(zdir, 0777)
                print "IMG SIZE:" + str(img2zone.size[0]) + "x" + str(img2zone.size[1])
                with open(fp) as f:
                    xy = []
                    zone_type = False
                    zone_content = False
                    zone_label = ""
                    zone_content_label=""
                    for l in f:
                        line=l
                        if line.lower().startswith("box"):
                            xy=pattern.findall(line)
                            #print "XY="+str(xy)
                        if zone_content:
                            zone_content_label = line.split(":")[-1].strip()
                            zone_content=False
                        if zone_type & (len(zone_content_label)>0):
                            zone_label=line.split(":")[-1].strip()
                            print "ZONE label="+zone_label
                            zone_labels.append(zone_content_label+"_"+zone_label)
                            zone_count=zone_count+1
                            zone_type=False
                            zone_content = False
                            if (len(xy) == 5):  # Box          000008: 964 296 1433 2465
                                x0 = int(xy[1])
                                y0 = int(xy[2])
                                width = int(xy[3])
                                height = int(xy[4])
                                x1 = x0 + width
                                y1 = y0 + height
                                zone_name = ""
                                try:
                                    print "(" + str(x0) + "," + str(y0) + ") , (" + str(x1) + "," + str(y1) + ")"
                                    zone_img = img2zone.crop((x0, y0, x1, y1))
                                    zone_name = str(x0) + "_" + str(y0) + "_" + str(width) + "_" + str(height) +"_"+zone_content_label+"_"+zone_label+ ".TIF"  # fn + "ZONE_"+str(x0)+"_"+str(y0)+"_"+str(width)+"_"+str(height)+".TIF"
                                    zpath = os.path.join(zdir, zone_name)
                                    zone_img.save(zpath)
                                    print ("ZONE SAVED:"+zpath)
                                    zone_content_label = ""
                                except Exception, e:
                                    print "error croping:" + zone_name + "(" + str(x0) + "," + str(y0) + ")(" + str(
                                        x1) + "," + str(y1) + ")" + "\n" + str(e)
                                    unfound.append(str(e))
                                    zone_img.close()
                                    img2zone.close()
                                    continue

                        if line.lower().endswith("zone_content\n"):
                            zone_content = True
                        if line.lower().endswith("zone_type\n"):
                            zone_type = True

                                # break

                                #seq = seq + 1

                img2zone.close()
                f.close()
                # '''
        except Exception, e:
            print "error exception:" + str(e)
            unfound.append(str(e))
            img2zone.close()
            # break#
            continue
except IOError, i:
    unfound.append(str(i))
    print "error io." + str(i)

print "ZONE NAME="+str(set(zone_labels))
print Counter(zone_labels)
print "ZONE COUNT="+str(zone_count)
