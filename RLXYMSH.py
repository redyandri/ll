import numpy as np
from PIL import Image
import cv2
import sys
import matplotlib.pyplot as plt
from Jansen_Shannon import JSD

class RLXYMSH(object):
    def __init__(self):
        return

    def get_rlbwxh(self, img):
        image=cv2.imread(img)
        runs=[]
        lengths=[]
        prev=-1
        count=0
        val=0
        height = image.shape[0]
        width = image.shape[1]
        for x in range(0,height):
            for y in range(0, width):
                val=image[x,y][0]
                if(prev==-1): #beginning of image
                    count=1
                else:
                    if (val==prev):
                        count=count+1
                    else:
                        runs.append(prev)
                        lengths.append(count)
                        count=1
                if ((x == height - 1) & (y == width - 1)):  # end of imsge
                    runs.append(val)
                    lengths.append(count)
                prev=val
        lengths_w=[lengths[k] for k,v in enumerate(runs) if v==255]   #lengths of only white pixels
        lengths_b = [lengths[k] for k, v in enumerate(runs) if v == 0] #lengths of only black pixels
        return self.get_histogram(lengths_b),self.get_histogram(lengths_w),self.get_histogram(lengths) #rlbxh,rlwxh,rlbwxh

    def get_rlbwyh(self, img):
        image=cv2.imread(img)
        runs=[]
        lengths=[]
        prev=-1
        count=0
        val=0
        height = image.shape[0]
        width = image.shape[1]
        for y in range(0,width):
            for x in range(0, height):
                val = image[x, y][0]
                if (prev == -1):  # beginning of image
                    count = 1
                else:
                    if (val == prev):
                        count = count + 1
                    else:
                        runs.append(prev)
                        lengths.append(count)
                        count = 1
                if ((x == height - 1) & (y == width - 1)):  # end of imsge
                    runs.append(val)
                    lengths.append(count)
                prev = val
        lengths_w = [lengths[k] for k, v in enumerate(runs) if v == 255]  # lengths of only white pixels
        lengths_b = [lengths[k] for k, v in enumerate(runs) if v == 0]  # lengths of only black pixels
        return self.get_histogram(lengths_b), self.get_histogram(lengths_w), self.get_histogram(
            lengths)  # rlbyh,rlwyh,rlbwyh

    def get_histogram(self, lengths):
        bins=[0,2,4,8,16,32,64,128,1024]
        hist=np.histogram(lengths,bins)
        return hist

    def show_histogram(self,lengths, bins, label):
        xlabels=["<=1","<=2","<=4","<=8","<=16","<=32","<=64",">=128"]
        y_pos=np.arange(len(bins)-1)
        plt.bar(y_pos,lengths)
        plt.xticks(y_pos,xlabels)
        plt.xlabel(label)
        plt.show()


#
# rl = RLXYMSH()
# img_path= sys.argv[1]
# h_b,h_w,h = rl.get_rlbwxh(img_path)
# print h_b
# print h_w
# print h
# #runs2,counts2 = rl.get_rly(img_path)
# rl.show_histogram(h_b[0],h_b[1],"RLBXH")
# rl.show_histogram(h_w[0],h_w[1],"RLWXH")
# rl.show_histogram(h[0],h[1],"RLBWXH")
# from Jansen_Shannon import JSD
#
# jsd=JSD()
# print jsd.get_distance(h[0],h_w[0])