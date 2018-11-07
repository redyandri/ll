import numpy as np
from PIL import Image
import cv2
import sys
import matplotlib.pyplot as plt
from Jansen_Shannon import JSD


class RLXYMSH:
    def __init__(self):
        self.files=[]
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
        lengths_w = []
        lengths_b = []
        try:
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
        except Exception, e:
            print "EXCEPTION ON RLBWMH:%s" % str(e)
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
        lengths_w = []
        lengths_b = []
        try:
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
        except Exception, e:
            print "EXCEPTION ON RLBWMH:%s" % str(e)
        return self.get_histogram(lengths_b), self.get_histogram(lengths_w), self.get_histogram(lengths)  # rlbyh,rlwyh,rlbwyh

    def get_diagonals(self,grid, bltr=True): #False:main diag, True: side diagnals
        dim=grid.shape[0]#len(grid.shape[0])
        # print "grid shape:%s"%str((np.array(grid)).shape)
        #dim = len(grid[0])
        return_grid = [[] for total in xrange(2 * dim -1)]#len(grid) - 1)]
        for row in xrange(dim):#len(grid)):
            #print "len(grid):%d" %(len(grid))
            for col in xrange(len(grid[row])):
                #print "len(grid[row]:%d" %(len(grid[row]))
                if bltr:
                    return_grid[row + col].append([col,row])
                    #print "return_grid[row + col].append([col,row]):%d,%d"%(col,row)
                else:
                    return_grid[col - row + (dim - 1)].append([col,row])
                    #print "return_grid[col - row + (dim - 1)].append([col,row]):%d,%d" %(col,row)
        grid = []
        for j in return_grid:
            for k in j:
                grid.append(k)
        return grid

    def get_rlbwmh(self, img): #get diagonal runs length
        image=cv2.imread(img)
        runs=[]
        lengths=[]
        prev=-1
        count=0
        val=0
        height = image.shape[0]
        width = image.shape[1]
        grid=np.zeros(shape=(height,width))
        lengths_w=[]
        lengths_b=[]
        try:
            if height < width:
                grid=grid.reshape((width,height))  # height must be larger than width to make diagonal runs detected
            diag_idx=self.get_diagonals(grid, False) #bltr True: side diag, False:main diag
            for x in diag_idx:
                val=image[x[0],x[1]][0]
                if(prev==-1): #beginning of image
                    count=1
                else:
                    if (val==prev):
                        count=count+1
                    else:
                        runs.append(prev)
                        lengths.append(count)
                        count=1
                if ((x[0] == height - 1) & (x[1] == width - 1)):  # end of imsge
                    runs.append(val)
                    lengths.append(count)
                prev=val
            lengths_w=[lengths[k] for k,v in enumerate(runs) if v==255]   #lengths of only white pixels
            lengths_b = [lengths[k] for k, v in enumerate(runs) if v == 0] #lengths of only black pixels
        except Exception,e:
            print "EXCEPTION ON RLBWMH:%s"%str(e)
        return self.get_histogram(lengths_b),self.get_histogram(lengths_w),self.get_histogram(lengths) #rlbxh,rlwxh,rlbwxh


    def get_rlbwsh(self, img): #get diagonal runs length
        image=cv2.imread(img)
        runs=[]
        lengths=[]
        prev=-1
        count=0
        val=0
        height = image.shape[0]
        width = image.shape[1]
        grid = np.zeros(shape=(height, width))
        lengths_w = []
        lengths_b = []
        try:
            if height < width:
                grid = grid.reshape((width, height))  # height must be larger than width to make diagonal runs detected
            diag_idx=self.get_diagonals(grid,True)
            for x in diag_idx:
                val=image[x[0],x[1]][0]
                if(prev==-1): #beginning of image
                    count=1
                else:
                    if (val==prev):
                        count=count+1
                    else:
                        runs.append(prev)
                        lengths.append(count)
                        count=1
                if ((x[0] == height - 1) & (x[1] == width - 1)):  # end of imsge
                    runs.append(val)
                    lengths.append(count)
                prev=val
            lengths_w=[lengths[k] for k,v in enumerate(runs) if v==255]   #lengths of only white pixels
            lengths_b = [lengths[k] for k, v in enumerate(runs) if v == 0] #lengths of only black pixels
        except Exception, e:
            print "EXCEPTION ON RLBWSH:%s" % str(e)
        return self.get_histogram(lengths_b),self.get_histogram(lengths_w),self.get_histogram(lengths) #rlbxh,rlwxh,rlbwxh


    def getRunLengthX(self, img):
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
        return runs, lengths

    def getRunLengthY(self, img):
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
        return runs, lengths


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

    def combineRunLength(self,runs,lengths):
        combiRuns=runs[0]
        combiLength=lengths[0]
        for i in range(1,len(runs)):
            r=runs[i]
            l=lengths[i]
            if combiRuns[len(combiRuns)-1]==r[0]:
                combiLength[len(combiLength)-1]=combiLength[len(combiLength)-1] + l[0]
                del r[0]
                del l[0]
            combiRuns=combiRuns+r
            combiLength = combiLength + l
        return combiRuns,combiLength


#
# rl = RLXYMSH()
# img_path= sys.argv[1]
#h_b,h_w,h = rl.get_rlbwxh(img_path)
# print h_b
# print h_w
# print h
# #runs2,counts2 = rl.get_rly(img_path)
#rl.show_histogram(h_b[0],h_b[1],"RLBXH")
# rl.show_histogram(h_w[0],h_w[1],"RLWXH")
# rl.show_histogram(h[0],h[1],"RLBWXH")
# from Jansen_Shannon import JSD
#
# jsd=JSD()
# print jsd.get_distance(h[0],h_w[0])