import csv
import sys
from _ctypes_test import func
from collections import Counter
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors.ball_tree import BallTree
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import os
sys.path.insert(0,"../")
from RLXYMSH import *
#from histogram_classifier import *
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

l1=[1,2,3,4,5]
r1=[0,255,0,255,0]
l2=[1,2,3]
r2=[0,255,0]
l3=[1,2,3]
r3=[255,0,255]
l4=[22,56,11]
r4=[255,0,255]
print np.sign(l1)





