import sys

# sys.path.insert(0,"../")
from tool.csv_feature_generator import *
from my_tool import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import cv2
from RLXYMSH import *
import numpy as np
from tool import *
from collections import Counter
import pandas as pd
import tool
from sklearn.model_selection import train_test_split
from KNNClassifier import *
from sklearn.neighbors import KNeighborsClassifier
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable
import xml.etree.ElementTree as ET
from collections import Counter
import csv
import shutil



sys.path.insert(0,"/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/tool")
from my_tool import *
src_path="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/gt_daten_dta_produktion"
src_dataset="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/zones"
mini_dataset="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/mini_dataset"
mini_dataset_txt="/home/andri/Documents/s2/5/master_arbeit/app/logical_labeling/mini_dataset.txt"
train_dataset_txt="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/train_mini_dataset.txt"
val_dataset_txt="/home/andri/Documents/s2/5/master_arbeit/dataset/historical/val_mini_dataset.txt"

tool=my_tool()
class_dict = {
            'paragraph': 1,
            'page-number': 2,
            'catch-word': 3,
            'header': 4,
            'heading': 5,
            'signature-mark': 6,
            'other': 7,
            'Separator': 8,
            'footnote': 9,
            'marginalia': 10,
            'Graphic': 11,
            'Maths': 12,
            'caption': 13,
            'Table': 14,
            'footnote-continued': 15,
            'endnote': 16
        }
#tool.convert_class_label_to_number
a="/home/andri/Documents/s2/5/master_arbeit/dataset/uw3/unrelocated/augmented/J03C/2592_3300_1499_836_RESIZED_SHEARQ1L3BO_math.bin.png"

spl=str(a).split("_")[-2]
aug=str(spl)[:-6]
print aug

