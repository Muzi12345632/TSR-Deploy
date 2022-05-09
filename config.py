import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')


#####------PARAMETERS------####
path= ''
labelfile= ''
batch_size_val = 50
steps_per_epoch_val= 2000
epochs_val= 30
imageDimensions= (32,32,3)
testRatio = 0.2
validationRatio= 0.2



########-------IMPORTING OF IMAGES---------#######
count= 0
images= []
classNo= []
myList= os.listdir(path)
print("total classes detected:", len(myList))
noOfclasses= len(myList)
print("Importing Classes........")
for x in range(0, len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
