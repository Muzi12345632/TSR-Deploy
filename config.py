import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
np.random.seed(42)

from matplotlib import style
style.use('fivethirtyeight')

## TODO finish upconfig file
## TODO code data ptoc file n splitting
## TODO build final model


#####------PARAMETERS------####
path= 'Train'
labelfile= 'labels.csv'
batch_size_val = 60
steps_per_epoch_val= 400
epochs_val= 30
imageDimensions= (32,32)
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
        curImg= cv2.resize(curImg,imageDimensions)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)


################Split Data

X_train,X_test, y_train, y_test =train_test_split(images,classNo, test_size=testRatio)
X_train,X_validation, y_train, y_validation =train_test_split(X_train,y_train, test_size=validationRatio)

#############Print out Data format and total number of pics
print("Data shapes")
print("Train",end='');print(X_train.shape,y_train.shape)
print("Validation",end='');print(y_validation.shape,X_validation.shape)
print("Test",end='');print(X_test.shape,y_test.shape)

######Read CSV file
data= pd.read_csv(labelfile)
print("data shape", data.shape, type(data))


##########Data pre-processing

def greyscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
#def equalise(img):
    #img= cv2.equalizeHist(img)
    #return img
def preprocessing(img):
    img= greyscale(img)
    #img= equalise(img)
    img= img/255
    return img

####TO Iterate and preprocess all the images

X_train= np.array(list(map(preprocessing, X_train)))
X_test= np.array(list(map(preprocessing, X_test)))
X_validation= np.array(list(map(preprocessing, X_validation)))

################ADD a depth of 1

X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation= X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test= X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

###############Augmentation of images To  Makeit more Generic
dataGen= ImageDataGenerator(width_shift_range= 0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)

y_train= to_categorical(y_train,noOfclasses)
y_test= to_categorical(y_test,noOfclasses)
y_validation= to_categorical(y_validation,noOfclasses)

##TODO import model file
from module import myModel

##Train model
model= myModel()
print(model.summary())
history= model.fit(dataGen.flow
                             (X_train,y_train,batch_size=batch_size_val),
                             steps_per_epoch=steps_per_epoch_val,
                             epochs=epochs_val,
                             validation_data=(X_test,y_test))

##########
score = model.evaluate(X_test,y_test,verbose=0)
print("Test Score:",score[0])
print("Test Accuracy:",score[1])

model.save("TSR_v22.h5")


