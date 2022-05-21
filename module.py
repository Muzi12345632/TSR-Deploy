from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizer_v1 import Adam
import os

##TODO build and save module
##TODO plot accuracy graphs
##TODO test module on web cam
##TODO start webplanning for technology


path= 'Train'
myList= os.listdir(path)
noOfclasses= len(myList)
imageDimensions= (32,32,3)

##build model

def myModel():
    no_of_filters= 60
    size_of_filter= (5,5) ##Kernel
    size_of_filter2= (3,3)
    size_of_pool= (2,2)
    no_of_nodes=500   ## number of nodes in hidden layers
    model= Sequential()
    model.add((Conv2D(no_of_filters,size_of_filter,input_shape=(imageDimensions[0],imageDimensions[1],1),activation='relu')))  #Adding moreconvolutional layers
    model.add((Conv2D(no_of_filters,size_of_filter,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    ##TODO add second part of model
    model.add((Conv2D(no_of_filters//2,size_of_filter2,activation='relu')))
    model.add((Conv2D(no_of_filters // 2,size_of_filter2,activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_nodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfclasses,activation='softmax'))
    ##Compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
