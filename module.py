from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D

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
    