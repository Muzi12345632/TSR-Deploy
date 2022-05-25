import pandas as pd
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from sklearn.metrics import accuracy_score


test = pd.read_csv('Test.csv')

##load model
model= load_model('TSR.h5')

imageDimensions= (32, 32)
data_dir= 'Test'
labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize(imageDimensions)
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

pred = model.predict_classes(X_test)

#Accuracy with the test data
print('Test Data accuracy: ',accuracy_score(labels, pred)*100)