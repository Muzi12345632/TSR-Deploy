import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model, Sequential

##########################

imageDimensions= (32,32)
frameWidth= 600
frameHeight= 480
brightness= 180
threshold= 0.90
font= cv2.FONT_HERSHEY_SIMPLEX

#################################

###Setup video camera
cap= cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
###Import  model
model= load_model('TSR.h5')


def grayscale(img):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalise(img):
    img= cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img= grayscale(img)
    img= equalise(img)
    img= img/255
    return img

def getClassname(classNo):
    if classNo == 0: return 'Speed limit (20km/h)'
    elif classNo == 1: return 'Speed limit (30km/h)'
    elif classNo == 2: return 'Speed limit (50km/h)'
    elif classNo == 3: return 'Speed limit (60km/h)'
    elif classNo == 4: return 'Speed limit (70km/h)'
    elif classNo == 5: return 'Speed limit (80km/h)'
    elif classNo == 6: return 'End of speed limit (80km/h)'
    elif classNo == 7: return 'Speed limit (100km/h)'
    elif classNo == 8: return 'Speed limit (120km/h)'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing veh over 3.5 tons'
    elif classNo == 11: return 'Right-of-way at intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo ==13: return 'Yield'
    elif classNo ==14: return 'Stop'
    elif classNo ==15: return 'No vehicles'
    elif classNo ==16: return 'Veh > 3.5 tons prohibited'
    elif classNo ==17: return 'No entry'
    elif classNo ==18: return 'General caution'
    elif classNo ==19: return 'Dangerous curve left'
    elif classNo ==20: return 'Dangerous curve right'
    elif classNo ==21: return 'Double curve'
    elif classNo ==22: return 'Bumpy road'
    elif classNo ==23: return 'Slippery road'
    elif classNo ==24: return 'Road narrows on the right'
    elif classNo ==25: return 'Road work'
    elif classNo ==26: return 'Traffic signals'
    elif classNo ==27: return 'Pedestrians'
    elif classNo ==28: return 'Children crossing'
    elif classNo ==29: return 'Bicycles crossing'
    elif classNo ==30: return 'Beware of ice/snow'
    elif classNo ==31: return 'Wild animals crossing'
    elif classNo ==32: return 'End speed + passing limits'
    elif classNo ==33: return 'Turn right ahead'
    elif classNo ==34: return 'Turn left ahead'
    elif classNo ==35: return 'Ahead only'
    elif classNo ==36: return 'Go straight or right'
    elif classNo ==37: return 'Go straight or left'
    elif classNo ==38: return 'Keep right'
    elif classNo ==39: return 'Keep left'
    elif classNo ==40: return 'Roundabout mandatory'
    elif classNo ==41: return 'End of no passing'
    elif classNo ==42: return 'End no passing veh > 3.5 tons'

while True:

    ##READ IMAGE

    