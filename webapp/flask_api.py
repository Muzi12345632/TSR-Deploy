from flask import *
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2


app= Flask(__name__)
camera=cv2.VideoCapture(0)
font= cv2.FONT_HERSHEY_SIMPLEX

def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #areaMin = cv2.getTrackbarPos("Area", "Parameters")
        areaMin= 2000
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            #print(len(approx))
            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)



def generate_frames():
    while True:
        success, frame = camera.read(0)
        imgContour= frame.copy()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(img, "CLASS: ", (20, 35), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        threshold1 = 169
        threshold2 = 185
        imgCanny = cv2.Canny(img, threshold1, threshold2)
        kernel = np.ones((5, 5))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        getContours(imgDil, imgContour)


        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', imgContour)

            img = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Contet-Type: image/jpeg\r\n\r\n' + img + b'\r\n')



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Video')
def Video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
##Classes of Traffic Signs
