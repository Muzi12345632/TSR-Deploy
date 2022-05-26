from flask import *
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2


app= Flask(__name__)
camera=cv2.VideoCapture(0)
font= cv2.FONT_HERSHEY_SIMPLEX


def generate_frames():
    while True:
        success, frame = camera.read(0)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.putText(img, "CLASS: ", (20, 35), font, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', img)
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
