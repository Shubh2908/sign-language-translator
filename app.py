from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

model = load_model("model/sign_language_model.h5")
class_names = ['A', 'B', 'C']  # Replace with your signs

app = Flask(__name__)
engine = pyttsx3.init()
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            roi = cv2.resize(frame, (64, 64))
            roi = img_to_array(roi) / 255.0
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)
            label = class_names[np.argmax(prediction)]

            cv2.putText(frame, f"Prediction: {label}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            engine.say(label)
            engine.runAndWait()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
