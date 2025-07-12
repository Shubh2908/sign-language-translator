from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
import pyttsx3

app = Flask(__name__)

model = tf.keras.models.load_model('model/sign_language_model.h5')

class_names = ['A', 'B', 'C']  

engine = pyttsx3.init()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    import time
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("❌ Camera failed to open")
        return

    last_prediction = ""
    engine = pyttsx3.init()

    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Failed to grab frame")
            break

        print("✅ Frame captured")

        img = cv2.resize(frame, (64, 64))  
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        print("👉 Calling model.predict...")
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        print(f"✅ Prediction: {predicted_class}")

        cv2.putText(frame, f"Prediction: {predicted_class}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("❌ Encoding failed")
            continue

        frame = buffer.tobytes()

        print("✅ Yielding frame to browser")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.05) 


if __name__ == '__main__':
    app.run(debug=True)
