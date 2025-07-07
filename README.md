# Sign Language to Speech Translator

Detects sign language gestures from webcam, predicts with a CNN, and converts to speech.

## How to Use

1. Place your dataset under `dataset/` with folders for each sign.
2. Train the model:

```bash
python train_model.py
```

3. Run the app:

```bash
python app.py
```

4. Open `http://127.0.0.1:5000/` in your browser.

## Tech

- Python, Flask, TensorFlow, OpenCV, pyttsx3
