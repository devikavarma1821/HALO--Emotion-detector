from flask import Flask, render_template, Response, request, jsonify
import cv2
from deepface import DeepFace
import threading
import numpy as np

app = Flask(__name__)

emotion_result = None  # Store detected emotion
lock = threading.Lock()  # Thread-safe variable access
cap = cv2.VideoCapture(0)  # Keep camera open globally

def detect_emotions():
    global emotion_result
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 15 == 0 and emotion_result is None:  # Process every 15th frame
            try:
                # Convert BGR to RGB for DeepFace
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (640, 480))

                analysis = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=True)

                with lock:
                    if analysis and 'dominant_emotion' in analysis[0]:
                        dominant_emotion = analysis[0]['dominant_emotion']
                        confidence = max(analysis[0]['emotion'].values())  # Get highest confidence score
                        if confidence > 30:  # Set threshold to filter weak detections
                            emotion_result = dominant_emotion
                        else:
                            emotion_result = "Uncertain"
            except:
                with lock:
                    emotion_result = "No face detected"

        # Display detected emotion
        with lock:
            if emotion_result:
                cv2.putText(frame, emotion_result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotions(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clear', methods=['POST'])
def clear():
    global emotion_result
    with lock:
        emotion_result = None  # Reset emotion
    return jsonify({"message": "Cleared"}), 200

if __name__ == "__main__":
    app.run(debug=True)
