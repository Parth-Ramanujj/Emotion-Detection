import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import collections
import traceback

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Use a deque to store the last N predictions for temporal smoothing
prediction_history = collections.deque(maxlen=7)

# Load model safely
try:
    model = tf.keras.models.load_model(
        "emotion-web-app\model\emotion_model.keras",
        compile=False
    )
except Exception as e:
    print(f"Error loading model:")
    traceback.print_exc()
    model = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup CLAHE for better contrast under varying lighting (improves face input quality)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def process_frame(frame):
    if model is None:
        return frame

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improve image contrast
    gray_equalized = clahe.apply(gray)
    
    # Use higher scaleFactor & minNeighbors to reduce false positive faces
    faces = face_cascade.detectMultiScale(gray_equalized, scaleFactor=1.3, minNeighbors=6, minSize=(60, 60))

    if len(faces) == 0:
        prediction_history.clear() # Reset if no face found to avoid stale predictions

    for (x, y, w, h) in faces:
        # Add a slight padding to the facial bounding box to give the model better context
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        
        face = gray_equalized[y1:y2, x1:x2]
        
        # Preprocess for the model (fer2013_mini_XCEPTION expects 64x64 grayscale)
        face = cv2.resize(face, (64, 64))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)  # Add channel dim
        face = np.expand_dims(face, axis=0)   # Add batch dim

        # Predict emotion
        prediction = model.predict(face, verbose=0)[0]
        
        # Add to history for temporal smoothing
        prediction_history.append(prediction)
        
        # Compute the average prediction over the last N frames to remove jittering
        avg_prediction = np.mean(prediction_history, axis=0)
        
        dominant_emotion_idx = np.argmax(avg_prediction)
        emotion = emotion_labels[dominant_emotion_idx]
        confidence = avg_prediction[dominant_emotion_idx] * 100

        # Draw a sleek rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 200, 0), 2)
        
        # Put the smoothed prediction text above
        display_text = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    return frame
