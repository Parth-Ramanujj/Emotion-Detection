from flask import Flask, render_template, Response
import cv2
from utils import process_frame

app = Flask(__name__)

# Route for rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

def gen_frames():
    # Capture video from standard webcam (index 0)
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame for emotion detection
            frame = process_frame(frame)
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            # Yield the next frame over the HTTP stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for streaming video frames
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
