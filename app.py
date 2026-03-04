from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from utils import process_frame

app = Flask(__name__)

# Route for rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        # Get the base64 encoded image from the request
        data = request.json['image']
        # Remove the "data:image/jpeg;base64," prefix
        img_data = base64.b64decode(data.split(',')[1])
        
        # Convert to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        # Decode into cv2 frame
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Process the frame for emotion detection
            processed_frame = process_frame(frame)
            
            # Encode frame back to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            # Convert to base64 string
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'image': f'data:image/jpeg;base64,{processed_base64}'
            })
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        
    return jsonify({'success': False})

if __name__ == '__main__':
    # Run the app locally on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
