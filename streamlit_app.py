import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")
st.markdown("Ensure your webcam is connected and allow browser access. Press **Start** to begin!")

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Process the frame using our existing utils logic
    processed_img = process_frame(img)
    
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

webrtc_streamer(
    key="emotion-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
