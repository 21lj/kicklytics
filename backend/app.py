import os
import gradio as gr

from utils.video_utils import get_video, save_video
from features.tracker import Tracker
from huggingface_hub import hf_hub_download


# Download model once when Space starts
model_path = hf_hub_download(
    repo_id="<your-username>/kicklytics-models",
    filename="yolov8l.pt"
)

tracker = Tracker(model_path)


def process_video(video_path):

    video_frames = get_video(video_path)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path="./stubs/track_stubs.pkl"
    )

    output_frames = tracker.draw_annotations(
        video_frames,
        tracks
    )

    os.makedirs("./output_videos", exist_ok=True)

    output_path = "./output_videos/out.avi"

    save_video(
        output_frames,
        output_path
    )

    return output_path


demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        label="Upload Football Video"
    ),
    outputs=gr.Video(
        label="Tracked Output"
    ),
    title="Kicklytics Player Tracking",
    description="Upload a football video and get object tracking output."
)


if __name__ == "__main__":
    demo.launch()