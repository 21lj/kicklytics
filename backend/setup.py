import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import subprocess
import tempfile
import os
# ================= CONFIG =================

BALL_ID = 0
GK_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
team_colors = {
    1: (255, 191, 0),
    2: (147, 20, 255),
}

MAX_VIDEO_SECONDS = 60          # reject uploads longer than this (Section 9)
TEAM_RECLASSIFY_STRIDE = 30     # re-run SigLIP on an already-known player every N frames (Section 4)
INFERENCE_STRIDE = 1            # placeholder only - NOT real frame skipping. ByteTrack +
                                 # PossessionTracker + touch counting are frame-dependent;
                                 # skipping would need detection propagation across skipped
                                 # frames. Left as a documented knob, not faked (Section 10).

HOMOGRAPHY_STRIDE = 30          # cam is mostly static within a clip -> recompute pitch
                                 # homography every N frames, not every frame. Last valid H
                                 # is reused in between (see cached_H below).
PITCH_PANEL_HEIGHT = None       # set to broadcast video height at runtime, keeps side-by-side
PITCH_RENDER_BACKEND = "opencv" # "opencv" for video, "mplsoccer" for established pitch plotting
                                 # panels the same height without hardcoding resolution.

# ================= LOAD MODEL =================
def load_models():
    print("[INFO] Loading YOLO model from HF...")

    MODEL_PATH = hf_hub_download(
        repo_id="Lijo21/kicklytics-models",
        filename="yolo-models/yolov8x/best.pt",
        repo_type="model",
    )

    model = YOLO(MODEL_PATH)
    model.to(DEVICE)

    print(f"[INFO] YOLO loaded on {DEVICE}")

    print("[INFO] Loading YOLO pose model from HF...")

    MODEL_PATH = hf_hub_download(
        repo_id="Lijo21/kicklytics-models",
        filename="yolo-models/yolov8x-pose/best.pt",
        repo_type="model",
    )

    pose_model = YOLO(MODEL_PATH)
    pose_model.to(DEVICE)

    print(f"[INFO] YOLO loaded on {DEVICE}")
    return model, pose_model

def convert_video_to_h264(input_video):
    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    command = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return output_path

    except subprocess.CalledProcessError as e:
        print(
            "[ERROR] Video conversion failed:"
        )
        print(
            e.stderr.decode(
                errors="ignore"
            )
        )

        return None