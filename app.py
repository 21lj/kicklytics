# ================= IMPORT SPACES FIRST =================
# MUST be imported before any CUDA-related packages
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False

# ================= STANDARD IMPORTS =================
import os
import tempfile
import json

import gradio as gr
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from backend.utils.video_utils import extract_crops, write_touches
from backend.features.team_assigner import TeamClassifier, resolve_goalkeepers_teamid
from backend.features.player_ball_assigner import PlayerBallAssigner

import torch


# ================= CONFIG =================

BALL_ID = 0
GK_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= LOAD MODEL =================

print("[INFO] Loading YOLO model from HF...")

MODEL_PATH = hf_hub_download(
    repo_id="Lijo21/kicklytics-models",
    filename="yolo-models/yolov8x/best.pt"
)

model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"[INFO] YOLO loaded on {DEVICE}")


# ================= INFERENCE =================

def process_video(input_video):
    if input_video is None:
        return None, "⚠️ Please upload a video clip first."
    
    output_video = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    try:
        print("[INFO] Extracting crops...")

        crops = extract_crops(
            input_video,
            model,
            stride=30,
            player_id=PLAYER_ID
        )

        team_classifier = TeamClassifier(device=DEVICE)
        team_classifier.fit(crops)

        ellipse_annotator = sv.EllipseAnnotator(
            thickness=2,
            color=sv.ColorPalette.from_hex(
                ['#00BFFF', '#FF1493', '#FFD700']
            )
        )

        triangle_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex('#00FF41')
        )

        possession_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex('#FF0000')
        )

        tracker = sv.ByteTrack()
        tracker.reset()

        ball_assigner = PlayerBallAssigner()

        team_touches = {0: 0, 1: 0}
        last_assigned_id = None

        video_info = sv.VideoInfo.from_video_path(input_video)

        sink = sv.VideoSink(
            output_video,
            video_info=video_info
        )

        frames = sv.get_video_frames_generator(input_video)

        print("[INFO] Processing video...")

        with sink:
            for frame in tqdm(
                frames,
                total=video_info.total_frames
            ):
                result = model.predict(
                    frame,
                    conf=0.3,
                    verbose=False,
                    device=DEVICE
                )[0]

                detections = sv.Detections.from_ultralytics(result)

                # -------- BALL --------
                ball_detections = detections[detections.class_id == BALL_ID]

                if len(ball_detections):
                    ball_detections.xyxy = sv.pad_boxes(
                        xyxy=ball_detections.xyxy,
                        px=10
                    )

                # -------- HUMANS --------
                human_detections = detections[detections.class_id != BALL_ID]

                human_detections = human_detections.with_nms(
                    threshold=0.5,
                    class_agnostic=True
                )

                human_detections = tracker.update_with_detections(
                    detections=human_detections
                )

                player_detections = human_detections[
                    human_detections.class_id == PLAYER_ID
                ]

                gk_detections = human_detections[
                    human_detections.class_id == GK_ID
                ]

                referee_detections = human_detections[
                    human_detections.class_id == REFEREE_ID
                ]

                # -------- TEAM CLASSIFICATION --------
                if len(player_detections):
                    crops = [
                        sv.crop_image(frame, box)
                        for box in player_detections.xyxy
                    ]

                    player_detections.class_id = team_classifier.predict(crops)

                # -------- GK TEAM --------
                if len(gk_detections) and len(player_detections):
                    gk_detections.class_id = resolve_goalkeepers_teamid(
                        player_detections,
                        gk_detections
                    )

                # -------- POSSESSION --------
                possession_detections = sv.Detections.empty()

                if len(ball_detections):
                    ball_bbox = ball_detections.xyxy[0]

                    candidates = sv.Detections.merge([
                        player_detections,
                        gk_detections
                    ])

                    players_dict = {
                        tid: {"bbox": bbox}
                        for tid, bbox in zip(
                            candidates.tracker_id,
                            candidates.xyxy
                        )
                    }

                    assigned_id = ball_assigner.assign_ball_to_player(
                        players_dict,
                        ball_bbox
                    )

                    if assigned_id != -1:
                        possession_detections = candidates[
                            candidates.tracker_id == assigned_id
                        ]

                        if assigned_id != last_assigned_id:
                            team_id = int(possession_detections.class_id[0])
                            if team_id in team_touches:
                                team_touches[team_id] += 1
                            last_assigned_id = assigned_id

                # -------- DRAW --------
                if len(referee_detections):
                    referee_detections.class_id -= 1

                team_detection = sv.Detections.merge([
                    player_detections,
                    gk_detections,
                    referee_detections
                ])

                annotated = frame.copy()

                annotated = ellipse_annotator.annotate(
                    annotated,
                    team_detection
                )

                annotated = triangle_annotator.annotate(
                    annotated,
                    ball_detections
                )

                annotated = possession_annotator.annotate(
                    annotated,
                    possession_detections
                )

                sink.write_frame(annotated)

        # write_touches(team_touches)

        # Format stats for display
        total_touches = max(sum(team_touches.values()), 1)
        stats_text = f"""
### 📊 Match Statistics

| Metric | Team 1 (Blue) | Team 2 (Pink) |
|--------|---------------|---------------|
| **Ball Touches** | {team_touches[0]} | {team_touches[1]} |
| **Possession %** | {team_touches[0]/total_touches*100:.1f}% | {team_touches[1]/total_touches*100:.1f}% |

**Total Events Tracked:** {sum(team_touches.values())}
        """

        return output_video, stats_text

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"❌ Error processing video: {str(e)}"


# Apply GPU decorator ONLY after function is defined and ONLY on Spaces
if HF_SPACES:
    process_video = spaces.GPU(process_video)


# ================= GRADIO UI =================

with gr.Blocks(title="Kicklytics - Football Analysis") as demo:
    gr.Markdown("""
    # ⚽ Kicklytics Player Detection
    
    Upload a football match video clip to get:
    - **Player detection** with team classification (Blue vs Pink)
    - **Ball tracking** with possession analysis
    - **Match statistics** (ball touches, possession %)
    
    *Supports MP4, AVI, MOV formats*
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(
                label="📹 Upload Match Video",
                sources=["upload"],
                format="mp4"
            )
            process_btn = gr.Button(
                "🚀 Analyze Video",
                variant="primary",
                size="lg"
            )
            
        with gr.Column(scale=2):
            output_video = gr.Video(
                label="🎬 Processed Video",
                interactive=False
            )
    
    stats_output = gr.Markdown(label="📊 Statistics")
    
    gr.Markdown("""
    ---
    **How it works:** The model detects players, assigns them to teams based on jersey colors, 
    tracks the ball, and calculates possession by detecting which player is closest to the ball.
    """)

    process_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=[output_video, stats_output]
    )


if __name__ == "__main__":
    demo.launch()