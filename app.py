# ================= IMPORT SPACES FIRST =================
# MUST be imported before any CUDA-related packages
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False

    # Local/CPU dev has no `spaces` package. Stub it so @spaces.GPU(...) below
    # works everywhere without falling back to the old post-hoc reassignment
    # pattern (`process_video = spaces.GPU(process_video)`).
    class _SpacesStub:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    spaces = _SpacesStub()

# ================= STANDARD IMPORTS =================
import os
import tempfile
import json

import numpy as np
import gradio as gr
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from backend.utils.video_utils import extract_crops, write_touches
from backend.features.team_assigner import TeamClassifier, resolve_goalkeepers_teamid
from backend.features.player_ball_assigner import PlayerBallAssigner
from backend.features.possession_track import PossessionTracker, draw_possession

import torch


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


# ================= LOAD MODEL =================

print("[INFO] Loading YOLO model from HF...")

MODEL_PATH = hf_hub_download(
    repo_id="Lijo21/kicklytics-models",
    filename="yolo-models/yolov8x/best.pt",
    repo_type="model",
)

model = YOLO(MODEL_PATH)
model.to(DEVICE)

print(f"[INFO] YOLO loaded on {DEVICE}")


# ================= INFERENCE =================

@spaces.GPU(duration=300)
def process_video(input_video):
    if input_video is None:
        return None, "⚠️ Please upload a video clip first."

    # ---- Video length protection (Section 9) ----
    try:
        video_info = sv.VideoInfo.from_video_path(input_video)
    except Exception:
        return None, "❌ Unable to open video."

    duration_seconds = video_info.total_frames / video_info.fps
    if duration_seconds > MAX_VIDEO_SECONDS:
        return None, (
            f"⚠️ Video is too long ({duration_seconds:.0f}s). "
            f"Please upload a clip under {MAX_VIDEO_SECONDS}s."
        )

    output_video = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    try:
        print("[INFO] Extracting crops...")

        # extract_crops must also return the detections it computed on the
        # sampled frames (every `stride`-th frame) so the main loop can reuse
        # them instead of running YOLO a second time on those frames. See
        # note below the file for the matching backend/utils/video_utils.py
        # change this depends on (Section 3).
        crops, warmup_detections = extract_crops(
            input_video,
            model,
            stride=30,
            player_id=PLAYER_ID,
            device=DEVICE
        )

        if len(crops) == 0:
            return None, "❌ No player detections found in this video."

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

        # Distance thresholds were tuned in pixels on a ~1280px-wide clip.
        # Scale them so possession/ball-assignment works on any uploaded resolution.
        res_scale = video_info.width / 1280

        ball_assigner = PlayerBallAssigner()
        ball_assigner.max_player_ball_distance = int(70 * res_scale)

        possession_tracker = PossessionTracker(proximity_px=int(60 * res_scale))

        team_touches = {0: 0, 1: 0}
        last_assigned_id = None

        # Per-video team-classification cache (Section 4 + 12).
        # tracker_id -> team_id. Declared fresh inside process_video() each
        # call so a new upload never reuses another video's team assignments.
        player_team_cache = {}
        player_seen_count = {}

        sink = sv.VideoSink(
            output_video,
            video_info=video_info
        )

        frames = sv.get_video_frames_generator(input_video)

        print("[INFO] Processing video...")

        with sink, torch.inference_mode():
            for frame_idx, frame in enumerate(tqdm(
                frames,
                total=video_info.total_frames
            )):
                # -------- DETECT (reuse cached detections where possible) --------
                if frame_idx in warmup_detections:
                    detections = warmup_detections[frame_idx]
                else:
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

                # -------- TEAM CLASSIFICATION (tracker_id cache, Section 4) --------
                if len(player_detections):
                    idx_needing_classification = []

                    for i, tid in enumerate(player_detections.tracker_id):
                        seen = player_seen_count.get(tid, 0)
                        if tid not in player_team_cache or seen % TEAM_RECLASSIFY_STRIDE == 0:
                            idx_needing_classification.append(i)
                        player_seen_count[tid] = seen + 1

                    if idx_needing_classification:
                        crops_to_classify = [
                            sv.crop_image(frame, player_detections.xyxy[i])
                            for i in idx_needing_classification
                        ]
                        predicted = team_classifier.predict(crops_to_classify)
                        for i, pred in zip(idx_needing_classification, predicted):
                            player_team_cache[player_detections.tracker_id[i]] = int(pred)

                    player_detections.class_id = np.array([
                        player_team_cache[tid] for tid in player_detections.tracker_id
                    ])

                # -------- GK TEAM --------
                if len(gk_detections) and len(player_detections):
                    gk_detections.class_id = resolve_goalkeepers_teamid(
                        player_detections,
                        gk_detections
                    )
                elif len(gk_detections):
                    # Can't tell GK's team without players on screen to compare
                    # against (e.g. camera zoomed tight on keeper). Raw class_id
                    # would still be GK_ID (1), which looks like a valid team id
                    # and gets drawn/counted wrong -> skip GK until players reappear.
                    gk_detections = sv.Detections.empty()

                # ---- Possession % tracking ----
                player_boxes = list(player_detections.xyxy) + list(gk_detections.xyxy)
                team_ids = (
                    [int(c) + 1 for c in player_detections.class_id] +
                    [int(c) + 1 for c in gk_detections.class_id]
                )
                ball_bbox = ball_detections.xyxy[0] if len(ball_detections) > 0 else None

                state = possession_tracker.update(
                    ball_bbox, player_boxes, team_ids,
                    frame_number=frame_idx, fps=video_info.fps
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

                annotated = draw_possession(annotated, state, team_colors)

                annotated = possession_annotator.annotate(
                    annotated,
                    possession_detections
                )

                sink.write_frame(annotated)

        # write_touches(state)

        # Format stats for display
        total_touches = max(sum(team_touches.values()), 1)
        stats_text = f"""
        ### 📊 Match Statistics

        | Metric | Team 1 (Blue) | Team 2 (Pink) |
        |--------|---------------|---------------|
        | **Ball Touches** | {team_touches[0]} | {team_touches[1]} |
        | **Possession %** | {state.team1_percentage:.1f}% | {state.team2_percentage:.1f}% |

        **Total Events Tracked:** {sum(team_touches.values())}
                """

        return output_video, stats_text

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"❌ Video processing failed: {str(e)}"


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