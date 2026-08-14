import os
import tempfile
import numpy as np

# ZeroGPU must be imported before CUDA-related packages.
try:
    import spaces
except ImportError:
    # Local-development fallback when the `spaces` package is unavailable.
    class _LocalSpaces:
        @staticmethod
        def GPU(duration=300):
            def decorator(func):
                return func
            return decorator

    spaces = _LocalSpaces()

import cv2
import torch
import gradio as gr
import supervision as sv

from tqdm import tqdm
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from backend.features.team_assigner import (
    TeamClassifier,
    resolve_goalkeepers_teamid,
)
from backend.features.player_ball_assigner import PlayerBallAssigner


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_REPO = "Lijo21/kicklytics-models"
MODEL_FILENAME = "yolo-models/yolov8x/best.pt"

MAX_VIDEO_SECONDS = 60

# Frames used to collect crops for team classification.
TEAM_CROP_STRIDE = 30

# Re-classify existing tracked players periodically.
#
# The original pipeline runs SigLIP on every player on every
# frame. That is unnecessarily expensive.
TEAM_RECLASSIFY_STRIDE = 30

YOLO_CONFIDENCE = 0.3
NMS_THRESHOLD = 0.5

BALL_ID = 0
GK_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[INFO] Device: {DEVICE}")


# ============================================================
# DOWNLOAD MODEL ONCE
# ============================================================
#
# hf_hub_download uses the Hugging Face cache. The model is
# downloaded only when it is not already available locally.
# ============================================================

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILENAME,
    repo_type="model",
)

print(f"[INFO] YOLO model: {MODEL_PATH}")


# ============================================================
# LOAD YOLO MODEL ONCE
# ============================================================
#
# Keep the model object global.
#
# We do NOT repeatedly construct YOLO inside process_video().
# During inference, Ultralytics is explicitly told which device
# to use.
# ============================================================

MODEL = YOLO(MODEL_PATH)


# ============================================================
# VIDEO VALIDATION
# ============================================================

def get_video_duration(video_path):
    """
    Return the duration of a video in seconds.
    """

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    capture.release()

    if fps is None or fps <= 0:
        raise ValueError("Unable to determine the video's frame rate.")

    return frame_count / fps


def validate_video(video_path):
    """
    Validate the uploaded video before allocating significant
    processing resources.
    """

    if not video_path:
        raise gr.Error("Please upload a football video.")

    try:
        duration = get_video_duration(video_path)
    except Exception as exc:
        raise gr.Error(
            f"Unable to read the uploaded video: {exc}"
        ) from exc

    if duration > MAX_VIDEO_SECONDS:
        raise gr.Error(
            f"Video is {duration:.1f} seconds long. "
            f"The maximum allowed duration is "
            f"{MAX_VIDEO_SECONDS} seconds."
        )

    return duration


# ============================================================
# TEAM CROP COLLECTION
# ============================================================

def collect_team_crops(video_path):
    """
    Run YOLO on every TEAM_CROP_STRIDE-th frame and collect
    player crops for fitting the team classifier.

    Important optimization:
    The detections generated here are cached.

    During the actual processing pass, these cached detections
    are reused instead of running YOLO a second time on the same
    sampled frames.
    """

    frame_generator = sv.get_video_frames_generator(
        video_path,
        stride=TEAM_CROP_STRIDE,
    )

    crops = []
    cached_detections = {}

    for sample_index, frame in enumerate(
        tqdm(
            frame_generator,
            desc="Collecting team-classification crops",
        )
    ):

        frame_index = sample_index * TEAM_CROP_STRIDE

        result = MODEL.predict(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False,
            device=DEVICE,
        )[0]

        detections = sv.Detections.from_ultralytics(result)

        cached_detections[frame_index] = detections

        player_detections = detections[
            detections.class_id == PLAYER_ID
        ]

        player_detections = player_detections.with_nms(
            threshold=NMS_THRESHOLD,
            class_agnostic=True,
        )

        crops.extend(
            [
                sv.crop_image(frame, xyxy)
                for xyxy in player_detections.xyxy
            ]
        )

    return crops, cached_detections


# ============================================================
# TEAM CLASSIFIER
# ============================================================

def create_team_classifier(crops):
    """
    Create and fit a fresh TeamClassifier for the current video.

    The classifier is intentionally created per request because
    its UMAP/KMeans state is video-specific.

    The expensive SigLIP model is therefore not allowed to retain
    fitted team information from a previous uploaded video.
    """

    if not crops:
        raise gr.Error(
            "No player crops were found. "
            "Unable to determine team classifications."
        )

    classifier = TeamClassifier(
        device=DEVICE,
        batch_size=32,
    )

    with torch.inference_mode():
        classifier.fit(crops)

    return classifier


# ============================================================
# TOUCH STATISTICS
# ============================================================

def save_touch_statistics(output_directory, team_touches):
    """
    Preserve the existing statistics format while avoiding the
    fixed ../output_videos/clip-info/info.txt path used by the
    development script.
    """

    info_directory = os.path.join(
        output_directory,
        "clip-info",
    )

    os.makedirs(
        info_directory,
        exist_ok=True,
    )

    info_path = os.path.join(
        info_directory,
        "info.txt",
    )

    total_touches = (
        team_touches.get(0, 0)
        + team_touches.get(1, 0)
    )

    if total_touches > 0:
        team1_percentage = (
            team_touches.get(0, 0)
            * 100
            / total_touches
        )

        team2_percentage = (
            team_touches.get(1, 0)
            * 100
            / total_touches
        )
    else:
        team1_percentage = 50.0
        team2_percentage = 50.0

    with open(info_path, "w") as file:
        file.write(
            f"Team 1: {team1_percentage:.1f}%\n"
        )
        file.write(
            f"Team 2: {team2_percentage:.1f}%\n"
        )

    return info_path


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

@spaces.GPU(duration=300)
def process_video(input_video):
    """
    Process a football video inside a ZeroGPU allocation.

    The complete processing pipeline is kept here so that GPU
    operations occur inside the ZeroGPU-decorated function.
    """

    try:
        # --------------------------------------------------------
        # Validate video
        # --------------------------------------------------------

        duration = validate_video(input_video)

        print(
            f"[INFO] Processing {duration:.1f}s video"
        )

        # --------------------------------------------------------
        # Team classification preparation
        # --------------------------------------------------------
        #
        # This performs the first, sparse YOLO pass.
        #
        # The important optimization is that its detections are
        # cached and reused during the second pass.
        # --------------------------------------------------------

        print(
            "[INFO] Collecting player crops "
            "for team classification..."
        )

        with torch.inference_mode():
            crops, cached_detections = collect_team_crops(
                input_video
            )

        print(
            f"[INFO] Collected {len(crops)} player crops."
        )

        # --------------------------------------------------------
        # Fit team classifier
        # --------------------------------------------------------

        print(
            "[INFO] Fitting team classifier..."
        )

        team_classifier = create_team_classifier(
            crops
        )

        print(
            "[INFO] Team classifier fitted."
        )

        # --------------------------------------------------------
        # Annotators
        # --------------------------------------------------------

        ellipse_annotator = sv.EllipseAnnotator(
            thickness=2,
            color=sv.ColorPalette.from_hex(
                [
                    "#00BFFF",
                    "#FF1493",
                    "#FFD700",
                ]
            ),
        )

        triangle_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex(
                "#00FF41"
            ),
        )

        possession_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex(
                "#FF0000"
            ),
        )

        # --------------------------------------------------------
        # ByteTrack
        # --------------------------------------------------------

        tracker = sv.ByteTrack()
        tracker.reset()

        ball_assigner = PlayerBallAssigner()

        # --------------------------------------------------------
        # Team-touch statistics
        # --------------------------------------------------------

        team_touches = {
            0: 0,
            1: 0,
        }

        last_assigned_id = None

        # --------------------------------------------------------
        # Team assignment cache
        # --------------------------------------------------------
        #
        # tracker_id -> team_id
        #
        # Instead of running SigLIP for every player on every
        # frame, only new/recently refreshed tracks are classified.
        # --------------------------------------------------------

        player_team_cache = {}

        last_team_classification_frame = {}

        # --------------------------------------------------------
        # Video information
        # --------------------------------------------------------

        video_info = sv.VideoInfo.from_video_path(
            input_video
        )

        # --------------------------------------------------------
        # Unique output directory
        # --------------------------------------------------------

        output_directory = tempfile.mkdtemp(
            prefix="kicklytics_"
        )

        output_path = os.path.join(
            output_directory,
            "tracked_output.avi",
        )

        # --------------------------------------------------------
        # Frame generator
        # --------------------------------------------------------

        frame_generator = sv.get_video_frames_generator(
            input_video
        )

        print(
            "[INFO] Processing video..."
        )

        # --------------------------------------------------------
        # Process video
        # --------------------------------------------------------

        with torch.inference_mode():

            with sv.VideoSink(
                output_path,
                video_info=video_info,
            ) as video_sink:

                for frame_number, frame in enumerate(
                    tqdm(
                        frame_generator,
                        total=video_info.total_frames,
                        desc="Processing",
                    )
                ):

                    # ==================================================
                    # YOLO DETECTION
                    # ==================================================
                    #
                    # Reuse detections from the crop-collection pass
                    # whenever this is one of the sampled frames.
                    #
                    # This removes duplicate YOLO inference for those
                    # frames.
                    # ==================================================

                    detections = cached_detections.get(
                        frame_number
                    )

                    if detections is None:

                        result = MODEL.predict(
                            frame,
                            conf=YOLO_CONFIDENCE,
                            verbose=False,
                            device=DEVICE,
                        )[0]

                        detections = (
                            sv.Detections.from_ultralytics(
                                result
                            )
                        )

                    # ==================================================
                    # BALL
                    # ==================================================

                    ball_detections = detections[
                        detections.class_id == BALL_ID
                    ]

                    if len(ball_detections) > 0:
                        ball_detections.xyxy = sv.pad_boxes(
                            xyxy=ball_detections.xyxy,
                            px=10,
                        )

                    # ==================================================
                    # HUMANS
                    # ==================================================

                    human_detections = detections[
                        detections.class_id != BALL_ID
                    ]

                    human_detections = (
                        human_detections.with_nms(
                            threshold=NMS_THRESHOLD,
                            class_agnostic=True,
                        )
                    )

                    human_detections = (
                        tracker.update_with_detections(
                            detections=human_detections
                        )
                    )

                    # ==================================================
                    # SPLIT ROLES
                    # ==================================================

                    player_detections = (
                        human_detections[
                            human_detections.class_id
                            == PLAYER_ID
                        ]
                    )

                    gk_detections = (
                        human_detections[
                            human_detections.class_id
                            == GK_ID
                        ]
                    )

                    referee_detections = (
                        human_detections[
                            human_detections.class_id
                            == REFEREE_ID
                        ]
                    )

                    # ==================================================
                    # TEAM CLASSIFICATION
                    # ==================================================
                    #
                    # Original code runs:
                    #
                    #     team_classifier.predict(...)
                    #
                    # for every player on every frame.
                    #
                    # We instead classify:
                    #
                    # 1. New tracker IDs immediately.
                    # 2. Existing IDs periodically.
                    #
                    # This dramatically reduces SigLIP inference while
                    # preserving dynamic correction.
                    # ==================================================

                    if len(player_detections) > 0:

                        crops_to_classify = []
                        crop_tracker_ids = []

                        for bbox, tracker_id in zip(
                            player_detections.xyxy,
                            player_detections.tracker_id,
                        ):

                            if tracker_id is None:
                                continue

                            should_classify = (
                                tracker_id
                                not in player_team_cache
                            )

                            last_frame = (
                                last_team_classification_frame.get(
                                    tracker_id
                                )
                            )

                            if (
                                not should_classify
                                and last_frame is not None
                                and (
                                    frame_number
                                    - last_frame
                                    >= TEAM_RECLASSIFY_STRIDE
                                )
                            ):
                                should_classify = True

                            if should_classify:

                                crop = sv.crop_image(
                                    frame,
                                    bbox,
                                )

                                crops_to_classify.append(
                                    crop
                                )

                                crop_tracker_ids.append(
                                    tracker_id
                                )

                        # ----------------------------------------------
                        # Batch team inference
                        # ----------------------------------------------

                        if crops_to_classify:

                            predicted_team_ids = (
                                team_classifier.predict(
                                    crops_to_classify
                                )
                            )

                            for (
                                tracker_id,
                                team_id,
                            ) in zip(
                                crop_tracker_ids,
                                predicted_team_ids,
                            ):

                                player_team_cache[
                                    tracker_id
                                ] = int(team_id)

                                last_team_classification_frame[
                                    tracker_id
                                ] = frame_number

                        # ----------------------------------------------
                        # Apply cached team IDs
                        # ----------------------------------------------

                        classified_team_ids = []

                        for tracker_id in (
                            player_detections.tracker_id
                        ):

                            if tracker_id is None:
                                classified_team_ids.append(
                                    0
                                )
                            else:
                                classified_team_ids.append(
                                    player_team_cache.get(
                                        tracker_id,
                                        0,
                                    )
                                )

                        # player_detections.class_id = (
                        #     classified_team_ids
                        # )
                        player_detections.class_id = np.asarray(
                              classified_team_ids,
                              dtype=np.int32,
                          )

                    # ==================================================
                    # GOALKEEPER TEAM ASSIGNMENT
                    # ==================================================

                    if (
                        len(gk_detections) > 0
                        and len(player_detections) > 0
                    ):

                        gk_detections.class_id = (
                            resolve_goalkeepers_teamid(
                                player_detections,
                                gk_detections,
                            )
                        )

                    # ==================================================
                    # BALL POSSESSION / BALL TOUCH
                    # ==================================================

                    possession_detections = (
                        sv.Detections.empty()
                    )

                    if len(ball_detections) > 0:

                        ball_bbox = (
                            ball_detections.xyxy[0]
                        )

                        candidates = sv.Detections.merge(
                            [
                                player_detections,
                                gk_detections,
                            ]
                        )

                        players_dict = {
                            tracker_id: {
                                "bbox": bbox
                            }
                            for tracker_id, bbox in zip(
                                candidates.tracker_id,
                                candidates.xyxy,
                            )
                            if tracker_id is not None
                        }

                        assigned_id = (
                            ball_assigner.assign_ball_to_player(
                                players_dict,
                                ball_bbox,
                            )
                        )

                        if assigned_id != -1:

                            possession_detections = (
                                candidates[
                                    candidates.tracker_id
                                    == assigned_id
                                ]
                            )

                            if (
                                assigned_id
                                != last_assigned_id
                                and len(
                                    possession_detections
                                ) > 0
                            ):

                                team_id = int(
                                    possession_detections
                                    .class_id[0]
                                )

                                if team_id in team_touches:
                                    team_touches[
                                        team_id
                                    ] += 1

                                last_assigned_id = (
                                    assigned_id
                                )

                    # ==================================================
                    # REFEREE COLOR MAPPING
                    # ==================================================

                    if len(referee_detections) > 0:
                        referee_detections.class_id -= 1

                    # ==================================================
                    # MERGE FOR VISUALIZATION
                    # ==================================================

                    team_wise_detection = (
                        sv.Detections.merge(
                            [
                                player_detections,
                                gk_detections,
                                referee_detections,
                            ]
                        )
                    )

                    # ==================================================
                    # ANNOTATION
                    # ==================================================

                    annotated_frame = frame.copy()

                    annotated_frame = (
                        ellipse_annotator.annotate(
                            scene=annotated_frame,
                            detections=team_wise_detection,
                        )
                    )

                    annotated_frame = (
                        triangle_annotator.annotate(
                            scene=annotated_frame,
                            detections=ball_detections,
                        )
                    )

                    annotated_frame = (
                        possession_annotator.annotate(
                            scene=annotated_frame,
                            detections=possession_detections,
                        )
                    )

                    # ==================================================
                    # WRITE FRAME
                    # ==================================================

                    video_sink.write_frame(
                        annotated_frame
                    )

        # ============================================================
        # SAVE TOUCH STATISTICS
        # ============================================================

        info_path = save_touch_statistics(
            output_directory,
            team_touches,
        )

        print(
            f"[INFO] Output saved to: {output_path}"
        )

        print(
            f"[INFO] Touch statistics saved to: {info_path}"
        )

        return output_path

    except gr.Error:
        raise

    except Exception as exc:
        print(
            f"[ERROR] Video processing failed: {exc}"
        )

        raise gr.Error(
            f"Video processing failed: {exc}"
        ) from exc


# ============================================================
# GRADIO INTERFACE
# ============================================================

demo = gr.Interface(
    fn=process_video,

    inputs=gr.Video(
        label="Upload Football Video",
    ),

    outputs=gr.Video(
        label="Tracked Output",
    ),

    title="Kicklytics Player Tracking",

    description=(
        "Upload a football video and get player detection, "
        "team classification, ball tracking, possession, "
        "ball-touch statistics, and annotated output."
    ),
)


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    demo.launch(debug=True, share=True)