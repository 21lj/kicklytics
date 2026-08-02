import os
import tempfile

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
    filename="best.pt"
)

model = YOLO(MODEL_PATH)

print("[INFO] YOLO loaded")


# ================= INFERENCE =================

def process_video(input_video):

    output_video = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name


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


    team_touches = {
        0:0,
        1:0
    }

    last_assigned_id = None



    video_info = sv.VideoInfo.from_video_path(input_video)


    sink = sv.VideoSink(
        output_video,
        video_info=video_info
    )


    frames = sv.get_video_frames_generator(
        input_video
    )


    print("[INFO] Processing video...")


    with sink:

        for frame in tqdm(
            frames,
            total=video_info.total_frames
        ):

            result = model.predict(
                frame,
                conf=0.3,
                verbose=False
            )[0]


            detections = sv.Detections.from_ultralytics(
                result
            )


            # -------- BALL --------

            ball_detections = detections[
                detections.class_id == BALL_ID
            ]


            if len(ball_detections):
                ball_detections.xyxy = sv.pad_boxes(
                    xyxy=ball_detections.xyxy,
                    px=10
                )


            # -------- HUMANS --------

            human_detections = detections[
                detections.class_id != BALL_ID
            ]


            human_detections = (
                human_detections
                .with_nms(
                    threshold=0.5,
                    class_agnostic=True
                )
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
                    sv.crop_image(
                        frame,
                        box
                    )
                    for box in player_detections.xyxy
                ]


                player_detections.class_id = (
                    team_classifier.predict(crops)
                )


            # -------- GK TEAM --------

            if (
                len(gk_detections)
                and len(player_detections)
            ):

                gk_detections.class_id = (
                    resolve_goalkeepers_teamid(
                        player_detections,
                        gk_detections
                    )
                )


            # -------- POSSESSION --------

            possession_detections = sv.Detections.empty()


            if len(ball_detections):

                ball_bbox = ball_detections.xyxy[0]


                candidates = sv.Detections.merge(
                    [
                        player_detections,
                        gk_detections
                    ]
                )


                players_dict = {
                    tid:{
                        "bbox":bbox
                    }
                    for tid,bbox
                    in zip(
                        candidates.tracker_id,
                        candidates.xyxy
                    )
                }


                assigned_id = (
                    ball_assigner.assign_ball_to_player(
                        players_dict,
                        ball_bbox
                    )
                )


                if assigned_id != -1:

                    possession_detections = (
                        candidates[
                            candidates.tracker_id == assigned_id
                        ]
                    )


                    if assigned_id != last_assigned_id:

                        team_id = int(
                            possession_detections.class_id[0]
                        )

                        if team_id in team_touches:
                            team_touches[team_id]+=1


                        last_assigned_id = assigned_id



            # -------- DRAW --------

            if len(referee_detections):

                referee_detections.class_id -= 1


            team_detection = sv.Detections.merge(
                [
                    player_detections,
                    gk_detections,
                    referee_detections
                ]
            )


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


            sink.write_frame(
                annotated
            )


    write_touches(team_touches)


    return output_video



# ================= GRADIO =================

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        label="Upload Match Video"
    ),
    outputs=gr.Video(
        label="Processed Video"
    ),
    title="Kicklytics Player Detection",
    description="YOLO player detection + team classification + possession analysis"
)


demo.launch()