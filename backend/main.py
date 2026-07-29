import os
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
from dotenv import load_dotenv

import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from utils.video_utils import extract_crops
from features.team_assigner import TeamClassifier, resolve_goalkeepers_teamid

# ==================== CONFIG ====================
INPUT_VIDEO   = "./input_videos/test-3.mp4"
OUTPUT_VIDEO  = "./output_videos/out-new.avi"
MODEL_PATH    = "./model/yolov8x/detect/train/weights/best.pt"
DEVICE        = "cpu"

BALL_ID       = 0
GK_ID         = 1
PLAYER_ID     = 2
REFEREE_ID    = 3
print(BALL_ID)
# ================================================


def main():
    # 1. Load model
    model = YOLO(MODEL_PATH)

    # 2. Extract crops & fit team classifier
    print("[INFO] Extracting player crops for team classification...")
    crops = extract_crops(INPUT_VIDEO, model, stride=30, player_id=PLAYER_ID)
    print(f"[INFO] Collected {len(crops)} crops.")

    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)
    print("[INFO] Team classifier fitted.")

    # 3. Annotators
    ellipse_annotator = sv.EllipseAnnotator(
        thickness=2,
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
    )

    triangle_annotator = sv.TriangleAnnotator(
        base=20,
        height=17,
        color=sv.Color.from_hex('#00FF41')
    )

    # 4. Tracker
    tracker = sv.ByteTrack()
    tracker.reset()

    # 5. Video I/O
    video_info = sv.VideoInfo.from_video_path(INPUT_VIDEO)
    video_sink = sv.VideoSink(OUTPUT_VIDEO, video_info=video_info)
    frame_generator = sv.get_video_frames_generator(INPUT_VIDEO)

    # 6. Process frames
    print("[INFO] Processing video...")
    with video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            result = model.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            # ---- Ball ----
            ball_detections = detections[detections.class_id == BALL_ID]
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # ---- Humans ----
            human_detections = detections[detections.class_id != BALL_ID]
            human_detections = human_detections.with_nms(threshold=0.5, class_agnostic=True)
            human_detections = tracker.update_with_detections(detections=human_detections)

            # ---- Split roles ----
            player_detections   = human_detections[human_detections.class_id == PLAYER_ID]
            gk_detections       = human_detections[human_detections.class_id == GK_ID]
            referee_detections  = human_detections[human_detections.class_id == REFEREE_ID]

            # ---- Classify player teams ----
            if len(player_detections) > 0:
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
                player_detections.class_id = team_classifier.predict(player_crops)

            # ---- Resolve GK teams via centroid distance ----
            if len(gk_detections) > 0 and len(player_detections) > 0:
                gk_detections.class_id = resolve_goalkeepers_teamid(player_detections, gk_detections)

            # ---- Map referee to third palette color (gold) ----
            if len(referee_detections) > 0:
                referee_detections.class_id -= 1

            # ---- Merge for visualization ----
            team_wise_detection = sv.Detections.merge([
                player_detections,
                gk_detections,
                referee_detections
            ])

            # ---- Annotate ----
            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=team_wise_detection
            )
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections
            )

            video_sink.write_frame(annotated_frame)

    print(f"[INFO] Output saved to: {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()