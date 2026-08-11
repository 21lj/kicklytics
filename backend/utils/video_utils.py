import supervision as sv
from tqdm import tqdm


def extract_crops(video_path, model, stride=30, player_id=2, conf=0.3, nms_threshold=0.5):
    """
    Extract player crops from a video to fit the team classifier.
    """
    frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
    crops = []

    for frame in tqdm(frame_generator, desc='Collecting crops'):
        result = model.predict(frame, conf=conf)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=nms_threshold, class_agnostic=True)
        detections = detections[detections.class_id == player_id]

        crops += [
            sv.crop_image(frame, xyxy)
            for xyxy in detections.xyxy
        ]

    return crops

def write_touches(state):
    print(state)
    with open("../output_videos/clip-info/info.txt", "w") as f:
        f.write(f"Team 1: {state.team1_percentage:.1f}%\n")
        f.write(f"Team 2: {state.team2_percentage:.1f}%\n")