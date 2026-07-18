from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2
from utils import get_center_of_bbox, get_bbox_width
import pickle
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """Detect video frames in batches of 20 frames."""
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=0.1
            )
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            'ball': [],
            'players': [],
            'referee': []
        }

        for frame_no, detection in enumerate(detections):
            class_names = detection.names
            # {0: ball, 1: GK, 2: player, 3: referee}
            class_name_rev = {v: k for k, v in class_names.items()}

            sv_detection = sv.Detections.from_ultralytics(detection)
            # converting GK -> player
            for obj_idx, class_id in enumerate(sv_detection.class_id):
                if class_names[class_id] == "goalkeeper":
                    sv_detection.class_id[obj_idx] = class_name_rev['player']

            detection_with_track = self.tracker.update_with_detections(sv_detection) #generates track ids for bounding boxes

            tracks['ball'].append({})
            tracks['players'].append({})
            tracks['referee'].append({})

            for frame_detect in detection_with_track:
                bbox = frame_detect[0].tolist()
                class_id = frame_detect[3]
                track_id = frame_detect[4]

                if class_id == class_name_rev['player']:
                    tracks['players'][frame_no][track_id] = {'bbox': bbox}

                elif class_id == class_name_rev['referee']:
                    tracks['referee'][frame_no][track_id] = {'bbox': bbox}

                elif class_id == class_name_rev['ball']:
                    tracks['ball'][frame_no][track_id] = {'bbox': bbox}
                            
            # print(sv_detection)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width=get_bbox_width(bbox)

        cv2.ellipse(
          frame,
          center=(x_center, y2),
          axes=(int(width), int(0.35*width)),
          angle=0.0,
          startAngle=-45,
          endAngle=235,
          color=color,
          thickness=2,
          lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y=int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
        out_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            ball_dict = tracks["ball"][frame_num]
            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referee"][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

            for tid, ref in referee_dict.items():
                frame = self.draw_ellipse(frame, ref['bbox'], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 100, 0))

            out_video_frames.append(frame)
        return out_video_frames

    







