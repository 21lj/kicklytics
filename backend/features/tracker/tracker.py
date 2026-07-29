from ultralytics import YOLO


class Tracker:
    """
    Lightweight YOLO wrapper. Tracking and annotation are now handled
    directly in main.py via supervision (sv.ByteTrack, sv.EllipseAnnotator, etc.).
    """

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, frame, conf=0.3):
        """Run inference on a single frame."""
        return self.model.predict(frame, conf=conf)[0]