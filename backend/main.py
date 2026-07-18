from utils.video_utils import get_video, save_video
from features.tracker import Tracker


def main():
    video_frames = get_video('./input_videos/test1.mp4')

    tracker = Tracker('./model/yolov8x/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='./stubs/track_stubs.pkl')
    outframe = tracker.draw_annotations(video_frames, tracks)
    save_video(outframe, './output_videos/out.avi')





if __name__ == '__main__':
    main()