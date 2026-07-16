from utils.video_utils import get_video, save_video

def main():
    video_frames = get_video('./input_videos/test1.mp4')

    save_video(video_frames, './output_videos/out.avi')





if __name__ == '__main__':
    main()