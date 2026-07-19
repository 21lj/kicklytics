from utils.video_utils import get_video, save_video
from features.tracker import Tracker
from features.team_assigner import TeamAssigner

def main():
    video_frames = get_video('./input_videos/test1.mp4')

    tracker = Tracker('./model/yolo11x/yolo11.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='./stubs/track_stubs.pkl')
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_no, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_no], track['bbox'], player_id)
            tracks['players'][frame_no][player_id]['team'] = team
            tracks['players'][frame_no][player_id]['team_color'] = team_assigner.team_color[team]
    
    outframe = tracker.draw_annotations(video_frames, tracks)
    save_video(outframe, './output_videos/out.avi')





if __name__ == '__main__':
    main()