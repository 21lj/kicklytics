from utils.video_utils import get_video, save_video
from features.tracker import Tracker
from features.team_assigner import TeamAssigner
from features.player_ball_assigner import PlayerBallAssigner

def main():
    video_frames = get_video('./input_videos/test1.mp4')

    tracker = Tracker('./model/yolo11x/yolo11.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='./stubs/track_stubs.pkl')
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_no, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_no], track['bbox'], player_id)
            tracks['players'][frame_no][player_id]['team'] = team
            tracks['players'][frame_no][player_id]['team_color'] = team_assigner.team_color[team]

    player_assigner = PlayerBallAssigner()
    for frame_no, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_no][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_no][assigned_player]['has_ball'] = True
    
    outframe = tracker.draw_annotations(video_frames, tracks)
    save_video(outframe, './output_videos/out.avi')





if __name__ == '__main__':
    main()