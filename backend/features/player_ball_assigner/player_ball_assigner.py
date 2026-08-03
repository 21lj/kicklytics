from backend.utils.bbox_util import get_center_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Assign ball to the nearest player based on foot-to-ball distance.
        
        Args:
            players: dict of {player_id: {'bbox': [x1, y1, x2, y2]}}
            ball_bbox: [x1, y1, x2, y2]
        
        Returns:
            assigned_player: player_id or -1
        """
        ball_position = get_center_of_bbox(ball_bbox)
        min_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id

            return assigned_player