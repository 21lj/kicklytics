import cv2
import math
import time
from dataclasses import dataclass


@dataclass
class PossessionState:
    frame: int
    timestamp: float
    current_team: int
    team1_percentage: float
    team2_percentage: float
    confidence: float
    duration: float


class PossessionTracker:

    def __init__(
        self,
        proximity_px=60,
        switch_frames=5,
        missing_ball_frames=8
    ):

        self.proximity = proximity_px

        # Number of consecutive frames required before changing possession
        self.switch_frames = switch_frames

        # Number of frames we keep previous possession if ball disappears
        self.missing_ball_frames = missing_ball_frames

        self.team_counts = {1: 0, 2: 0}

        self.current_team = -1

        self.candidate_team = -1

        self.candidate_counter = 0

        self.frames_without_ball = 0

        self.possession_start_frame = 0

    def _nearest_team(self, ball_box, player_boxes, team_ids):

        if not ball_box:
            return -1, None

        bx = (ball_box[0] + ball_box[2]) / 2
        by = (ball_box[1] + ball_box[3]) / 2

        best_distance = float("inf")
        best_team = -1

        for bbox, team in zip(player_boxes, team_ids):

            x1, y1, x2, y2 = bbox

            px = max(x1, min(x2, bx))
            py = max(y1, min(y2, by))

            d = math.hypot(bx - px, by - py)

            if d < best_distance:
                best_distance = d
                best_team = team

        if best_distance > self.proximity:
            return -1, best_distance

        return best_team, best_distance

    def update(
        self,
        ball_bbox,
        player_boxes,
        team_ids,
        frame_number=0,
        fps=30
    ):

        team, distance = self._nearest_team(
            ball_bbox,
            player_boxes,
            team_ids
        )

        if team == -1:

            self.frames_without_ball += 1

            if self.frames_without_ball > self.missing_ball_frames:
                self.current_team = -1

            return self.get_state(frame_number, fps)

        self.frames_without_ball = 0

        if self.current_team == -1:

            self.current_team = team
            self.possession_start_frame = frame_number
            self.team_counts[team] += 1

            return self.get_state(frame_number, fps)

        if team == self.current_team:

            self.team_counts[team] += 1
            self.candidate_counter = 0
            self.candidate_team = -1

            return self.get_state(frame_number, fps)

        if self.candidate_team != team:

            self.candidate_team = team
            self.candidate_counter = 1

        else:

            self.candidate_counter += 1

        if self.candidate_counter >= self.switch_frames:

            self.current_team = self.candidate_team
            self.possession_start_frame = frame_number
            self.team_counts[self.current_team] += 1

            self.candidate_counter = 0
            self.candidate_team = -1

        return self.get_state(frame_number, fps)

    def percentages(self):

        total = self.team_counts[1] + self.team_counts[2]

        if total == 0:
            return (50.0, 50.0)

        return (
            self.team_counts[1] * 100 / total,
            self.team_counts[2] * 100 / total
        )


    def get_state(self, frame_number, fps):

        p1, p2 = self.percentages()

        duration = (frame_number - self.possession_start_frame) / fps

        confidence = min(
            1.0,
            self.candidate_counter / max(1, self.switch_frames)
        )

        return PossessionState(
            frame=frame_number,
            timestamp=frame_number / fps,
            current_team=self.current_team,
            team1_percentage=p1,
            team2_percentage=p2,
            confidence=confidence,
            duration=duration,
        )

def draw_possession(frame, state, team_colors):

    h, w = frame.shape[:2]

    bar_width = 420
    bar_height = 22

    x = (w - bar_width) // 2
    y = h - 55

    team1_color = team_colors.get(1, (0, 0, 255))
    team2_color = team_colors.get(2, (255, 0, 0))

    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (40, 40, 40), -1)

    team1_width = int(bar_width * state.team1_percentage / 100)

    cv2.rectangle(
        frame,
        (x, y),
        (x + team1_width, y + bar_height),
        team1_color,
        -1,
    )

    cv2.rectangle(
        frame,
        (x + team1_width, y),
        (x + bar_width, y + bar_height),
        team2_color,
        -1,
    )

    cv2.putText(
        frame,
        f"{state.team1_percentage:.1f}%",
        (x - 70, y + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        team1_color,
        2,
    )

    cv2.putText(
        frame,
        f"{state.team2_percentage:.1f}%",
        (x + bar_width + 10, y + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        team2_color,
        2,
    )

    team_name = {
        1: "TEAM 1",
        2: "TEAM 2",
        -1: "LOOSE BALL"
    }

    cv2.putText(
        frame,
        f"Possession : {team_name[state.current_team]}",
        (x, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )

    return frame