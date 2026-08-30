import cv2
import math
import numpy as np

from .constants import (
    PITCH_LENGTH_CM,
    PITCH_WIDTH_CM,
)

class PitchRenderer:
    """Renders player/ball positions on a canonical top-down pitch.

    the render canvas — and therefore marker pixel size —
    is always computed from a FIXED canonical height, never from the source
    video's resolution. `render_for_panel()` renders at that fixed canonical
    size, then does a single uniform resize to whatever panel height the
    caller needs for compositing. A uniform resize scales pitch lines and
    markers together, so proportions stay identical across videos; only the
    old code's mistake — deriving marker size from a per-video out_height —
    is what caused inconsistent dot sizes."""

    CANONICAL_CANVAS_HEIGHT = 700  # fixed, independent of any source video

    def __init__(self, canvas_height=CANONICAL_CANVAS_HEIGHT,
                 length_cm=PITCH_LENGTH_CM, width_cm=PITCH_WIDTH_CM,
                 margin_cm=600.0):
        self.length_cm = length_cm
        self.width_cm = width_cm
        self.margin_cm = margin_cm
        self.canvas_height = canvas_height

        total_h_cm = width_cm + 2 * margin_cm
        total_w_cm = length_cm + 2 * margin_cm
        self.scale = canvas_height / total_h_cm
        self.canvas_width = int(round(total_w_cm * self.scale))

        self._base = self._draw_base()

    def _to_px(self, x_cm, y_cm):
        px = (x_cm + self.margin_cm) * self.scale
        py = (y_cm + self.margin_cm) * self.scale
        return int(round(px)), int(round(py))

    def _draw_base(self):
        img = np.full((self.canvas_height, self.canvas_width, 3), (40, 120, 40), dtype=np.uint8)
        white = (255, 255, 255)
        t = max(1, int(round(self.scale * 15)))

        L, W = self.length_cm, self.width_cm

        def line(p1, p2):
            cv2.line(img, self._to_px(*p1), self._to_px(*p2), white, t)

        # outline
        line((0, 0), (L, 0)); line((L, 0), (L, W))
        line((L, W), (0, W)); line((0, W), (0, 0))
        # halfway line
        line((L / 2, 0), (L / 2, W))
        # center circle
        cv2.circle(img, self._to_px(L / 2, W / 2), int(round(915 * self.scale)), white, t)
        # penalty boxes: 2015 cm deep x 4100 cm wide (standard canonical geometry)
        line((0, 1450), (2015, 1450)); line((2015, 1450), (2015, 5550)); line((2015, 5550), (0, 5550))
        line((L, 1450), (L - 2015, 1450)); line((L - 2015, 1450), (L - 2015, 5550)); line((L - 2015, 5550), (L, 5550))
        return img

    def render(self, positions, team_colors, ball_position=None,
               status_label=None, player_radius_cm=110.0, ball_radius_cm=55.0,
               passing_lane_result=None):
        """
        positions: list of {"team": int, "x": float, "y": float} in canonical
        pitch-cm (the geometry-stage output of build_positions — visualization
        never sees raw boxes, video resolution, or H).
        ball_position: optional {"x": float, "y": float} in pitch-cm.
        player_radius_cm / ball_radius_cm: real-world marker size, scaled by
        the FIXED canonical canvas scale only — never by video resolution or
        homography values.
        """
        img = self._base.copy()
        player_r = max(2, int(round(player_radius_cm * self.scale)))
        ball_r = max(1, int(round(ball_radius_cm * self.scale)))

        for pos in positions:
            x, y = pos["x"], pos["y"]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            color = team_colors.get(int(pos["team"]), (200, 200, 200))
            cv2.circle(img, self._to_px(x, y), player_r, color, -1)
            cv2.circle(img, self._to_px(x, y), player_r, (0, 0, 0), 1)

        if ball_position is not None:
            bx, by = ball_position["x"], ball_position["y"]
            if np.isfinite(bx) and np.isfinite(by):
                cv2.circle(img, self._to_px(bx, by), ball_r, (0, 255, 255), -1)

        if passing_lane_result:
            img = self.draw_passing_lanes(img, passing_lane_result)

        if status_label is not None:
            cv2.putText(img, f"Homography: {status_label}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def draw_passing_lanes(self, img, result):
        """Draw canonical passing-lane results on the OpenCV pitch image."""
        if img is None or not result or result.get("carrier") is None:
            return img

        output = img.copy()
        for lane in result.get("lanes", []):
            if not lane.get("clear", False) and not result.get("draw_blocked", True):
                continue

            start = self._to_px(lane["from"]["x"], lane["from"]["y"])
            end = self._to_px(lane["to"]["x"], lane["to"]["y"])
            color = tuple(int(v) for v in lane.get("color_bgr", (0, 0, 255)))
            thickness = max(2, int(round(12 * self.scale)))

            if lane.get("clear", False):
                cv2.line(output, start, end, color, thickness, cv2.LINE_AA)
            else:
                sx, sy = start
                ex, ey = end
                length = max(1.0, math.hypot(ex - sx, ey - sy))
                dash = max(10, int(round(35 * self.scale)))
                gap = max(8, int(round(22 * self.scale)))
                step = dash + gap
                for offset in np.arange(0.0, length, step):
                    t0 = offset / length
                    t1 = min(1.0, (offset + dash) / length)
                    p0 = (int(round(sx + (ex - sx) * t0)), int(round(sy + (ey - sy) * t0)))
                    p1 = (int(round(sx + (ex - sx) * t1)), int(round(sy + (ey - sy) * t1)))
                    cv2.line(output, p0, p1, color, thickness, cv2.LINE_AA)

            receiver_radius = max(5, int(round(90 * self.scale)))
            cv2.circle(output, end, receiver_radius, color, 2, cv2.LINE_AA)

        return output

    def render_for_panel(self, positions, team_colors, panel_height,
                          ball_position=None, status_label=None,
                          player_radius_cm=110.0, ball_radius_cm=55.0,
                          passing_lane_result=None):
        """Render at the fixed canonical canvas, then uniformly resize to
        `panel_height` (e.g. to match a broadcast video's height for
        side-by-side compositing). This is the entry point app.py should
        call — never construct PitchRenderer with a video-derived height."""
        img = self.render(
            positions, team_colors, ball_position=ball_position,
            status_label=status_label,
            player_radius_cm=player_radius_cm, ball_radius_cm=ball_radius_cm,
            passing_lane_result=passing_lane_result,
        )
        if panel_height == self.canvas_height:
            return img
        target_width = int(round(self.canvas_width * (panel_height / self.canvas_height)))
        return cv2.resize(img, (target_width, panel_height), interpolation=cv2.INTER_AREA)

    def render_trajectories(self, tracks, team_colors, track_labels=None,
                             line_thickness=2, dot_radius_cm=60.0):
        """
        Step 10 — trajectory visualization. Separate entry point from
        render()/render_for_panel() (single-frame positions); this one draws
        polylines over a track's full history. Does not touch or call them.

        tracks: {track_id: [{"team": int, "x": float, "y": float}, ...]}
                (chronological order) — e.g. from
                TrajectoryStore.tracks[track_id], already canonical pitch-cm.
        """
        img = self._base.copy()
        dot_r = max(2, int(round(dot_radius_cm * self.scale)))

        for track_id, records in tracks.items():
            pts = [(r["x"], r["y"]) for r in records if math.isfinite(r["x"]) and math.isfinite(r["y"])]
            if len(pts) < 2:
                continue
            team = records[0]["team"]
            color = team_colors.get(int(team), (200, 200, 200))

            px_pts = [self._to_px(x, y) for x, y in pts]
            for p1, p2 in zip(px_pts, px_pts[1:]):
                cv2.line(img, p1, p2, color, line_thickness, cv2.LINE_AA)

            cv2.circle(img, px_pts[0], dot_r, color, -1)
            cv2.circle(img, px_pts[-1], dot_r, color, -1)
            cv2.circle(img, px_pts[-1], dot_r + 2, (255, 255, 255), 1)

            if track_labels:
                label = track_labels.get(track_id, str(track_id))
                cv2.putText(img, label, (px_pts[-1][0] + 8, px_pts[-1][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def render_clean_trajectories(self, clean_result, team_colors,
                                   line_thickness=2, dot_radius_cm=60.0):
        """
        Draws the OUTPUT of trajectory.cleaner.clean_trajectories(), status-
        aware. New/separate method from render_trajectories() (raw) so the
        raw-panel drawing code stays untouched.

        clean_result: {track_id: [clean_point, ...]} — clean_point has
        frame/team/x/y/raw_x/raw_y/valid/confidence/status, per
        trajectory/cleaner.py.

        Legend:
          solid line + dot  = valid, connected segment
          dashed line       = interpolated segment
          orange X          = id_switch_suspect (drawn at raw_x/raw_y)
          red X             = jump_rejected (drawn at raw_x/raw_y)
        """
        img = self._base.copy()
        dot_r = max(2, int(round(dot_radius_cm * self.scale)))

        for track_id, points in clean_result.items():
            usable = [p for p in points if p["status"] in ("valid", "interpolated")]
            team = next((p["team"] for p in points if p["team"] is not None), None)
            color = team_colors.get(int(team), (200, 200, 200)) if team is not None else (200, 200, 200)

            for prev, curr in zip(usable, usable[1:]):
                if curr["frame"] - prev["frame"] != 1:
                    continue  # don't draw a line straight across an unfilled gap
                p1 = self._to_px(prev["x"], prev["y"])
                p2 = self._to_px(curr["x"], curr["y"])
                if curr["status"] == "interpolated" or prev["status"] == "interpolated":
                    self._dashed_line(img, p1, p2, color, line_thickness)
                else:
                    cv2.line(img, p1, p2, color, line_thickness, cv2.LINE_AA)

            if usable:
                cv2.circle(img, self._to_px(usable[0]["x"], usable[0]["y"]), dot_r, color, -1)
                cv2.circle(img, self._to_px(usable[-1]["x"], usable[-1]["y"]), dot_r, color, -1)
                cv2.circle(img, self._to_px(usable[-1]["x"], usable[-1]["y"]), dot_r + 2, (255, 255, 255), 1)

            for p in points:
                if p["status"] not in ("jump_rejected", "id_switch_suspect"):
                    continue
                if not (math.isfinite(p["raw_x"]) and math.isfinite(p["raw_y"])):
                    continue
                mark_color = (0, 140, 255) if p["status"] == "id_switch_suspect" else (0, 0, 220)
                self._draw_x(img, self._to_px(p["raw_x"], p["raw_y"]), mark_color, size=dot_r)

        return img

    @staticmethod
    def _draw_x(img, center, color, size):
        cx, cy = center
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, 2, cv2.LINE_AA)

    @staticmethod
    def _dashed_line(img, p1, p2, color, thickness, dash_len=6):
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return
        n_dashes = max(1, int(dist // dash_len))
        for i in range(0, n_dashes, 2):
            t0 = i / n_dashes
            t1 = min((i + 1) / n_dashes, 1.0)
            sx, sy = x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0
            ex, ey = x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1
            cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, thickness, cv2.LINE_AA)

