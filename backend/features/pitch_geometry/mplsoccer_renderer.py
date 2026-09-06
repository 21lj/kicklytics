import cv2
import math
import numpy as np

from .constants import (
    PITCH_LENGTH_CM,
    PITCH_WIDTH_CM,
)

class MplSoccerRenderer:
    """Render the same canonical pitch state using mplsoccer.

    Geometry contract:
        canonical x/y are centimetres in [0, 12000] x [0, 7000].
        mplsoccer custom pitch uses metres, so the adapter performs only:

            x_m = x_cm / 100
            y_m = y_cm / 100

    `invert_y=True` preserves the notebook's image-style convention where
    y=0 is the top touchline and y=7000 is the bottom touchline. H itself is
    never changed.
    """

    CANONICAL_CANVAS_HEIGHT = 700
    CANONICAL_CANVAS_WIDTH = 1200

    def __init__(self, canvas_height=CANONICAL_CANVAS_HEIGHT,
                 length_cm=PITCH_LENGTH_CM, width_cm=PITCH_WIDTH_CM):
        try:
            import matplotlib.pyplot as plt
            from mplsoccer import Pitch
        except ImportError as exc:
            raise ImportError(
                "MplSoccerRenderer requires mplsoccer and matplotlib. "
                "Install with: pip install mplsoccer matplotlib"
            ) from exc

        self.plt = plt
        self.Pitch = Pitch
        self.length_m = float(length_cm) / 100.0
        self.width_m = float(width_cm) / 100.0
        self.canvas_height = int(canvas_height)
        self.canvas_width = int(round(self.canvas_height * self.length_m / self.width_m))
        self.pitch = Pitch(
            pitch_type="custom",
            pitch_length=self.length_m,
            pitch_width=self.width_m,
            invert_y=True,
            pitch_color="grass",
            line_color="white",
            axis=False,
            label=False,
            tick=False,
            linewidth=1.5,
        )

    @staticmethod
    def _xy_m(x_cm, y_cm):
        return float(x_cm) / 100.0, float(y_cm) / 100.0

    @staticmethod
    def _color_rgb(color_bgr):
        b, g, r = [int(v) for v in color_bgr]
        return (r / 255.0, g / 255.0, b / 255.0)

    def _new_canvas(self):
        fig = self.plt.figure(
            figsize=(self.canvas_width / 100.0, self.canvas_height / 100.0),
            dpi=100,
        )
        ax = fig.add_axes([0, 0, 1, 1])
        self.pitch.draw(ax=ax)
        return fig, ax

    def _to_image(self, fig):
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        self.plt.close(fig)
        return image

    def _draw_players(self, ax, positions, team_colors):
        for team_id in sorted(set(int(p["team"]) for p in positions if p.get("team") is not None)):
            team_positions = [p for p in positions if int(p["team"]) == team_id]
            if not team_positions:
                continue
            xs = np.asarray([p["x"] for p in team_positions], dtype=float) / 100.0
            ys = np.asarray([p["y"] for p in team_positions], dtype=float) / 100.0
            color = self._color_rgb(team_colors.get(team_id, (200, 200, 200)))
            self.pitch.scatter(
                xs, ys, ax=ax, s=90, color=color,
                edgecolors="black", linewidth=0.8, zorder=10
            )

    def _draw_passing_lanes(self, ax, result, team_colors):
        if not result or result.get("carrier") is None:
            return
        for lane in result.get("lanes", []):
            if not lane.get("clear", False) and not result.get("draw_blocked", True):
                continue
            x1, y1 = self._xy_m(lane["from"]["x"], lane["from"]["y"])
            x2, y2 = self._xy_m(lane["to"]["x"], lane["to"]["y"])
            color = self._color_rgb(
                lane.get("color_bgr", team_colors.get(int(lane["from"]["team"]), (200, 200, 200)))
            )
            style = "-" if lane.get("clear", False) else "--"
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0,
                    linestyle=style, zorder=8)
            ax.scatter([x2], [y2], s=180, facecolors="none",
                       edgecolors=color, linewidths=1.5, zorder=11)

    def render(self, positions, team_colors, ball_position=None,
               status_label=None, player_radius_cm=110.0, ball_radius_cm=55.0,
               passing_lane_result=None):
        fig, ax = self._new_canvas()
        self._draw_players(ax, positions or [], team_colors)

        if ball_position is not None and np.isfinite(ball_position.get("x", np.nan)) and np.isfinite(ball_position.get("y", np.nan)):
            bx, by = self._xy_m(ball_position["x"], ball_position["y"])
            self.pitch.scatter([bx], [by], ax=ax, s=55, color="#FFD700",
                               edgecolors="black", linewidth=0.8, zorder=12)

        self._draw_passing_lanes(ax, passing_lane_result, team_colors)

        if status_label is not None:
            ax.text(0.01, 0.99, f"Homography: {status_label}",
                    transform=ax.transAxes, va="top", ha="left",
                    color="white", fontsize=8, zorder=20)

        return self._to_image(fig)

    def render_for_panel(self, positions, team_colors, panel_height,
                         ball_position=None, status_label=None,
                         player_radius_cm=110.0, ball_radius_cm=55.0,
                         passing_lane_result=None):
        image = self.render(
            positions, team_colors, ball_position=ball_position,
            status_label=status_label,
            player_radius_cm=player_radius_cm,
            ball_radius_cm=ball_radius_cm,
            passing_lane_result=passing_lane_result,
        )
        if panel_height == self.canvas_height:
            return image
        target_width = int(round(self.canvas_width * panel_height / self.canvas_height))
        return cv2.resize(image, (target_width, int(panel_height)), interpolation=cv2.INTER_AREA)

    def _render_tracks(self, tracks, team_colors, clean=False):
        fig, ax = self._new_canvas()
        for track_id, records in tracks.items():
            usable = []
            for r in records:
                if not (math.isfinite(float(r.get("x", np.nan))) and math.isfinite(float(r.get("y", np.nan)))):
                    continue
                usable.append(r)
            if len(usable) < 2:
                continue
            team = int(usable[0]["team"])
            color = self._color_rgb(team_colors.get(team, (200, 200, 200)))
            xs = np.asarray([r["x"] for r in usable], dtype=float) / 100.0
            ys = np.asarray([r["y"] for r in usable], dtype=float) / 100.0

            if clean:
                for prev, curr in zip(usable, usable[1:]):
                    if int(curr.get("frame", 0)) - int(prev.get("frame", 0)) != 1:
                        continue
                    x0, y0 = self._xy_m(prev["x"], prev["y"])
                    x1, y1 = self._xy_m(curr["x"], curr["y"])
                    dashed = prev.get("status") == "interpolated" or curr.get("status") == "interpolated"
                    ax.plot([x0, x1], [y0, y1], color=color, linewidth=1.5,
                            linestyle="--" if dashed else "-", zorder=6)
            else:
                ax.plot(xs, ys, color=color, linewidth=1.5, zorder=6)

            ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=35, color=color,
                       edgecolors="black", linewidth=0.5, zorder=7)

        return self._to_image(fig)

    def render_trajectories(self, tracks, team_colors, track_labels=None,
                            line_thickness=2, dot_radius_cm=60.0):
        return self._render_tracks(tracks, team_colors, clean=False)

    def render_clean_trajectories(self, clean_result, team_colors,
                                  line_thickness=2, dot_radius_cm=60.0):
        return self._render_tracks(clean_result, team_colors, clean=True)