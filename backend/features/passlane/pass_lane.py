import math


class HybridPassingLane:
    """
    Tracks the current ball carrier. When possession moves to a DIFFERENT
    player on the SAME team, draws a solid line from the previous carrier
    to the new carrier, in that team's configured color. That is the whole
    feature: no interception classification, no blocker/lane-quality
    geometry, no hypothetical passing options, no pass-event state machine.

    Debouncing: a raw per-frame carrier id must hold the ball for
    `min_possession_frames` in a row before it can become (or replace) the
    confirmed carrier. This is the only stabilization needed to ignore
    one-frame flicker, brief tracking dropouts, and the carrier briefly
    disappearing and reappearing as the same player.

    Expected player fields (canonical pitch cm, same as PitchRenderer):
        {"track_id": int, "team": 1 or 2, "x": float, "y": float}
    """

    def __init__(
        self,
        fps=30.0,
        possession_distance_cm=120.0,   # kept for signature/call-site compatibility;
                                         # carrier id already comes from the existing
                                         # ball-possession assigner, not recomputed here
        min_possession_frames=3,        # debounce: frames a new carrier must hold the
                                         # ball before a transfer is confirmed
        render_window_seconds=1.5,      # how long a drawn lane stays on screen
    ):
        self.fps = float(fps) if fps else 30.0
        self.possession_distance_cm = float(possession_distance_cm)
        self.min_possession_frames = max(1, int(min_possession_frames))
        self.render_window_frames = max(1, int(round(render_window_seconds * self.fps)))

        self.carrier = None       # confirmed stable carrier: {"track_id","team","x","y"}
        self.carrier_id = None    # raw candidate id currently being debounced
        self.carrier_streak = 0   # consecutive frames the raw candidate has held the ball

        self.lanes = []           # completed same-team transfers: {"from","to","team","frame"}

    # -------------------- validation --------------------

    @staticmethod
    def _valid_player(p):
        try:
            return (
                isinstance(p, dict)
                and p.get("track_id") is not None
                and p.get("team") in (1, 2)
                and math.isfinite(float(p.get("x")))
                and math.isfinite(float(p.get("y")))
            )
        except (TypeError, ValueError):
            return False

    @classmethod
    def _find_player(cls, positions, track_id):
        if track_id is None:
            return None
        for p in positions or []:
            if cls._valid_player(p) and p.get("track_id") == track_id:
                return p
        return None

    # -------------------- state --------------------

    def update(self, frame_idx, positions, ball_position, observed_carrier_id=None):
        """observed_carrier_id: raw per-frame carrier id from the existing
        ball-possession assigner. -1 or None both mean "no carrier this
        frame". ball_position is accepted for call-site compatibility but
        isn't needed here — possession is already resolved upstream."""
        raw_id = None if observed_carrier_id in (None, -1) else observed_carrier_id
        player = self._find_player(positions, raw_id)
        # only trust an id we can actually resolve to a valid player this frame
        raw_id = player["track_id"] if player is not None else None

        if raw_id == self.carrier_id:
            self.carrier_streak += 1
        else:
            self.carrier_id = raw_id
            self.carrier_streak = 1

        if raw_id is None or self.carrier_streak < self.min_possession_frames:
            return  # no confirmed carrier yet this frame; keep previous state

        if self.carrier is not None and self.carrier["track_id"] == raw_id:
            return  # same player still has it — nothing happened

        new_carrier = {
            "track_id": player["track_id"],
            "team": player["team"],
            "x": float(player["x"]),
            "y": float(player["y"]),
        }

        if self.carrier is not None and self.carrier["team"] == new_carrier["team"]:
            # confirmed same-team transfer -> draw a lane
            self.lanes.append({
                "from": {"track_id": self.carrier["track_id"], "x": self.carrier["x"], "y": self.carrier["y"]},
                "to": {"track_id": new_carrier["track_id"], "x": new_carrier["x"], "y": new_carrier["y"]},
                "team": new_carrier["team"],
                "frame": frame_idx,
            })
        # else: first-ever carrier, or possession changed teams -> no lane,
        # just adopt the new carrier

        self.carrier = new_carrier

    # -------------------- outputs --------------------

    def get_passes(self):
        return list(self.lanes)

    def get_statistics(self):
        team_stats = {1: {"attempts": 0, "successful": 0}, 2: {"attempts": 0, "successful": 0}}
        for lane in self.lanes:
            t = lane["team"]
            if t in team_stats:
                team_stats[t]["attempts"] += 1
                team_stats[t]["successful"] += 1
        return {"team": team_stats}

    def format_report_text(self):
        stats = self.get_statistics()["team"]
        lines = []
        for team in (1, 2):
            lines.append(f"Team {team}")
            lines.append("------")
            lines.append(f"Same-team passes: {stats[team]['successful']}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def get_render_result(self, frame_idx, team_colors, window_frames=None):
        """Same renderer-neutral schema PitchRenderer.draw_passing_lanes already
        consumes: a solid team-colored line + receiver marker, shown for
        `render_window_seconds` after the transfer happened."""
        window = self.render_window_frames if window_frames is None else int(window_frames)
        lanes = []
        for lane in reversed(self.lanes):
            if frame_idx - lane["frame"] > window:
                break
            lanes.append({
                "from": lane["from"],
                "to": lane["to"],
                "clear": True,
                "color_bgr": team_colors.get(lane["team"], (255, 255, 255)),
            })
        if not lanes:
            return None
        return {"carrier": {"placeholder": True}, "lanes": lanes, "draw_blocked": True}
