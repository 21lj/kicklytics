import numpy as np
import cv2

# ============================================================
# HELPERS: image-space points needed for projection
# ============================================================

def feet_point(box):
    """Bottom-center of a bbox — the ground-contact point homography is
    valid for (box centers are NOT on the ground plane)."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


def project_points(H, image_pts):
    """image_pts: (N, 2) -> pitch-cm (N, 2). Returns NaNs if H is None."""
    image_pts = np.asarray(image_pts, dtype=np.float32)
    if H is None or len(image_pts) == 0:
        return np.full((len(image_pts), 2), np.nan, dtype=np.float32)
    pts = image_pts.reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)

# ============================================================
# GEOMETRY STAGE OUTPUT (Step 2 — pure canonical coords, no rendering)
# ============================================================

def build_positions(H, boxes, team_ids, tracker_ids=None):
    """Geometry-stage output: image boxes -> canonical pitch-cm (x, y) per
    player. No visualization concerns here — just track_id/team/x/y, per
    prompt.md Step 2. Returns list[dict]. Requires feet_point/project_points
    (defined below) — pure geometry, independent of render scale."""
    if len(boxes) == 0:
        return []

    feet_pts = [feet_point(b) for b in boxes]
    pitch_pts = project_points(H, feet_pts)

    if tracker_ids is None:
        tracker_ids = [None] * len(boxes)

    positions = []
    for (x, y), team, tid in zip(pitch_pts, team_ids, tracker_ids):
        positions.append({
            "track_id": None if tid is None else int(tid),
            "team": int(team),
            "x": float(x),
            "y": float(y),
        })
    return positions

def homography_is_consistent(
    old_H,
    new_H,
    image_shape,
    max_pitch_displacement_cm=250.0,
):
    """
    Compare two homographies by projecting representative image points.

    Returns:
        (True/False, max_displacement_cm, median_displacement_cm)
    """

    if old_H is None:
        return True, 0.0, 0.0

    h, w = image_shape[:2]

    # Representative image locations.
    # Avoid only using the center because bad homographies can pivot
    # around the middle while being wildly wrong near the pitch edges.
    sample_points = np.array([
        [0.10 * w, 0.15 * h],
        [0.50 * w, 0.15 * h],
        [0.90 * w, 0.15 * h],
        [0.10 * w, 0.50 * h],
        [0.50 * w, 0.50 * h],
        [0.90 * w, 0.50 * h],
        [0.10 * w, 0.85 * h],
        [0.50 * w, 0.85 * h],
        [0.90 * w, 0.85 * h],
    ], dtype=np.float32).reshape(-1, 1, 2)

    try:
        old_proj = cv2.perspectiveTransform(sample_points, old_H)
        new_proj = cv2.perspectiveTransform(sample_points, new_H)
    except cv2.error:
        return False, float("inf"), float("inf")

    old_proj = old_proj.reshape(-1, 2)
    new_proj = new_proj.reshape(-1, 2)

    displacement = np.linalg.norm(new_proj - old_proj, axis=1)

    max_disp = float(np.max(displacement))
    median_disp = float(np.median(displacement))

    return (
        max_disp <= max_pitch_displacement_cm,
        max_disp,
        median_disp,
    )


def classify_geometry_status(metrics, min_inliers=6, min_coverage=0.05,
                              max_mean_error_cm=25.0):
    """Cheap ACCEPTED / WEAK label for on-frame display. Not a hard gate —
    weak geometry still renders, just flagged, so pipeline never blocks."""
    if not metrics.get("valid"):
        return "REJECTED"
    ok = (
        metrics["inliers"] >= min_inliers
        and metrics["coverage"] >= min_coverage
        and metrics["mean_error_cm"] <= max_mean_error_cm
    )
    return "ACCEPTED" if ok else "WEAK"

