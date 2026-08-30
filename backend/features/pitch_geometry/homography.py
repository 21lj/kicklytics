import numpy as np
from .constants import PITCH_POINTS

class HomographyEstimator:

    def __init__(self, geometry_filter):
        self.geometry_filter = geometry_filter

    def estimate(self, keypoints, confidences, image_shape):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        if len(keypoints) != len(PITCH_POINTS):
            raise ValueError(
                f"Expected {len(PITCH_POINTS)} keypoints because PITCH_POINTS "
                f"contains {len(PITCH_POINTS)} mappings, got {len(keypoints)}."
            )

        valid_mask = self.geometry_filter.basic_filter(
            keypoints, confidences, image_shape
        )

        used_ids = np.flatnonzero(valid_mask)
        image_points = keypoints[valid_mask]
        pitch_points = PITCH_POINTS[valid_mask]

        if len(image_points) < 4:
            return {
                "valid": False,
                "reason": "not_enough_points",
                "num_points": len(image_points),
                "ids": used_ids.tolist(),
            }

        H, inlier_mask = self.geometry_filter.estimate_homography(
            image_points,
            pitch_points
        )

        if H is None or inlier_mask is None:
            return {
                "valid": False,
                "reason": "homography_failed",
                "ids": used_ids.tolist(),
            }

        inlier_count = int(inlier_mask.sum())

        if inlier_count < self.geometry_filter.min_inliers:
            return {
                "valid": False,
                "reason": "not_enough_inliers",
                "inliers": inlier_count,
                "ids": used_ids.tolist(),
            }

        refined_H = self.geometry_filter.refine_homography(
            image_points,
            pitch_points,
            inlier_mask
        )

        if refined_H is None:
            return {
                "valid": False,
                "reason": "refinement_failed",
                "inliers": inlier_count,
                "ids": used_ids.tolist(),
            }

        metrics = self.geometry_filter.evaluate_homography(
            refined_H,
            image_points,
            pitch_points,
            inlier_mask,
            image_shape
        )

        return {
            "valid": metrics["valid"],
            "H": refined_H,
            "inliers": inlier_count,
            "used_ids": used_ids,
            "metrics": metrics,
        }