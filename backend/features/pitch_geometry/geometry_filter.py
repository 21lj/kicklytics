import cv2
import numpy as np


class PitchGeometryFilter:

    def __init__(
        self,
        confidence_threshold=0.45,
        ransac_threshold=75.0,
        min_inliers=6,
        min_coverage=0.05,
    ):
        self.confidence_threshold = confidence_threshold
        self.ransac_threshold = ransac_threshold
        self.min_inliers = min_inliers
        self.min_coverage = min_coverage

    def basic_filter(self, keypoints, confidences, image_shape):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        confidences = np.asarray(confidences, dtype=np.float32)

        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"Expected keypoints shape (N, 2), got {keypoints.shape}")
        if len(keypoints) != len(confidences):
            raise ValueError(
                f"Keypoint/confidence mismatch: {len(keypoints)} vs {len(confidences)}"
            )

        h, w = image_shape[:2]
        valid = np.ones(len(keypoints), dtype=bool)
        valid &= confidences >= self.confidence_threshold
        valid &= np.isfinite(keypoints).all(axis=1)
        valid &= keypoints[:, 0] > 1
        valid &= keypoints[:, 0] < w - 1
        valid &= keypoints[:, 1] > 1
        valid &= keypoints[:, 1] < h - 1
        return valid

    def estimate_homography(self, image_points, pitch_points):
        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)

        if len(image_points) != len(pitch_points):
            raise ValueError(
                "image_points and pitch_points must have identical length: "
                f"{len(image_points)} != {len(pitch_points)}"
            )
        if len(image_points) < 4:
            return None, None

        H, mask = cv2.findHomography(
            image_points, pitch_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=5000,
            confidence=0.995,
        )
        if H is None or mask is None:
            return None, None
        return H, mask.ravel().astype(bool)

    def refine_homography(self, image_points, pitch_points, inlier_mask):
        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        inlier_mask = np.asarray(inlier_mask, dtype=bool)

        if len(image_points) != len(pitch_points) or len(image_points) != len(inlier_mask):
            raise ValueError("Point/mask arrays are not aligned.")

        inlier_image = image_points[inlier_mask]
        inlier_pitch = pitch_points[inlier_mask]
        if len(inlier_image) < 4:
            return None

        refined_H, _ = cv2.findHomography(inlier_image, inlier_pitch, method=0)
        return refined_H

    def reprojection_error(self, H, image_points, pitch_points):
        if H is None:
            raise ValueError("H cannot be None.")

        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        if len(image_points) != len(pitch_points):
            raise ValueError("image_points and pitch_points are not aligned.")
        if len(image_points) == 0:
            return np.empty(0, dtype=np.float32)

        projected = cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        return np.linalg.norm(projected - pitch_points, axis=1)

    def spatial_coverage(self, points, image_shape):
        points = np.asarray(points, dtype=np.float32)
        if len(points) < 4:
            return 0.0
        h, w = image_shape[:2]
        x, y = points[:, 0], points[:, 1]
        return float(((x.max() - x.min()) / w) * ((y.max() - y.min()) / h))

    def evaluate_homography(self, H, image_points, pitch_points, inlier_mask, image_shape):
        if H is None:
            return {"valid": False, "reason": "no_homography"}

        image_points = np.asarray(image_points, dtype=np.float32)
        pitch_points = np.asarray(pitch_points, dtype=np.float32)
        inlier_mask = np.asarray(inlier_mask, dtype=bool)

        inlier_count = int(inlier_mask.sum())
        if inlier_count < 4:
            return {"valid": False, "reason": "too_few_inliers", "inliers": inlier_count}

        inlier_image = image_points[inlier_mask]
        inlier_pitch = pitch_points[inlier_mask]
        errors = self.reprojection_error(H, inlier_image, inlier_pitch)
        coverage = self.spatial_coverage(inlier_image, image_shape)

        return {
            "valid": True,
            "inliers": inlier_count,
            "mean_error_cm": float(errors.mean()),
            "median_error_cm": float(np.median(errors)),
            "p90_error_cm": float(np.percentile(errors, 90)),
            "max_error_cm": float(errors.max()),
            "coverage": coverage,
        }