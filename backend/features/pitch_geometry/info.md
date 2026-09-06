# Pitch Geometry & Projection Module

This module is responsible for converting football pitch keypoints detected from a broadcast/video frame into a **canonical top-down representation of the football pitch**.

It handles the complete pitch-geometry pipeline:

```text
Detected Pitch Keypoints
        │
        ▼
Keypoint Filtering
        │
        ▼
Homography Estimation
        │
        ▼
Homography Validation
        │
        ▼
Player Position Projection
        │
        ▼
Canonical Pitch Coordinates
        │
        ├───────────────┐
        ▼               ▼
 OpenCV Renderer    MplSoccer Renderer
        │               │
        └───────┬───────┘
                ▼
        Top-Down Pitch View
```

The module is intentionally separated from **player detection, tracking, and team classification**. Its responsibility is limited to pitch geometry, coordinate transformation, and pitch visualization.

---

## Purpose

A football broadcast video provides player locations in **image/pixel coordinates**. These coordinates are not directly useful for tactical analysis because the camera perspective changes the apparent position and distance between players.

This module solves that problem by estimating a **homography matrix (`H`)** between:

* the original camera/image coordinate system, and
* a fixed canonical football-pitch coordinate system.

The canonical pitch uses centimetres:

```text
Pitch Length = 12000 cm
Pitch Width  = 7000 cm
```

Once the homography is estimated, player positions can be transformed from the broadcast image into their corresponding locations on the top-down pitch.

---

# Module Responsibilities

The module has five main responsibilities:

### 1. Pitch Keypoint Filtering

Detected pitch keypoints are checked for validity before being used for homography estimation.

The filtering checks:

* confidence threshold
* finite coordinates
* image boundaries
* correct keypoint/confidence alignment

This prevents invalid detections from contaminating the homography calculation.

---

### 2. Homography Estimation

The valid pitch keypoints are matched against a predefined set of canonical pitch landmarks.

OpenCV's RANSAC-based homography estimation is used to calculate:

```text
Image Coordinates → Canonical Pitch Coordinates
```

The process also identifies which detected points are considered geometric inliers.

The homography is then refined using the detected inlier points.

---

### 3. Homography Validation

A homography is not automatically considered reliable simply because OpenCV successfully calculated one.

The module evaluates the estimated homography using metrics such as:

* number of inliers
* mean reprojection error
* median reprojection error
* 90th percentile reprojection error
* maximum reprojection error
* spatial coverage of the detected points

The geometry can then be classified as:

```text
ACCEPTED
WEAK
REJECTED
```

`WEAK` geometry is still allowed to render; the status is used as a diagnostic/display label rather than acting as a hard pipeline blocker.

---

### 4. Player Projection

Player bounding boxes are converted into a ground-contact point using the **bottom-center of the bounding box**.

```text
Player Bounding Box

┌─────────────┐
│             │
│    Player   │
│             │
└───────●─────┘
        ↑
    Feet Point
```

The feet point is used instead of the bounding-box center because the homography represents the football pitch ground plane.

The resulting point is projected through the homography:

```text
Image-space feet point
          │
          ▼
       Homography
          │
          ▼
Canonical pitch (x, y)
```

The output is represented as:

```python
{
    "track_id": ...,
    "team": ...,
    "x": ...,
    "y": ...
}
```

The coordinates are always expressed in **canonical pitch centimetres**.

---

# Canonical Pitch Coordinate System

The entire geometry pipeline uses one fixed coordinate system:

```text
(0, 0) ───────────────────────────────► (12000, 0)
  │                                      │
  │                                      │
  │             FOOTBALL PITCH           │
  │                                      │
  │                                      │
  ▼                                      ▼
(0, 7000) ─────────────────────────── (12000, 7000)
```

This canonical representation is important because it makes player positions independent of:

* video resolution
* camera resolution
* renderer resolution
* visualization backend

The homography itself always operates in canonical pitch centimetres.

---

# Visualization

The module supports two visualization backends.

## OpenCV Renderer

The OpenCV renderer provides a lightweight pitch visualization using `cv2`.

It renders:

* pitch markings
* players
* team-specific colors
* ball position
* homography status
* passing lanes
* trajectories
* cleaned trajectories

The renderer uses a **fixed canonical canvas size** so that marker sizes remain consistent between different source videos.

---

## MplSoccer Renderer

The `mplsoccer` renderer provides an alternative football-pitch visualization.

It receives exactly the same canonical pitch coordinates as the OpenCV renderer.

The only conversion performed at the visualization boundary is:

```text
centimetres → metres
```

For example:

```text
6000 cm → 60 m
7000 cm → 70 m
```

The homography itself is never modified for the renderer.

This allows the visualization backend to change without changing the underlying geometry pipeline.

---

# Modular Structure

The original implementation can be separated into smaller modules based on responsibility:

```text
pitch_geometry/
│
├── constants.py
│      └── PITCH_POINTS
│
├── geometry_filter.py
│      └── PitchGeometryFilter
│
├── homography.py
│      └── HomographyEstimator
│
├── projection.py
│      ├── feet_point
│      ├── project_points
│      ├── build_positions
│      ├── homography_is_consistent
│      └── classify_geometry_status
│
├── opencv_renderer.py
│      └── PitchRenderer
│
├── mplsoccer_renderer.py
│      └── MplSoccerRenderer
│
├── renderer_factory.py
│      └── create_pitch_renderer
│
└── __init__.py
```

---

## `constants.py`

Contains the canonical football-pitch configuration.

```text
PITCH_LENGTH_CM
PITCH_WIDTH_CM
PITCH_POINTS
```

`PITCH_POINTS` contains the 32 canonical pitch landmarks used by the homography system.

---

## `geometry_filter.py`

Contains:

```python
PitchGeometryFilter
```

Responsible for:

* filtering invalid keypoints
* estimating the initial homography
* refining the homography
* calculating reprojection errors
* calculating spatial coverage
* evaluating homography quality

---

## `homography.py`

Contains:

```python
HomographyEstimator
```

Responsible for orchestrating the homography-estimation process.

It connects the detected keypoints to the `PitchGeometryFilter` and produces the final homography result and associated metrics.

---

## `projection.py`

Contains common geometry/projection functions:

```python
feet_point()
project_points()
build_positions()
homography_is_consistent()
classify_geometry_status()
```

These functions operate on geometry data and do not depend on a particular visualization backend.

---

## `renderer.py`

Contains:

```python
PitchRenderer
```

Responsible for OpenCV-based pitch visualization.

It receives already-projected canonical coordinates and renders them onto a top-down pitch.

---

## `mplsoccer_renderer.py`

Contains:

```python
MplSoccerRenderer
```

Responsible for `mplsoccer`-based visualization.

It uses the same canonical geometry as the OpenCV renderer.

---

## `renderer_factory.py`

Contains:

```python
create_pitch_renderer()
```

This provides a single entry point for selecting the visualization backend.

Example:

```python
renderer = create_pitch_renderer("opencv")
```

or:

```python
renderer = create_pitch_renderer("mplsoccer")
```

The rest of the pipeline does not need to know which renderer is being used.

---

# Architecture

The important design principle is the separation between **geometry** and **visualization**.

```text
                    ┌──────────────────────┐
                    │  Pitch Keypoint      │
                    │  Detection           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ PitchGeometryFilter  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ HomographyEstimator  │
                    └──────────┬───────────┘
                               │
                               ▼
                         Homography H
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Projection       │
                    │ feet → pitch (x,y)   │
                    └──────────┬───────────┘
                               │
                               ▼
                  Canonical Pitch Coordinates
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             OpenCV Renderer       MplSoccer Renderer
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Pitch Visualization
```

---

# Important Design Principle

The most important contract in this module is:

> **Geometry produces canonical pitch coordinates. Renderers only visualize those coordinates.**

In other words:

```text
                GEOMETRY
                   │
                   │
                   ▼
          Canonical x / y (cm)
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
     OpenCV                mplsoccer
     renderer               renderer
```

The renderer should never need:

* raw player bounding boxes
* video resolution
* pitch keypoint detections
* homography estimation logic

This keeps the geometry pipeline independent of visualization.

---

# Data Flow

A typical frame follows this process:

```text
1. Detect pitch keypoints
             ↓
2. Filter invalid/low-confidence keypoints
             ↓
3. Estimate homography using RANSAC
             ↓
4. Refine homography using inliers
             ↓
5. Evaluate homography quality
             ↓
6. Determine geometry status
             ↓
7. Extract player feet points
             ↓
8. Project feet points through H
             ↓
9. Obtain canonical pitch positions
             ↓
10. Render players/ball/passing lanes
             ↓
11. Display or composite pitch panel
```

---

# Renderer Independence

Both rendering implementations consume the same logical data:

```python
positions
team_colors
ball_position
passing_lane_result
status_label
```

Therefore:

```text
                  Same Geometry
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       OpenCV Output       MplSoccer Output
```

The visualization can change without changing the underlying player coordinates.

---

# Trajectory Visualization

The renderer also supports player trajectories.

A trajectory is represented as a sequence of canonical pitch positions:

```text
Track ID
   │
   ├── frame 1 → (x, y)
   ├── frame 2 → (x, y)
   ├── frame 3 → (x, y)
   ├── frame 4 → (x, y)
   └── ...
```

The system supports both:

* raw trajectories
* cleaned/status-aware trajectories

Clean trajectories can distinguish between:

```text
valid
interpolated
id_switch_suspect
jump_rejected
```

This allows trajectory visualization to communicate the quality of the underlying tracking data.

---

# Passing Lane Visualization

Passing-lane results are also rendered using canonical pitch coordinates.

A lane contains:

```text
from → to
```

and can be classified as:

```text
clear
blocked
```

Clear lanes are rendered as solid lines, while blocked lanes can be rendered as dashed lines.

The important point is that passing-lane visualization happens **after** the geometry stage, so the lane coordinates are already expressed in canonical pitch space.

---

# Design Goal

The overall goal of this module is to provide a stable geometry layer between the football-video processing pipeline and visualization.

```text
Detection / Tracking
        │
        ▼
┌──────────────────────────┐
│   Pitch Geometry Layer   │
│                          │
│  Keypoints               │
│      ↓                   │
│  Homography              │
│      ↓                   │
│  Projection              │
│      ↓                   │
│  Canonical Coordinates   │
└────────────┬─────────────┘
             │
             ▼
       Visualization
```

This means the rest of the application can work with a consistent representation of the football pitch regardless of the original camera resolution or visualization backend.

---

## Non-Goals

This module does **not** own:

* player detection
* player tracking
* team classification
* ball detection
* team assignment

Those components remain outside the pitch-geometry layer.

The module assumes that player boxes, team IDs, tracker IDs, and other required inputs are provided by the upstream pipeline.

---

## Summary

The pitch-geometry module converts perspective-based football video information into a consistent top-down pitch representation.

> **All geometry is calculated in canonical pitch centimetres, and visualization backends only convert/render that canonical state.**
