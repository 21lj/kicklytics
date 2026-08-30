import numpy as np

PITCH_RENDER_BACKEND = "opencv"
PITCH_LENGTH_CM = 12000.0
PITCH_WIDTH_CM = 7000.0

# Standard 32-landmark canonical pitch geometry, in centimetres.
# The ordering is intentionally preserved as the model's keypoint index order.
# It matches the 32-vertex ordering used by Roboflow's SoccerPitchConfiguration.
PITCH_POINTS = np.array([
    # 0 - 5: left touchline / penalty-area landmarks
    [0, 0],
    [0, 1450],
    [0, 2584],
    [0, 4416],
    [0, 5550],
    [0, 7000],
    # 6 - 12: left goal/penalty-area interior landmarks
    [550, 2584],
    [550, 4416],
    [1100, 3500],
    [2015, 1450],
    [2015, 2584],
    [2015, 4416],
    [2015, 5550],
    # 13 - 16: halfway line / centre-circle landmarks
    [6000, 0],
    [6000, 2585],
    [6000, 4415],
    [6000, 7000],
    # 17 - 24: right penalty-area landmarks
    [9985, 1450],
    [9985, 2584],
    [9985, 4416],
    [9985, 5550],
    [10900, 3500],
    [11450, 2584],
    [11450, 4416],
    # 25 - 29: right touchline / penalty-area landmarks
    [12000, 0],
    [12000, 1450],
    [12000, 2584],
    [12000, 4416],
    [12000, 5550],
    [12000, 7000],
    # 30 - 31: centre-circle horizontal extrema
    [5085, 3500],
    [6915, 3500],
], dtype=np.float32)