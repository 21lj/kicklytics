import numpy as np
import supervision as sv
from tqdm import tqdm
import torch
import tempfile
import cv2

from utils.video_utils import extract_crops
from features.team_assigner import TeamClassifier, resolve_goalkeepers_teamid
from features.player_ball_assigner import PlayerBallAssigner
from features.possession_track import PossessionTracker, draw_possession
from features.passlane import HybridPassingLane

from .setup import (
    convert_video_to_h264,
    load_models,
    MAX_VIDEO_SECONDS,
    PLAYER_ID,
    BALL_ID,
    REFEREE_ID,
    GK_ID,
    DEVICE,
    team_colors,
    TEAM_RECLASSIFY_STRIDE,
    HOMOGRAPHY_STRIDE,
    PITCH_RENDER_BACKEND,
)

from features.pitch_geometry import (
    PitchGeometryFilter,
    HomographyEstimator,
    build_positions,
    project_points,
    create_pitch_renderer,
    classify_geometry_status,
)

# ================= IMPORT SPACES FIRST =================
# MUST be imported before any CUDA-related packages
try:
    import spaces
    HF_SPACES = True
except ImportError:
    HF_SPACES = False

    # Local/CPU dev has no `spaces` package. Stub it so @spaces.GPU(...) below
    # works everywhere without falling back to the old post-hoc reassignment
    # pattern (`process_video = spaces.GPU(process_video)`).
    class _SpacesStub:
        @staticmethod
        def GPU(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

    spaces = _SpacesStub()

model, pitch_model = load_models()
# ==========================================================
@spaces.GPU(duration=120)
def process_video(input_video):
    if input_video is None:
        return None, "⚠️ Please upload a video clip first."

    print("[INFO] Converting video to H.264...")
    converted_video = convert_video_to_h264(input_video)
    if converted_video is None:
        return None, "❌ Video conversion failed."

    input_video = converted_video

    try:
        video_info = sv.VideoInfo.from_video_path(input_video)
    except Exception:
        return None, "❌ Unable to open video."

    duration_seconds = video_info.total_frames / video_info.fps

    if duration_seconds > MAX_VIDEO_SECONDS:
        return None, (
            f"⚠️ Video is too long ({duration_seconds:.0f}s). "
            f"Please upload a clip under {MAX_VIDEO_SECONDS}s."
        )

    output_video = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    try:
        
        geometry_filter = PitchGeometryFilter()
        homography_estimator = HomographyEstimator(geometry_filter)

        print("[INFO] Extracting crops...")

        crops, warmup_detections = extract_crops(
            input_video,
            model,
            stride=30,
            player_id=PLAYER_ID,
            device=DEVICE
        )

        if len(crops) == 0:
            return None, "❌ No player detections found in this video."

        team_classifier = TeamClassifier(device=DEVICE)
        team_classifier.fit(crops)

        ellipse_annotator = sv.EllipseAnnotator(
            thickness=2,
            color=sv.ColorPalette.from_hex(
                ['#00BFFF', '#FF1493', '#FFD700']
            )
        )

        triangle_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex('#00FF41')
        )

        possession_annotator = sv.TriangleAnnotator(
            base=20,
            height=17,
            color=sv.Color.from_hex('#FF0000')
        )

        tracker = sv.ByteTrack()
        tracker.reset()

        ball_assigner = PlayerBallAssigner()
        possession_tracker = PossessionTracker()

        team_touches = {0: 0, 1: 0}
        last_assigned_id = None

        player_team_cache = {}
        player_seen_count = {}

        cached_H = None
        cached_status = "ACQUIRING"

        pitch_renderer = create_pitch_renderer(PITCH_RENDER_BACKEND)

        # --------------------------------------------------------
        # OUTPUT VIDEO CONFIGURATION
        # Top    : Player detection video
        # Bottom : Pitch graph video
        # --------------------------------------------------------

        panel_height = video_info.height

        combined_video_info = sv.VideoInfo(
            width=video_info.width,
            height=video_info.height * 2,
            fps=video_info.fps,
            total_frames=video_info.total_frames,
        )

        passing_lane = HybridPassingLane(
            fps=video_info.fps,
            possession_distance_cm=120.0,
            min_possession_frames=3,   # debounce: ignore <3-frame flicker before
                                        # confirming a possession transfer
            render_window_seconds=1.5,
        )

        sink = sv.VideoSink(
            output_video,
            video_info=combined_video_info
        )

        frames = sv.get_video_frames_generator(input_video)

        print("[INFO] Processing video...")

        with sink, torch.inference_mode():

            for frame_idx, frame in enumerate(
                tqdm(
                    frames,
                    total=video_info.total_frames
                )
            ):

                # --------------------------------------------------------
                # DETECTION
                # --------------------------------------------------------

                if frame_idx in warmup_detections:

                    detections = warmup_detections[frame_idx]

                else:

                    result = model.predict(
                        frame,
                        conf=0.3,
                        verbose=False,
                        device=DEVICE
                    )[0]

                    detections = sv.Detections.from_ultralytics(result)

                # --------------------------------------------------------
                # BALL DETECTION
                # --------------------------------------------------------

                ball_detections = detections[
                    detections.class_id == BALL_ID
                ]

                if len(ball_detections):

                    best_ball_idx = np.argmax(
                        ball_detections.confidence
                    )

                    ball_detections = ball_detections[
                        [best_ball_idx]
                    ]

                    ball_detections.xyxy = sv.pad_boxes(
                        xyxy=ball_detections.xyxy,
                        px=10
                    )

                # --------------------------------------------------------
                # HUMAN DETECTIONS
                # --------------------------------------------------------

                human_detections = detections[
                    detections.class_id != BALL_ID
                ]

                human_detections = human_detections.with_nms(
                    threshold=0.5,
                    class_agnostic=True
                )

                human_detections = tracker.update_with_detections(
                    detections=human_detections
                )

                player_detections = human_detections[
                    human_detections.class_id == PLAYER_ID
                ]

                gk_detections = human_detections[
                    human_detections.class_id == GK_ID
                ]

                referee_detections = human_detections[
                    human_detections.class_id == REFEREE_ID
                ]

                # --------------------------------------------------------
                # TEAM CLASSIFICATION
                # --------------------------------------------------------

                if len(player_detections):
                    idx_needing_classification = []
                    for i, tid in enumerate(player_detections.tracker_id):

                        seen = player_seen_count.get(tid, 0)

                        if (
                            tid not in player_team_cache
                            or seen % TEAM_RECLASSIFY_STRIDE == 0
                        ):
                            idx_needing_classification.append(i)

                        player_seen_count[tid] = seen + 1

                    if idx_needing_classification:

                        crops_to_classify = [
                            sv.crop_image(frame,player_detections.xyxy[i])
                            for i in idx_needing_classification
                        ]

                        predicted = team_classifier.predict(crops_to_classify)
                        for i, pred in zip(idx_needing_classification,predicted):

                            player_team_cache[
                                player_detections.tracker_id[i]
                            ] = int(pred)

                    player_detections.class_id = np.array([
                            player_team_cache[tid]
                            for tid in player_detections.tracker_id
                        ])

                # --------------------------------------------------------
                # GOALKEEPER TEAM ASSIGNMENT
                # --------------------------------------------------------

                if len(gk_detections) and len(player_detections):

                    gk_detections.class_id = resolve_goalkeepers_teamid(
                        player_detections,
                        gk_detections
                    )

                elif len(gk_detections):

                    gk_detections = sv.Detections.empty()

                # --------------------------------------------------------
                # POSSESSION TRACKING
                # --------------------------------------------------------

                player_boxes = (list(player_detections.xyxy) + list(gk_detections.xyxy))

                team_ids = (
                    [
                        int(c) + 1
                        for c in player_detections.class_id
                    ] + [
                        int(c) + 1
                        for c in gk_detections.class_id
                    ])

                ball_bbox = (

                    ball_detections.xyxy[0]

                    if len(ball_detections) > 0

                    else None
                )

                state = possession_tracker.update(
                    ball_bbox,
                    player_boxes,
                    team_ids,
                    frame_number=frame_idx,
                    fps=video_info.fps
                )

                # --------------------------------------------------------
                # BALL POSSESSION ASSIGNMENT
                # --------------------------------------------------------

                assigned_id = -1

                possession_detections = sv.Detections.empty()

                if len(ball_detections):

                    ball_bbox = ball_detections.xyxy[0]
                    candidates = sv.Detections.merge([
                            player_detections,
                            gk_detections
                        ])

                    players_dict = {
                        tid: {"bbox": bbox}

                        for tid, bbox in zip(
                            candidates.tracker_id,
                            candidates.xyxy
                        )
                    }

                    assigned_id = (ball_assigner.assign_ball_to_player(players_dict,ball_bbox))

                    if assigned_id != -1:

                        possession_detections = candidates[candidates.tracker_id == assigned_id]

                        if assigned_id != last_assigned_id:

                            team_id = int(possession_detections.class_id[0])

                            if team_id in team_touches:

                                team_touches[team_id] += 1

                            last_assigned_id = assigned_id

                # --------------------------------------------------------
                # DRAW PLAYER DETECTION VIDEO
                # --------------------------------------------------------

                if len(referee_detections):

                    referee_detections.class_id -= 1

                team_detection = sv.Detections.merge([
                        player_detections,
                        gk_detections,
                        referee_detections
                    ])

                annotated = frame.copy()

                annotated = ellipse_annotator.annotate(
                    annotated,
                    team_detection
                )

                annotated = triangle_annotator.annotate(
                    annotated,
                    ball_detections
                )

                annotated = draw_possession(
                    annotated,
                    state,
                    team_colors
                )

                annotated = possession_annotator.annotate(
                    annotated,
                    possession_detections
                )

                # --------------------------------------------------------
                # PITCH HOMOGRAPHY
                # --------------------------------------------------------

                if frame_idx % HOMOGRAPHY_STRIDE == 0:

                    pitch_result = pitch_model(
                        frame,
                        verbose=True
                    )[0]

                    if (pitch_result.keypoints is not None and len(pitch_result.keypoints.xy)):

                        kps = (
                            pitch_result
                            .keypoints
                            .xy[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        kconf = (
                            pitch_result
                            .keypoints
                            .conf[0]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        est = homography_estimator.estimate(
                            kps,
                            kconf,
                            frame.shape
                        )

                        if est["valid"]:
                            cached_H = est["H"]
                            cached_status = (classify_geometry_status(est["metrics"]))

                # --------------------------------------------------------
                # BUILD CANONICAL PLAYER POSITIONS
                # --------------------------------------------------------

                boxes = (list(player_detections.xyxy)+list(gk_detections.xyxy))

                proj_team_ids = ([
                        int(c) + 1
                        for c in player_detections.class_id
                    ]+[
                        int(c) + 1
                        for c in gk_detections.class_id
                    ])

                proj_tracker_ids = (
                    list(player_detections.tracker_id)
                    +
                    list(gk_detections.tracker_id)
                )

                positions = build_positions(
                    cached_H,
                    boxes,
                    proj_team_ids,
                    proj_tracker_ids
                )

                # --------------------------------------------------------
                # BALL POSITION ON PITCH
                # --------------------------------------------------------

                ball_position = None

                if (len(ball_detections)and cached_H is not None):

                    bx1, by1, bx2, by2 = (ball_detections.xyxy[0])

                    ball_img_pt = np.array([
                            [
                                (bx1 + bx2) / 2.0,
                                (by1 + by2) / 2.0
                            ]
                        ])

                    bx, by = project_points(
                        cached_H,
                        ball_img_pt
                    )[0]

                    ball_position = {
                        "x": float(bx),
                        "y": float(by)
                    }

                # --------------------------------------------------------
                # PASSING LANE ANALYSIS
                # --------------------------------------------------------

                passing_lane.update(
                    frame_idx,
                    positions,
                    ball_position,
                    observed_carrier_id=assigned_id,
                )

                passing_lane_result = (
                    passing_lane.get_render_result(
                        frame_idx,
                        team_colors,
                    ))

                # --------------------------------------------------------
                # RENDER PITCH GRAPH
                # --------------------------------------------------------

                pitch_panel = (
                    pitch_renderer.render_for_panel(
                        positions,
                        team_colors,
                        panel_height,
                        ball_position=ball_position,
                        status_label=cached_status,
                        passing_lane_result=passing_lane_result,
                    )
                )

                # Ensure pitch graph has exactly the same dimensions
                # as the player detection video.
                pitch_panel = cv2.resize(
                    pitch_panel,
                    (video_info.width,
                     video_info.height),
                    interpolation=cv2.INTER_AREA)

                # --------------------------------------------------------
                # VERTICAL VIDEO COMBINATION
                #
                # TOP    → Player Detection Video
                # BOTTOM → Pitch Graph Video
                # --------------------------------------------------------

                combined = np.vstack([
                        annotated,
                        pitch_panel
                    ])

                sink.write_frame(combined)

        # --------------------------------------------------------
        # MATCH STATISTICS
        # --------------------------------------------------------

        pass_stats = passing_lane.get_statistics()["team"]

        t1 = pass_stats.get(1, {})
        t2 = pass_stats.get(2, {})

        stats_text = f"""
            ### 📊 Match Statistics

            | Metric | Team 1 (Blue) | Team 2 (Pink) |
            |--------|---------------|---------------|
            | **Ball Touches** | {team_touches[0]} | {team_touches[1]} |
            | **Possession %** | {state.team1_percentage:.1f}% | {state.team2_percentage:.1f}% |
            | **Pass Attempts** | {t1.get('attempts', 0)} | {t2.get('attempts', 0)} |
            | **Successful Passes** | {t1.get('successful', 0)} | {t2.get('successful', 0)} |
            | **Intercepted Passes** | {t1.get('intercepted', 0)} | {t2.get('intercepted', 0)} |
            | **Pass Success Rate** | {t1.get('success_rate', 0.0):.1f}% | {t2.get('success_rate', 0.0):.1f}% |
            | **Clear Lanes** | {t1.get('clear', 0)} | {t2.get('clear', 0)} |
            | **Contested Lanes** | {t1.get('contested', 0)} | {t2.get('contested', 0)} |
            | **Blocked Lanes** | {t1.get('blocked', 0)} | {t2.get('blocked', 0)} |

            **Total Events Tracked:** {sum(team_touches.values())}
            """

        print("[INFO] Passing report:")
        print(passing_lane.format_report_text())

        return output_video, stats_text

    except Exception as e:

        import traceback

        print(traceback.format_exc())

        return None, (f"❌ Video processing failed: {str(e)}")