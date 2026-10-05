import cv2
import math
import tempfile
import time
from pathlib import Path

import mediapipe as mp
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Push-Up Counter",
    page_icon="💪",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.75;
        margin-bottom: 20px;
    }

    .metric-box {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.25);
        text-align: center;
        background: rgba(128,128,128,.08);
    }

    .big-number {
        font-size: 42px;
        font-weight: 800;
    }

    .small-label {
        font-size: 14px;
        opacity: .7;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">💪 AI Push-Up Counter</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Upload a side-view push-up video and watch AI count your repetitions.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    min_visibility = st.slider(
        "Minimum landmark visibility",
        0.30,
        0.90,
        0.50,
        0.05,
    )

    down_angle = st.slider(
        "DOWN position angle",
        70,
        110,
        90,
    )

    up_angle = st.slider(
        "UP position angle",
        140,
        175,
        155,
    )

    process_every = st.slider(
        "Process every Nth frame",
        1,
        3,
        1,
    )


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"


if not MODEL_PATH.exists():

    st.error(
        "❌ Pose model not found.\n\n"
        "Please place **pose_landmarker_lite.task** "
        "in the same folder as app.py."
    )

    st.code(
        """
PushUp Counter/
│
├── app.py
├── pose_landmarker_lite.task
└── .venv/
        """
    )

    st.stop()


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded = st.file_uploader(
    "🎥 Upload push-up video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
    ],
)


if not uploaded:

    st.info(
        "Upload a side-view push-up video to start. "
        "Best results: full body visible, good lighting, "
        "and fixed camera."
    )

    st.stop()


# =========================================================
# SAVE TEMP VIDEO
# =========================================================

suffix = Path(uploaded.name).suffix or ".mp4"

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix,
) as tmp:

    tmp.write(uploaded.getbuffer())
    input_path = tmp.name


# =========================================================
# OPEN VIDEO
# =========================================================

cap = cv2.VideoCapture(input_path)


if not cap.isOpened():

    st.error(
        "❌ Could not open this video. "
        "Try MP4 (H.264)."
    )

    st.stop()


# =========================================================
# VIDEO INFORMATION
# =========================================================

fps = cap.get(cv2.CAP_PROP_FPS)

if not fps or math.isnan(fps) or fps <= 0:
    fps = 30.0


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
)


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 960
)


height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 540
)


# =========================================================
# STREAMLIT UI
# =========================================================

preview = st.empty()

c1, c2, c3 = st.columns(3)

count_box = c1.empty()
state_box = c2.empty()
progress_box = c3.empty()


# =========================================================
# STAT DISPLAY
# =========================================================

def show_stats(count, state, progress):

    count_box.markdown(
        f"""
        <div class="metric-box">

            <div class="small-label">
                PUSH-UPS
            </div>

            <div class="big-number">
                {count}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    state_box.markdown(
        f"""
        <div class="metric-box">

            <div class="small-label">
                STATE
            </div>

            <div class="big-number"
                 style="font-size:30px">

                {state}

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    progress_box.progress(
        min(max(progress, 0.0), 1.0),
        text="Processing video",
    )


# =========================================================
# MEDIAPIPE 1.0.1 TASKS API
# =========================================================

try:

    BaseOptions = mp.tasks.BaseOptions

    PoseLandmarker = mp.tasks.vision.PoseLandmarker

    PoseLandmarkerOptions = (
        mp.tasks.vision.PoseLandmarkerOptions
    )

    VisionRunningMode = (
        mp.tasks.vision.RunningMode
    )

except Exception as e:

    st.error(
        "❌ MediaPipe Tasks API could not be loaded."
    )

    st.exception(e)

    st.stop()


# =========================================================
# ANGLE FUNCTION
# =========================================================

def calculate_angle(a, b, c):

    """
    Calculate angle ABC in degrees.
    """

    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = (
        ax - bx,
        ay - by,
    )

    bc = (
        cx - bx,
        cy - by,
    )

    dot = (
        ba[0] * bc[0]
        +
        ba[1] * bc[1]
    )

    mag = (
        math.hypot(*ba)
        *
        math.hypot(*bc)
    )

    if mag == 0:

        return None

    value = max(
        -1.0,
        min(
            1.0,
            dot / mag
        )
    )

    return math.degrees(
        math.acos(value)
    )


# =========================================================
# LANDMARK HELPER
# =========================================================

def get_landmark(landmarks, index):

    lm = landmarks[index]

    return (
        lm.x,
        lm.y
    )


# =========================================================
# POSE ANGLE EXTRACTION
# =========================================================

def get_pose_angles(result):

    if not result.pose_landmarks:

        return None

    if len(result.pose_landmarks) == 0:

        return None

    landmarks = result.pose_landmarks[0]

    # MediaPipe Pose landmark indexes
    NOSE = 0

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14

    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    LEFT_HIP = 23
    RIGHT_HIP = 24

    needed = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,

        LEFT_ELBOW,
        RIGHT_ELBOW,

        LEFT_WRIST,
        RIGHT_WRIST,

        LEFT_HIP,
        RIGHT_HIP,
    ]

    # Check visibility
    for index in needed:

        visibility = getattr(
            landmarks[index],
            "visibility",
            1.0
        )

        if visibility < min_visibility:

            return None

    left_shoulder = get_landmark(
        landmarks,
        LEFT_SHOULDER
    )

    right_shoulder = get_landmark(
        landmarks,
        RIGHT_SHOULDER
    )

    left_elbow = get_landmark(
        landmarks,
        LEFT_ELBOW
    )

    right_elbow = get_landmark(
        landmarks,
        RIGHT_ELBOW
    )

    left_wrist = get_landmark(
        landmarks,
        LEFT_WRIST
    )

    right_wrist = get_landmark(
        landmarks,
        RIGHT_WRIST
    )

    left_hip = get_landmark(
        landmarks,
        LEFT_HIP
    )

    right_hip = get_landmark(
        landmarks,
        RIGHT_HIP
    )

    # -----------------------------------------------------
    # ARM ANGLES
    # -----------------------------------------------------

    left_arm = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_arm = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    # -----------------------------------------------------
    # BODY ANGLES
    # -----------------------------------------------------

    left_body = calculate_angle(
        left_hip,
        left_shoulder,
        left_elbow
    )

    right_body = calculate_angle(
        right_hip,
        right_shoulder,
        right_elbow
    )

    arm_angles = [
        x
        for x in [
            left_arm,
            right_arm
        ]
        if x is not None
    ]

    body_angles = [
        x
        for x in [
            left_body,
            right_body
        ]
        if x is not None
    ]

    if not arm_angles:

        return None

    arm_angle = (
        sum(arm_angles)
        /
        len(arm_angles)
    )

    body_angle = (
        sum(body_angles)
        /
        len(body_angles)
        if body_angles
        else 0
    )

    return (
        arm_angle,
        body_angle
    )


# =========================================================
# CREATE MEDIAPIPE POSE LANDMARKER
# =========================================================

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5,
)


# =========================================================
# PUSH-UP VARIABLES
# =========================================================

count = 0

state = "READY"

seen_down = False

last_angle = None

frame_no = 0

start_time = time.time()


# =========================================================
# CREATE LANDMARKER
# =========================================================

with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        frame_no += 1

        # Skip frames if requested
        if (
            process_every > 1
            and frame_no % process_every != 0
        ):

            continue

        # -------------------------------------------------
        # BGR -> RGB
        # -------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # Create MediaPipe Image
        # -------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # -------------------------------------------------
        # Detect pose
        # -------------------------------------------------

        try:

            result = landmarker.detect(
                mp_image
            )

        except Exception:

            result = None

        # -------------------------------------------------
        # Calculate angles
        # -------------------------------------------------

        angles = None

        if result is not None:

            angles = get_pose_angles(
                result
            )

        # -------------------------------------------------
        # Push-up state machine
        # -------------------------------------------------

        if angles:

            arm_angle, body_angle = angles

            last_angle = arm_angle

            # DOWN
            if arm_angle <= down_angle:

                seen_down = True

                state = "DOWN"

            # UP
            elif arm_angle >= up_angle:

                if seen_down:

                    count += 1

                    seen_down = False

                state = "UP"

            # MOVING
            else:

                state = "MOVING"

        else:

            state = "NO POSE"

        # =================================================
        # DRAW LANDMARKS
        # =================================================

        if result is not None and result.pose_landmarks:

            for pose_landmarks in result.pose_landmarks:

                # Connections for MediaPipe Pose
                connections = [
                    (11, 13),
                    (13, 15),

                    (12, 14),
                    (14, 16),

                    (11, 12),

                    (11, 23),
                    (12, 24),

                    (23, 24),

                    (23, 25),
                    (25, 27),

                    (24, 26),
                    (26, 28),
                ]

                # Draw points
                for index, lm in enumerate(
                    pose_landmarks
                ):

                    x = int(
                        lm.x * width
                    )

                    y = int(
                        lm.y * height
                    )

                    if (
                        0 <= x < width
                        and
                        0 <= y < height
                    ):

                        cv2.circle(
                            frame,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                # Draw skeleton
                for a, b in connections:

                    if (
                        a >= len(pose_landmarks)
                        or
                        b >= len(pose_landmarks)
                    ):

                        continue

                    x1 = int(
                        pose_landmarks[a].x
                        *
                        width
                    )

                    y1 = int(
                        pose_landmarks[a].y
                        *
                        height
                    )

                    x2 = int(
                        pose_landmarks[b].x
                        *
                        width
                    )

                    y2 = int(
                        pose_landmarks[b].y
                        *
                        height
                    )

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        # =================================================
        # ON VIDEO HUD
        # =================================================

        cv2.rectangle(
            frame,
            (15, 15),
            (350, 155),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"PUSH-UPS: {count}",
            (30, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (30, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        if last_angle is not None:

            cv2.putText(
                frame,
                f"ELBOW: {last_angle:.0f} deg",
                (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        # =================================================
        # SHOW VIDEO
        # =================================================

        preview.image(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            channels="RGB",
            use_container_width=True
        )

        # =================================================
        # PROGRESS
        # =================================================

        progress = (
            frame_no / total_frames
            if total_frames
            else 0
        )

        show_stats(
            count,
            state,
            progress
        )


# =========================================================
# CLEANUP
# =========================================================

cap.release()


# =========================================================
# FINAL RESULT
# =========================================================

elapsed = time.time() - start_time


st.success(
    f"✅ Analysis complete — "
    f"**{count} push-ups** detected "
    f"in {elapsed:.1f} seconds."
)


st.caption(
    "Tip: Side-view videos with the whole body visible "
    "give the most reliable results. "
    "This is a computer-vision estimate, not a "
    "medical or fitness-grade measurement."
)