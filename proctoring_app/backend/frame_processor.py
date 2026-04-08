"""
frame_processor.py — Decode base64 webcam frames and extract 8 visual features.

Primary pipeline: MediaPipe Face Mesh → gaze angles, head pose, face count.
Fallback pipeline: OpenCV Haar cascades → simplified features.

Visual features returned (in model order):
  [0] gaze_yaw            horizontal gaze angle (degrees)
  [1] gaze_pitch          vertical gaze angle (degrees)
  [2] head_yaw            head horizontal rotation (degrees)
  [3] head_pitch          head vertical tilt (degrees)
  [4] head_roll           head roll angle (degrees)
  [5] face_count          number of detected faces
  [6] gaze_deviation_ratio fraction of recent frames with |gaze_yaw| > 25°
  [7] face_embedding_norm  L2 norm of face crop feature vector (approx)
"""

import base64
import collections
import logging
import math
import re
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Phone detector optional import ───────────────────────────────────────────
try:
    from cheating_detection.models.phone_detector import detect_phone
    _PHONE_DETECTION_AVAILABLE = True
except ImportError:
    _PHONE_DETECTION_AVAILABLE = False
    logger.warning("Phone detector not available")

# ── MediaPipe optional import ─────────────────────────────────────────────────
try:
    import mediapipe as mp

    _mp_face_mesh = mp.solutions.face_mesh
    _MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe available — using Face Mesh pipeline")
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available — falling back to OpenCV Haar cascades")

# ── Constants ─────────────────────────────────────────────────────────────────
_GAZE_DEVIATION_THRESHOLD = 25.0  # degrees
_ROLLING_WINDOW_SIZE = 30          # frames kept for gaze_deviation_ratio

# 3D model points for head pose (standard face model)
_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0),   # Right mouth corner
], dtype=np.float64)

# Mediapipe landmark indices for the 6 model points above
_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# Mediapipe iris landmark indices (left / right)
_LEFT_IRIS_IDX = [474, 475, 476, 477]
_RIGHT_IRIS_IDX = [469, 470, 471, 472]
_LEFT_EYE_CORNERS = [33, 133]
_RIGHT_EYE_CORNERS = [362, 263]


class FrameProcessor:
    """
    Stateful frame processor that maintains a rolling gaze-deviation buffer.

    One instance per active WebSocket session.
    """

    def __init__(self) -> None:
        self._gaze_history: collections.deque = collections.deque(
            maxlen=_ROLLING_WINDOW_SIZE
        )
        self._face_mesh = None
        self._haar_cascade = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialise MediaPipe or Haar cascade detector."""
        if _MEDIAPIPE_AVAILABLE:
            self._face_mesh = _mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        else:
            self._haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    # ── Public interface ──────────────────────────────────────────────────────

    def process_frame(self, base64_frame: str) -> dict:
        """
        Decode a base64 JPEG frame and extract visual features.

        Parameters
        ----------
        base64_frame : str — base64-encoded JPEG image (may include data URI prefix).

        Returns
        -------
        dict with keys matching the 8 visual feature names, plus:
          phone_detected  : bool
          phone_confidence: float
        If no face is detected, returns a high-suspicion default vector.
        """
        image = self._decode_frame(base64_frame)
        if image is None:
            return self._no_face_defaults()

        if _MEDIAPIPE_AVAILABLE and self._face_mesh is not None:
            result = self._process_mediapipe(image)
        else:
            result = self._process_haar(image)

        # ── Phone detection (runs in addition to face pipeline) ───────────────
        if _PHONE_DETECTION_AVAILABLE:
            phone_detected, phone_conf = detect_phone(image)
        else:
            phone_detected, phone_conf = False, 0.0

        result["phone_detected"]   = phone_detected
        result["phone_confidence"] = phone_conf
        return result

    # ── MediaPipe pipeline ────────────────────────────────────────────────────

    def _process_mediapipe(self, image: np.ndarray) -> dict:
        """Extract visual features using MediaPipe Face Mesh."""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._gaze_history.append(999.0)  # off-screen sentinel
            return self._no_face_defaults()

        face_count = len(results.multi_face_landmarks)
        # Use the first (largest/closest) face for gaze and head pose
        landmarks = results.multi_face_landmarks[0].landmark

        # --- Gaze angles from iris ---
        gaze_yaw, gaze_pitch = self._estimate_gaze(landmarks, w, h)

        # --- Head pose ---
        head_yaw, head_pitch, head_roll = self._estimate_head_pose(landmarks, w, h)

        # --- Gaze deviation ratio (rolling window) ---
        self._gaze_history.append(abs(gaze_yaw))
        gaze_dev_ratio = (
            sum(1 for g in self._gaze_history if g > _GAZE_DEVIATION_THRESHOLD)
            / max(len(self._gaze_history), 1)
        )

        # --- Face embedding norm (approx: L2 norm of face crop pixel vector) ---
        emb_norm = self._face_embedding_norm(image, landmarks, w, h)

        return {
            "gaze_yaw": float(gaze_yaw),
            "gaze_pitch": float(gaze_pitch),
            "head_yaw": float(head_yaw),
            "head_pitch": float(head_pitch),
            "head_roll": float(head_roll),
            "face_count": float(face_count),
            "gaze_deviation_ratio": float(gaze_dev_ratio),
            "face_embedding_norm": float(emb_norm),
        }

    def _estimate_gaze(self, landmarks, w: int, h: int):
        """
        Estimate gaze_yaw and gaze_pitch from iris landmark offsets.

        Computes normalised iris centre position relative to the eye bounding
        box and maps to angular estimates.
        """
        def lm(idx):
            return landmarks[idx].x * w, landmarks[idx].y * h

        # Left eye
        lx0, ly0 = lm(_LEFT_EYE_CORNERS[0])
        lx1, ly1 = lm(_LEFT_EYE_CORNERS[1])
        l_iris = np.mean([[landmarks[i].x * w, landmarks[i].y * h]
                           for i in _LEFT_IRIS_IDX], axis=0)

        # Right eye
        rx0, ry0 = lm(_RIGHT_EYE_CORNERS[0])
        rx1, ry1 = lm(_RIGHT_EYE_CORNERS[1])
        r_iris = np.mean([[landmarks[i].x * w, landmarks[i].y * h]
                           for i in _RIGHT_IRIS_IDX], axis=0)

        def eye_ratio(cx, c0x, c1x):
            span = abs(c1x - c0x)
            if span < 1e-3:
                return 0.5
            return (cx - min(c0x, c1x)) / span

        l_ratio_x = eye_ratio(l_iris[0], lx0, lx1)
        r_ratio_x = eye_ratio(r_iris[0], rx0, rx1)
        l_ratio_y = eye_ratio(l_iris[1], ly0, ly1)
        r_ratio_y = eye_ratio(r_iris[1], ry0, ry1)

        avg_x = (l_ratio_x + r_ratio_x) / 2.0  # 0=left, 1=right
        avg_y = (l_ratio_y + r_ratio_y) / 2.0  # 0=top, 1=bottom

        # Convert to degrees: centre (0.5) → 0°, extremes → ±45°
        gaze_yaw = (avg_x - 0.5) * 90.0
        gaze_pitch = (avg_y - 0.5) * 60.0

        return gaze_yaw, gaze_pitch

    def _estimate_head_pose(self, landmarks, w: int, h: int):
        """
        Estimate head Euler angles via solvePnP on 6 facial landmarks.

        Returns (yaw, pitch, roll) in degrees.
        """
        image_points = np.array([
            (landmarks[idx].x * w, landmarks[idx].y * h)
            for idx in _LANDMARK_INDICES
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rotation_vec)
        # Decompose to Euler angles
        sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
        else:
            pitch = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
            yaw = math.atan2(-rot_mat[2, 0], sy)
            roll = 0.0

        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )

    def _face_embedding_norm(self, image: np.ndarray, landmarks, w: int, h: int) -> float:
        """
        Approximate face embedding L2 norm from face crop statistics.

        In production this would be replaced by a ResNet-50 ArcFace embedding.
        Here we compute L2 norm of normalised pixel intensities as a proxy.

        Returns float close to 1.0 for typical faces.
        """
        try:
            xs = [landmarks[i].x for i in range(0, 468, 20)]
            ys = [landmarks[i].y for i in range(0, 468, 20)]
            x1 = max(0, int(min(xs) * w) - 10)
            y1 = max(0, int(min(ys) * h) - 10)
            x2 = min(w, int(max(xs) * w) + 10)
            y2 = min(h, int(max(ys) * h) + 10)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return 1.0
            vec = crop.astype(np.float32).flatten() / 255.0
            norm = float(np.linalg.norm(vec) / (math.sqrt(len(vec)) + 1e-6))
            return round(norm * 2.0, 4)  # scale to ~1.0 range
        except Exception:
            return 1.0

    # ── Haar cascade fallback ─────────────────────────────────────────────────

    def _process_haar(self, image: np.ndarray) -> dict:
        """Simplified feature extraction using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            self._gaze_history.append(999.0)
            return self._no_face_defaults()

        face_count = len(faces)
        # Simple heuristics from face bounding box position
        x, y, fw, fh = faces[0]
        h, w = image.shape[:2]
        cx = x + fw / 2
        gaze_yaw = ((cx / w) - 0.5) * 60.0

        self._gaze_history.append(abs(gaze_yaw))
        gaze_dev_ratio = (
            sum(1 for g in self._gaze_history if g > _GAZE_DEVIATION_THRESHOLD)
            / max(len(self._gaze_history), 1)
        )

        return {
            "gaze_yaw": float(gaze_yaw),
            "gaze_pitch": 5.0,
            "head_yaw": float(gaze_yaw * 1.2),
            "head_pitch": 0.0,
            "head_roll": 0.0,
            "face_count": float(face_count),
            "gaze_deviation_ratio": float(gaze_dev_ratio),
            "face_embedding_norm": 1.0,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_frame(base64_frame: str) -> Optional[np.ndarray]:
        """
        Decode a base64-encoded JPEG/PNG frame to an OpenCV BGR image.

        Handles both raw base64 strings and data URI prefixes.
        """
        try:
            if "," in base64_frame:
                base64_frame = base64_frame.split(",", 1)[1]
            img_bytes = base64.b64decode(base64_frame)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _no_face_defaults(self) -> dict:
        """
        Return high-suspicion defaults when no face is detected.

        A missing face is treated as suspicious (high gaze deviation).
        """
        gaze_dev_ratio = (
            sum(1 for g in self._gaze_history if g > _GAZE_DEVIATION_THRESHOLD)
            / max(len(self._gaze_history), 1)
        )
        return {
            "gaze_yaw": 45.0,
            "gaze_pitch": 20.0,
            "head_yaw": 40.0,
            "head_pitch": 15.0,
            "head_roll": 5.0,
            "face_count": 0.0,
            "gaze_deviation_ratio": max(gaze_dev_ratio, 0.6),
            "face_embedding_norm": 1.0,
        }

    def reset(self) -> None:
        """Clear the rolling gaze history (call when session ends)."""
        self._gaze_history.clear()
