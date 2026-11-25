"""WebRTC dashboard for Hybrid Driver Drowsiness Detection.

Run from the project root:
    .venv/Scripts/python.exe -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

# Ensure project root is available for Streamlit Cloud imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import av
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from streamlit_geolocation import streamlit_geolocation

from app.main import (
    LEFT_EYE, RIGHT_EYE, MOUTH, FACE_OVAL, POSE_IDS,
    ear, dist, head_pose, get_dl_stage, OptionalModel, EmergencyEmailAlert,
)
from app.auth import require_login


# ---------------------------------------------------------------------
st.set_page_config(
    page_title="DriveGuard AI | Driver Safety",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Require an authenticated account before showing the dashboard.
require_login()

# Configuration
# ---------------------------------------------------------------------
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.21"))
MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", "0.55"))
CLOSED_SECONDS = float(os.getenv("CLOSED_EYE_SECONDS", "1.5"))
CRITICAL_SECONDS = float(os.getenv("CRITICAL_SECONDS", "8"))
HEAD_DOWN_ANGLE = float(os.getenv("HEAD_DOWN_ANGLE", "25"))
FACE_LOST_SECONDS = float(os.getenv("FACE_LOST_SECONDS", "3"))
MESH_INTERVAL = float(os.getenv("MESH_INTERVAL_SECONDS", "0.10"))
DL_EVERY_N_FRAMES = int(os.getenv("DL_EVERY_N_FRAMES", "15"))
RISK_NAMES = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
RISK_COLORS = [
    (50, 210, 135), (70, 210, 170), (0, 190, 255),
    (0, 130, 255), (40, 50, 255),
]


# ---------------------------------------------------------------------
def risk_css(name: str) -> str:
    return f"risk-{name.lower()}"


class BrowserLocation:
    """Stores coordinates explicitly shared by the browser."""
    def __init__(self):
        self.lock = threading.Lock()
        self.data = None

    def start(self):
        pass

    def update(self, coordinates):
        try:
            data = {
                "latitude": float(coordinates["latitude"]),
                "longitude": float(coordinates["longitude"]),
                "accuracy": float(coordinates.get("accuracy") or 0),
                "timestamp": time.time(),
            }
            with self.lock:
                self.data = data
        except (KeyError, TypeError, ValueError):
            pass

    def get_latest(self, max_age=90):
        with self.lock:
            data = self.data.copy() if self.data else None
        if data is None or time.time() - data["timestamp"] > max_age:
            return None
        return data

    def stop(self):
        pass


class DrowsinessVideoProcessor:
    """One processor instance per WebRTC session.

    Camera frames arrive from the browser. No server-side VideoCapture is
    opened. Expensive model/CV resources are initialized once per session.
    """

    def __init__(self, emergency_email=None):
        self.lock = threading.RLock()
        # Start the processor immediately. Heavy resources are initialized in
        # daemon workers so WebRTC can return video frames without waiting for
        # TensorFlow or MediaPipe startup.
        self.mesh = None
        self.mesh_status = "INITIALIZING"
        self.model = None
        self.model_status = "LOADING"
        self._resource_lock = threading.RLock()
        self._dl_lock = threading.Lock()
        self._dl_busy = False
        self.email_alert = EmergencyEmailAlert(recipient=emergency_email)
        self.location = BrowserLocation()
        self.location_status = "STARTING"

        self._mesh_thread = threading.Thread(
            target=self._initialize_mesh, name="FaceMeshInit", daemon=True
        )
        self._model_thread = threading.Thread(
            target=self._initialize_model, name="MobileNetInit", daemon=True
        )
        self._location_thread = threading.Thread(
            target=self._initialize_location, name="GPSInit", daemon=True
        )

        self.eye_closed = False
        self.eye_closed_start = 0.0
        self.ear_history = deque()
        self.blink_times = deque(maxlen=60)
        self.closure_durations = deque(maxlen=40)

        self.risk = 0
        self.critical_since = None
        self.critical_email_sent = False
        self.recovery_since = None
        self.face_missing_since = None
        self.normal_since = None
        self._cached_landmarks = None

        self.frame_count = 0
        self.last_mesh_time = 0.0
        self.last_dl_time = 0.0
        self.dl_probability = None
        self.dl_stage = "WAITING"

        self.ear_value = 0.0
        self.mar_value = 0.0
        self.perclos = 0.0
        self.closed_seconds = 0.0
        self.pose = (None, None, None)
        self.face_state = "WAITING FOR CAMERA"
        self.capture_fps = 0.0
        self._fps_count = 0
        self._fps_start = time.time()
        self.processing_fps = 0.0
        self._last_frame_time = time.time()

        self.history = deque(maxlen=180)
        self.events = deque(maxlen=30)
        self.last_event_risk = None
        self.last_alarm = 0.0
        self.last_error = None
        self.closed = False
        self.camera_status = "CAMERA NOT STARTED"
        self.latest_frame = None
        self.latest_jpeg = None
        self.video_lock = threading.Lock()
        self.video_sequence = 0
        self.camera_thread = None
        self.processing_thread = None
        self.camera_capture = None
        self.raw_frame = None
        self.raw_frame_lock = threading.Lock()
        self.raw_frame_sequence = 0
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        self._workers_started = False
        self._startup_lock = threading.Lock()

    def _start_workers(self):
        """Start background resources once, after the camera stream connects."""
        with self._startup_lock:
            if self._workers_started or self.closed:
                return

            self._workers_started = True
            print("[CAMERA] Starting CV, DL and GPS workers for direct webcam capture.")

            self._mesh_thread.start()
            self._model_thread.start()
            self._location_thread.start()

    def _initialize_mesh(self):
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            with self._resource_lock:
                if self.closed:
                    mesh.close()
                    return
                self.mesh = mesh
                self.mesh_status = "READY"
        except Exception as exc:
            self.mesh_status = f"ERROR: {exc}"
            self.last_error = f"FaceMesh initialization failed: {exc}"

    def _initialize_model(self):
        try:
            model = OptionalModel()
            with self._resource_lock:
                if self.closed:
                    return
                self.model = model
                self.model_status = model.label
        except Exception as exc:
            self.model_status = f"ERROR: {exc}"
            self.last_error = f"MobileNetV2 initialization failed: {exc}"

    def _initialize_location(self):
        self.location_status = "WAITING FOR BROWSER LOCATION"

    def _run_dl_prediction(self, crop: np.ndarray):
        try:
            model = self.model
            if model is None or model.model is None or self.closed:
                return
            prediction = model.predict(crop)
            if prediction is not None and len(prediction):
                probability = float(np.clip(
                    np.asarray(prediction).reshape(-1)[0], 0.0, 1.0
                ))
                with self.lock:
                    self.dl_probability = probability
                    self.dl_stage = get_dl_stage(probability)
        except Exception as exc:
            self.last_error = f"MobileNetV2 inference failed: {exc}"
        finally:
            with self._dl_lock:
                self._dl_busy = False

    def _queue_dl_prediction(self, crop: np.ndarray):
        if self.model is None or self.model.model is None or self.closed:
            return
        with self._dl_lock:
            if self._dl_busy:
                return
            self._dl_busy = True
        threading.Thread(
            target=self._run_dl_prediction,
            args=(crop.copy(),),
            name="MobileNetInference",
            daemon=True,
        ).start()

    def _record_event(self, message: str, level: str):
        self.events.appendleft(
            (datetime.now().strftime("%H:%M:%S"), message, level)
        )

    def _update_risk(self, now: float, long_closure: bool, yawn: bool,
                     head_down: bool, recent_blinks: int):
        """Five-stage CV risk controller with separated risk indicators."""

        moderate_signs = sum((
            long_closure,
            yawn,
            head_down,
            self.perclos > 0.30,
        ))

        if long_closure or self.perclos > 0.40:
            raw = 4
        elif self.perclos > 0.30:
            raw = 3
        elif moderate_signs >= 2 or recent_blinks >= 25:
            raw = 2
        elif (
            yawn
            or self.closed_seconds > 0.4
            or self.perclos > 0.18
            or recent_blinks >= 18
            or head_down
        ):
            raw = 1
        else:
            raw = 0

        # Escalate immediately; require sustained normality to recover.
        if raw > self.risk:
            self.risk = raw
            self.recovery_since = None
        elif raw < self.risk:
            if raw == 0:
                if self.recovery_since is None:
                    self.recovery_since = now
                if now - self.recovery_since >= 8.0:
                    self.risk = max(0, self.risk - 1)
                    self.recovery_since = now
            else:
                self.risk = max(raw, self.risk - 1)
                self.recovery_since = None
        else:
            self.recovery_since = None

    def _critical_actions(self, now: float):
        if self.risk == 4:
            if self.critical_since is None:
                self.critical_since = now
                self._record_event("Critical condition detected", "CRITICAL")

            if (now - self.critical_since >= CRITICAL_SECONDS
                    and not self.critical_email_sent):
                location = self.location.get_latest(max_age=90)
                message = (
                    "EMERGENCY DROWSINESS ALERT: The driver has shown sustained "
                    "critical drowsiness indicators. Please contact/check the "
                    "driver immediately. This is an automated fatigue warning, "
                    "not accident confirmation."
                )
                self.email_alert.send(message, location=location)
                self.critical_email_sent = True
                self._record_event("Critical email attempt initiated", "CRITICAL")
        else:
            self.critical_since = None

        if self.risk == 0:
            if self.normal_since is None:
                self.normal_since = now
            elif now - self.normal_since >= 8.0:
                self.critical_email_sent = False
        else:
            self.normal_since = None

        # Browser-based voice warnings replace the local beep.

    def _process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        now = time.time()
        h, w = frame_bgr.shape[:2]
        self.frame_count += 1
        self._fps_count += 1

        elapsed_fps = now - self._fps_start
        if elapsed_fps >= 1.0:
            self.capture_fps = self._fps_count / elapsed_fps
            self._fps_count = 0
            self._fps_start = now

        dt = now - self._last_frame_time
        if dt > 0:
            instant = 1.0 / dt
            self.processing_fps = (
                0.85 * self.processing_fps + 0.15 * instant
                if self.processing_fps else instant
            )
        self._last_frame_time = now

        frame = cv2.flip(frame_bgr, 1)

        # FaceMesh is intentionally rate-limited; the browser video continues.
        mesh = self.mesh
        if mesh is not None and now - self.last_mesh_time >= MESH_INTERVAL:
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.last_mesh_time = now
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                self._cached_landmarks = [(p.x, p.y) for p in landmarks]
            else:
                # Clear stale landmarks immediately when FaceMesh loses the face.
                self._cached_landmarks = None

        # On intervening video frames, reuse the most recent FaceMesh result.

        self.face_state = "NO FACE"
        self.ear_value = 0.0
        self.mar_value = 0.0
        self.closed_seconds = 0.0
        self.pose = (None, None, None)

        cached = getattr(self, "_cached_landmarks", None)
        if cached:
            self.face_missing_since = None
            points = [(int(x * w), int(y * h)) for x, y in cached]

            for idx in LEFT_EYE + RIGHT_EYE + MOUTH + POSE_IDS:
                cv2.circle(frame, points[idx], 2, (0, 220, 255), -1)
            for idx in FACE_OVAL:
                cv2.circle(frame, points[idx], 2, (40, 220, 130), -1)

            self.ear_value = (
                ear(points, LEFT_EYE) + ear(points, RIGHT_EYE)
            ) / 2.0
            self.mar_value = (
                dist(points[13], points[14])
                / max(dist(points[61], points[291]), 1e-6)
            )
            self.pose = head_pose(points, frame.shape)

            self.ear_history.append((now, self.ear_value))
            while self.ear_history and now - self.ear_history[0][0] > 30.0:
                self.ear_history.popleft()

            closed_now = self.ear_value < EAR_THRESHOLD
            if closed_now and not self.eye_closed:
                self.eye_closed = True
                self.eye_closed_start = now
            elif not closed_now and self.eye_closed:
                duration = now - self.eye_closed_start
                self.closure_durations.append(duration)
                if duration < 0.7:
                    self.blink_times.append(now)
                self.eye_closed = False

            self.closed_seconds = (
                now - self.eye_closed_start if self.eye_closed else 0.0
            )
            self.face_state = (
                "YAWN DETECTED" if self.mar_value > MAR_THRESHOLD
                else "EYES CLOSED" if closed_now
                else "FACE TRACKED"
            )

            recent_ear = [
                value for timestamp, value in self.ear_history
                if now - timestamp <= 30.0
            ]
            self.perclos = (
                sum(value < EAR_THRESHOLD for value in recent_ear)
                / max(1, len(recent_ear))
            )
            recent_blinks = sum(
                1 for timestamp in self.blink_times
                if now - timestamp <= 60.0
            )
            long_closure = self.closed_seconds >= CLOSED_SECONDS
            yawn = self.mar_value > MAR_THRESHOLD

            # head_pose returns (pitch, yaw, roll); pitch indicates head-down.
            pitch = self.pose[0]
            head_down = pitch is not None and abs(pitch) > HEAD_DOWN_ANGLE

            self._update_risk(
                now, long_closure, yawn, head_down, recent_blinks
            )

            # MobileNetV2 inference runs in a worker so it cannot stall video.
            if self.frame_count % DL_EVERY_N_FRAMES == 0:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, x2 = max(0, min(xs) - 15), min(w, max(xs) + 15)
                y1, y2 = max(0, min(ys) - 15), min(h, max(ys) + 15)
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    self._queue_dl_prediction(crop)

            self._critical_actions(now)

        else:
            if self.face_missing_since is None:
                self.face_missing_since = now
            missing = now - self.face_missing_since
            self.face_state = (
                "MONITORING LOST" if missing >= FACE_LOST_SECONDS else "NO FACE"
            )
            # Do not downgrade risk just because the face is temporarily lost.

        risk_name = RISK_NAMES[self.risk]
        if risk_name != self.last_event_risk:
            self._record_event(f"Risk changed to {risk_name}", risk_name)
            self.last_event_risk = risk_name

        self.history.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "EAR": self.ear_value,
            "MAR": self.mar_value,
            "PERCLOS": self.perclos,
            "DL probability": (
                self.dl_probability if self.dl_probability is not None else 0.0
            ),
        })

        color = RISK_COLORS[self.risk]
        cv2.rectangle(frame, (0, 0), (w, 48), (8, 18, 35), -1)
        status_text = self.face_state
        if self.mesh is None:
            status_text = "INITIALIZING FACEMESH"
        cv2.putText(
            frame, f"DRIVER RISK: {risk_name} | {status_text}",
            (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2
        )
        cv2.putText(
            frame,
            f"EAR {self.ear_value:.3f}  MAR {self.mar_value:.3f}  "
            f"PERCLOS {self.perclos:.2f}  FPS {self.processing_fps:.1f}",
            (16, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (245, 245, 245), 2
        )
        return frame

    def start_browser(self):
        """Start background processing for annotated browser frames."""
        if self.closed:
            return

        self._start_workers()

        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="BrowserFrameProcessor",
                daemon=True,
            )
            self.processing_thread.start()

        self.camera_status = "WAITING FOR BROWSER CAMERA"

    def process_browser_frame(self, frame_bgr):
        """Display the newest browser frame with the latest CV annotations."""
        if self.closed:
            return frame_bgr

        if not self._workers_started:
            self._start_workers()

        # Send the newest frame to the background CV processing worker.
        with self.raw_frame_lock:
            self.raw_frame = frame_bgr.copy()
            self.raw_frame_sequence += 1

        self.camera_status = "CAMERA CONNECTED"

        # Always display the current camera frame, not a stale processed frame.
        frame = cv2.flip(frame_bgr, 1)
        h, w = frame.shape[:2]

        # Reuse the most recently detected FaceMesh landmarks.
        cached = getattr(self, "_cached_landmarks", None)
        if cached:
            points = [(int(x * w), int(y * h)) for x, y in cached]

            for idx in LEFT_EYE + RIGHT_EYE + MOUTH + POSE_IDS:
                cv2.circle(frame, points[idx], 2, (0, 220, 255), -1)

            for idx in FACE_OVAL:
                cv2.circle(frame, points[idx], 2, (40, 220, 130), -1)

        # Draw the current risk and live metric values over the fresh frame.
        risk_name = RISK_NAMES[self.risk]
        color = RISK_COLORS[self.risk]

        cv2.rectangle(frame, (0, 0), (w, 48), (8, 18, 35), -1)

        status_text = self.face_state
        if self.mesh is None:
            status_text = "INITIALIZING FACEMESH"

        cv2.putText(
            frame,
            f"DRIVER RISK: {risk_name} | {status_text}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"EAR {self.ear_value:.3f}  MAR {self.mar_value:.3f}  "
            f"PERCLOS {self.perclos:.2f}  FPS {self.processing_fps:.1f}",
            (16, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            2,
        )

        return frame

    def start_camera(self):
        """Start independent webcam capture and frame-processing workers."""
        if self.closed or (self.camera_thread and self.camera_thread.is_alive()):
            return

        self._start_workers()

        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="FrameProcessor",
                daemon=True,
            )
            self.processing_thread.start()

        self.camera_status = "OPENING CAMERA"
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            name="WindowsWebcam",
            daemon=True,
        )
        self.camera_thread.start()

    def _camera_loop(self):
        """Capture continuously; never wait for CV or JPEG encoding."""
        cap = None
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.camera_index)

            self.camera_capture = cap
            print(f"[CAMERA DEBUG] Processor ID: {id(self)}")
            print(f"[CAMERA DEBUG] Opening camera index: {self.camera_index}")

            if not cap.isOpened():
                self.camera_status = (
                    f"CAMERA ERROR: Could not open camera index {self.camera_index}. "
                    "Close other camera applications or change CAMERA_INDEX."
                )
                print("[CAMERA ERROR] OpenCV could not open the webcam.")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.camera_status = "CAMERA CONNECTED"
            frame_count = 0
            print("[CAMERA DEBUG] Webcam capture worker started.")

            while not self.closed:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.camera_status = "CAMERA CONNECTED - WAITING FOR FRAME"
                    time.sleep(0.01)
                    continue

                frame_count += 1
                with self.raw_frame_lock:
                    self.raw_frame = frame
                    self.raw_frame_sequence += 1

                # Publish the mirrored camera preview independently of CV.
                preview = cv2.flip(frame, 1)

                # Draw lightweight facial landmark dots on the live preview.
                # Avoid expensive full FaceMesh tessellation on every camera frame.
                cached = getattr(self, "_cached_landmarks", None)

                if cached:
                    ph, pw = preview.shape[:2]
                    mesh_points = [
                        (int(x * pw), int(y * ph))
                        for x, y in cached
                    ]

                    for idx in FACE_OVAL:
                        if idx < len(mesh_points):
                            cv2.circle(
                                preview, mesh_points[idx], 2,
                                (40, 220, 130), -1
                            )

                    for idx in set(LEFT_EYE + RIGHT_EYE + MOUTH + POSE_IDS):
                        if idx < len(mesh_points):
                            cv2.circle(
                                preview, mesh_points[idx], 2,
                                (0, 220, 255), -1
                            )

                # LIVE CAMERA STATUS OVERLAY
                risk_index = max(
                    0, min(int(self.risk), len(RISK_NAMES) - 1)
                )
                risk_name = RISK_NAMES[risk_index]
                status_text = self.face_state

                if self.mesh is None:
                    status_text = "INITIALIZING FACEMESH"

                ph, pw = preview.shape[:2]

                # Header bar
                cv2.rectangle(
                    preview, (0, 0), (pw, 27), (16, 28, 45), -1
                )
                cv2.putText(
                    preview,
                    "LIVE DRIVER DETECTION",
                    (12, 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (245, 245, 245),
                    1,
                    cv2.LINE_AA,
                )

                # Dynamic risk bar with risk-specific colours
                risk_colors = {
                    "NORMAL": (0, 220, 100),
                    "LOW": (0, 220, 100),
                    "MEDIUM": (0, 215, 255),
                    "HIGH": (0, 145, 255),
                    "CRITICAL": (0, 0, 255),
                }
                risk_color = risk_colors.get(
                    risk_name.upper(), (245, 245, 245)
                )

                cv2.rectangle(
                    preview, (0, 27), (pw, 70), (28, 18, 12), -1
                )
                cv2.putText(
                    preview,
                    f"DRIVER RISK: {risk_name}  |  {status_text}",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    risk_color,
                    2,
                    cv2.LINE_AA,
                )

                encoded_ok, encoded_jpeg = cv2.imencode(
                    ".jpg",
                    preview,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 68],
                )
                if encoded_ok:
                    with self.video_lock:
                        self.latest_jpeg = encoded_jpeg.tobytes()
                        self.video_sequence += 1

                if frame_count == 1:
                    print(f"[CAMERA DEBUG] First frame received: {frame.shape}")
                elif frame_count % 100 == 0:
                    print(f"[CAMERA DEBUG] Frames captured: {frame_count}")

                self.camera_status = "CAMERA CONNECTED"

        except Exception as exc:
            self.camera_status = f"CAMERA ERROR: {exc}"
            self.last_error = f"Camera capture failed: {exc}"
            print(f"[CAMERA ERROR] {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.camera_capture = None
            if not self.closed and not self.camera_status.startswith("CAMERA ERROR"):
                self.camera_status = "CAMERA STOPPED"

    def _processing_loop(self):
        """Process only the newest captured frame; discard stale frames."""
        last_sequence = -1

        while not self.closed:
            with self.raw_frame_lock:
                sequence = self.raw_frame_sequence
                frame = (
                    self.raw_frame.copy()
                    if self.raw_frame is not None and sequence != last_sequence
                    else None
                )

            if frame is None:
                time.sleep(0.003)
                continue

            last_sequence = sequence

            try:
                with self.lock:
                    output = self._process_frame(frame)
                    self.latest_frame = output.copy()

            except Exception as exc:
                self.last_error = f"Frame processing failed: {exc}"
                print(f"[CAMERA PROCESSING ERROR] {exc}")
                time.sleep(0.01)

    def latest_frame_copy(self):
        with self.lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def latest_jpeg_copy(self):
        with self.video_lock:
            return self.video_sequence, self.latest_jpeg

    def snapshot(self) -> dict:
        with self.lock:
            try:
                location = self.location.get_latest(max_age=90)
            except Exception:
                location = None
            return {
                "risk": self.risk,
                "face_state": self.face_state,
                "ear": self.ear_value,
                "mar": self.mar_value,
                "perclos": self.perclos,
                "closed_seconds": self.closed_seconds,
                "pose": self.pose,
                "dl_probability": self.dl_probability,
                "dl_stage": self.dl_stage,
                "model_status": self.model_status,
                "mesh_status": self.mesh_status,
                "location_status": self.location_status,
                "fps": self.processing_fps,
                "history": list(self.history),
                "events": list(self.events),
                "critical_since": self.critical_since,
                "email_enabled": self.email_alert.enabled,
                "email_busy": self.email_alert.busy,
                "location": location,
                "error": self.last_error,
            }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.camera_status = "CAMERA STOPPING"
        try:
            if self.camera_capture is not None:
                self.camera_capture.release()
        except Exception:
            pass
        try:
            self.location.stop()
        except Exception:
            pass
        try:
            mesh = self.mesh
            if mesh is not None:
                mesh.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------


class BrowserVideoProcessor(VideoProcessorBase):
    """Receives browser webcam frames and runs the existing detector."""
    def __init__(self, processor):
        self.processor = processor

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        processed = self.processor.process_browser_frame(image)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


class _MJPEGHandler(BaseHTTPRequestHandler):
    processor = None

    def do_GET(self):
        if self.path.split("?", 1)[0] != "/video":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        last_sequence = -1
        try:
            while True:
                proc = type(self).processor
                if proc is None or proc.closed:
                    time.sleep(0.05)
                    continue
                sequence, jpeg = proc.latest_jpeg_copy()
                if not jpeg or sequence == last_sequence:
                    time.sleep(0.005)
                    continue
                last_sequence = sequence
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def log_message(self, fmt, *args):
        return


@st.cache_resource
def _get_mjpeg_server():
    server = ThreadingHTTPServer(("127.0.0.1", 8765), _MJPEGHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="MJPEGServer", daemon=True).start()
    print("[VIDEO] MJPEG server listening on 127.0.0.1:8765")
    return server



# DRIVEGUARD_PRO_UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(ellipse at 10% 0%, #152b42 0%, transparent 40%),
                linear-gradient(135deg, #08111f, #0d1728 55%, #101d30);
    color: #e8f0fa;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: #0b1626;
    border-right: 1px solid #24354a;
}

section[data-testid="stSidebar"] * {
    color: #dce8f5;
}

h1, h2, h3 {
    color: #edf5ff !important;
    font-weight: 700 !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #15273c, #101e30);
    border: 1px solid #293e55;
    padding: 17px;
    border-radius: 13px;
    box-shadow: 0 5px 18px rgba(0,0,0,.12);
}

div[data-testid="stMetricLabel"] {
    color: #9aafc3 !important;
}

div[data-testid="stMetricValue"] {
    color: #f0f7ff !important;
    font-weight: 750;
}

.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
    border: 1px solid #31546b;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #087f8c, #12a995);
    color: white;
    border: none;
}

.stButton > button:hover {
    border-color: #47d8c3;
    color: white;
}

.risk {
    padding: 17px 20px;
    border-radius: 13px;
    font-size: 25px;
    font-weight: 800;
    text-align: center;
    margin: 10px 0 15px;
    border: 1px solid;
}

.risk-normal {
    color: #4ce0b0;
    background: rgba(24,174,126,.12);
    border-color: #239b78;
}

.risk-low {
    color: #71dfcf;
    background: rgba(42,180,170,.12);
    border-color: #269b9a;
}

.risk-medium {
    color: #ffcf69;
    background: rgba(239,169,45,.12);
    border-color: #b98532;
}

.risk-high {
    color: #ff9c69;
    background: rgba(242,106,55,.13);
    border-color: #c75d37;
}

.risk-critical {
    color: #ff6979;
    background: rgba(229,58,78,.15);
    border-color: #d4475a;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #293b50;
}

a {
    color: #55d6ce !important;
}

footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# Dashboard
# ---------------------------------------------------------------------

st.markdown("""
<style>
/* DriveGuard AI refined dashboard */
section[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.4rem;
}

.hero {
    background: linear-gradient(110deg, #12283b, #164052);
    border: 1px solid #28536a;
    border-radius: 16px;
    padding: 20px 26px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 27px;
    font-weight: 800;
    color: #f3f8fc;
    letter-spacing: -.7px;
}

.hero-sub {
    color: #8ecfc9;
    font-size: 11px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.panel {
    background: linear-gradient(145deg, #15273a, #101e2e);
    border: 1px solid #2b4054;
    border-radius: 14px;
    padding: 17px;
    margin-bottom: 12px;
}

.panel-label {
    color: #91aabd;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    margin-bottom: 12px;
}

.risk {
    padding: 18px 10px;
    border-radius: 11px;
    font-size: 29px;
    font-weight: 800;
    text-align: center;
    margin: 8px 0 14px;
    border: 1px solid;
}

.risk-normal {
    color: #4ce0b0;
    background: rgba(24,174,126,.12);
    border-color: #239b78;
}
.risk-low {
    color: #71dfcf;
    background: rgba(42,180,170,.12);
    border-color: #269b9a;
}
.risk-medium {
    color: #ffcf69;
    background: rgba(239,169,45,.12);
    border-color: #b98532;
}
.risk-high {
    color: #ff9c69;
    background: rgba(242,106,55,.13);
    border-color: #c75d37;
}
.risk-critical {
    color: #ff6979;
    background: rgba(229,58,78,.15);
    border-color: #d4475a;
}

div[data-testid="stMetric"] {
    background: #15273a;
    border: 1px solid #293e55;
    border-radius: 11px;
    padding: 12px 14px;
}

div[data-testid="stMetricLabel"] {
    color: #9aafc3 !important;
    font-size: 12px;
}

div[data-testid="stMetricValue"] {
    color: #f0f7ff !important;
    font-weight: 750;
}

.stButton > button {
    border-radius: 9px;
    min-height: 42px;
    font-weight: 700;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(100deg, #087f8c, #12a995);
    border: none;
    color: white;
}

.status-line {
    color: #a5bdca;
    font-size: 13px;
    padding: 7px 0;
}

div[data-testid="stAlert"] {
    border-radius: 11px;
}

footer, #MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">?? DRIVEGUARD AI</div>'
    '<div class="hero-sub">HYBRID DRIVER SAFETY MONITORING SYSTEM</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Camera controls
if "drowsiness_processor" not in st.session_state:
    st.session_state.drowsiness_processor = None

start_col, stop_col, camera_status_col = st.columns([1, 1, 4])

with start_col:
    start_clicked = st.button(
        "? START MONITORING",
        type="primary",
        use_container_width=True
    )

with stop_col:
    stop_clicked = st.button(
        "? STOP",
        use_container_width=True
    )

if start_clicked:
    old = st.session_state.drowsiness_processor
    if old is not None:
        old.close()

    user = st.session_state["authenticated_user"]
    processor = DrowsinessVideoProcessor(
        emergency_email=user["emergency_email"]
    )
    st.session_state.drowsiness_processor = processor
    processor.start_browser()
    st.rerun()

if stop_clicked:
    processor = st.session_state.drowsiness_processor
    if processor is not None:
        processor.close()

    st.session_state.drowsiness_processor = None
    st.rerun()

active_processor = st.session_state.drowsiness_processor

# Request the driver's location from their browser.
if active_processor is not None:
    st.caption("Share browser location for emergency email alerts:")
    browser_location = streamlit_geolocation()
    if isinstance(browser_location, dict) and browser_location.get("latitude") is not None:
        active_processor.location.update(browser_location)
        active_processor.location_status = "BROWSER LOCATION READY"

with camera_status_col:
    if active_processor is None:
        st.caption("? MONITORING STOPPED")
    else:
        st.caption(f"? {active_processor.camera_status}")

st.markdown("---")

# Main camera and risk section
camera_col, risk_col = st.columns([1.8, 1], gap="medium")
risk_placeholder = risk_col.empty()

with camera_col:
    st.markdown(
        '<div class="panel-label">LIVE DRIVER MONITORING</div>',
        unsafe_allow_html=True,
    )

    if active_processor is not None:
        try:
            webrtc_ctx = webrtc_streamer(
                key=f"driver-monitor-{id(active_processor)}",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: BrowserVideoProcessor(active_processor),
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 480, "max": 480},
                        "height": {"ideal": 360, "max": 360},
                        "frameRate": {"ideal": 30, "max": 30},
                    },
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                },
                video_receiver_size=1,
                async_processing=True,
            )

            st.write("WebRTC playing:", webrtc_ctx.state.playing)
            st.write("WebRTC state:", webrtc_ctx.state)
            st.write("Camera status:", active_processor.camera_status)
        except Exception as exc:
            st.error(f"Browser camera error: {exc}")
    else:
        st.markdown("""
        <div style="
            background:#0b1420;
            border:1px solid #293e55;
            border-radius:12px;
            height:340px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            color:#829caf;
        ">
            <div style="font-size:45px;">?</div>
            <div style="font-size:16px;font-weight:700;margin-top:12px;">
                CAMERA STANDBY
            </div>
            <div style="font-size:12px;margin-top:7px;">
                Start monitoring to activate driver detection
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_primary_status():
    current = st.session_state.get("drowsiness_processor")

    with risk_placeholder.container():
        st.markdown(
            '<div class="panel-label">DRIVER SAFETY STATUS</div>',
            unsafe_allow_html=True,
        )

        if current is None:
            st.markdown(
                '<div class="risk risk-normal">STANDBY</div>',
                unsafe_allow_html=True,
            )
            st.caption("Monitoring has not started.")
            return

        data = current.snapshot()
        risk_name = RISK_NAMES[data["risk"]]

        st.markdown(
            f'<div class="risk {risk_css(risk_name)}">{risk_name}</div>',
            unsafe_allow_html=True,
        )

        st.caption("HYBRID CV + MOBILENETV2 FUSION")
        st.progress((data["risk"] + 1) / 5)

        voice_html = '''
        <button id="voice-enable"
          style="padding:9px 14px;border-radius:8px;border:1px solid #52718a;
          background:#14283a;color:#eaf4fb;font-weight:600;cursor:pointer;">
          ENABLE VOICE WARNINGS
        </button>
        <script>
        (() => {
          const host = window.parent;
          const risk = __RISK__;

          const messages = {
            1: "Low drowsiness warning. Stay alert and keep your eyes on the road.",
            2: "Medium drowsiness detected. Please stay focused.",
            3: "High drowsiness detected. Stop safely and take a break.",
            4: "Critical drowsiness detected. Stop safely and rest immediately."
          };

          const state = host.__driveguardVoiceState ||
            (host.__driveguardVoiceState = {
              enabled: false,
              risk: 0,
              lastRisk: null,
              lastAt: 0
            });

          state.risk = risk;
          host.__driveguardVoiceMessages = messages;

          try {
            if (localStorage.getItem("driveguard_voice_enabled") === "1") {
              state.enabled = true;
            }
          } catch(e) {}

          function speak(message) {
            const synth = host.speechSynthesis;
            const Utterance = host.SpeechSynthesisUtterance;
            if (!synth || !Utterance) return;

            synth.cancel();
            const speech = new Utterance(message);
            speech.lang = "en-US";
            speech.rate = 0.92;
            synth.speak(speech);
          }

          function processRisk() {
            if (!state.enabled) return;

            const currentRisk = state.risk;
            const now = Date.now();

            if (currentRisk === 0) {
              state.lastRisk = 0;
              state.lastAt = 0;
              return;
            }

            const changed = currentRisk !== state.lastRisk;

            const interval = currentRisk === 4 ? 6000 :
                             currentRisk === 3 ? 10000 : 0;

            const repeat = interval > 0 &&
                           now - state.lastAt >= interval;

            if (changed || repeat) {
              const message = messages[currentRisk];

              if (message) {
                speak(message);
                state.lastRisk = currentRisk;
                state.lastAt = now;
              }
            }
          }

          const button = document.getElementById("voice-enable");

          if (button) {
            button.textContent = state.enabled
              ? "VOICE WARNINGS ENABLED"
              : "ENABLE VOICE WARNINGS";

            button.onclick = () => {
              state.enabled = true;
              state.lastRisk = null;
              state.lastAt = 0;

              try {
                localStorage.setItem("driveguard_voice_enabled", "1");
              } catch(e) {}

              button.textContent = "VOICE WARNINGS ENABLED";
              speak("Voice warnings are enabled.");

              // Announce an existing risk after the activation message.
              host.setTimeout(processRisk, 1400);
            };
          }

          // Called again on every one-second dashboard refresh.
          processRisk();
        })();
        </script>
        '''
        voice_html = voice_html.replace("__RISK__", str(int(data["risk"])))
        st.components.v1.html(voice_html, height=55, scrolling=False)


        st.markdown("---")

        st.markdown(
            '<div class="panel-label">DEEP LEARNING</div>',
            unsafe_allow_html=True,
        )

        if data["dl_probability"] is None:
            st.metric("Drowsiness probability", "Waiting")
        else:
            st.metric(
                "Drowsiness probability",
                f'{data["dl_probability"]:.3f}'
            )
            st.progress(data["dl_probability"])

        st.caption(f'Model: {data["model_status"]}')
        st.caption(f'FaceMesh: {data["mesh_status"]}')

        st.markdown("---")

        st.markdown(
            '<div class="panel-label">MONITORING</div>',
            unsafe_allow_html=True,
        )

        st.write(f'**Face:** {data["face_state"]}')
        st.write(f'**Processing speed:** {data["fps"]:.1f} FPS')

        if data["critical_since"] is not None:
            remaining = max(
                0.0,
                CRITICAL_SECONDS - (
                    time.time() - data["critical_since"]
                )
            )
            st.warning(f"Critical alert timer: {remaining:.1f} seconds")

# Essential detection measurements
def render_detection_metrics():
    current = st.session_state.get("drowsiness_processor")

    st.markdown(
        '<div class="panel-label">LIVE DETECTION PARAMETERS</div>',
        unsafe_allow_html=True,
    )

    if current is None:
        st.caption("Detection measurements will appear when monitoring starts.")
        return

    data = current.snapshot()
    pitch, yaw, roll = data["pose"]

    metrics = st.columns(5)

    values = [
        ("EAR", f'{data["ear"]:.3f}'),
        ("MAR", f'{data["mar"]:.3f}'),
        ("PERCLOS", f'{data["perclos"]:.2f}'),
        ("HEAD PITCH", f"{pitch:.1f}?" if pitch is not None else "N/A"),
        ("FPS", f'{data["fps"]:.1f}'),
    ]

    for column, (label, value) in zip(metrics, values):
        column.metric(label, value)

# Compact emergency and GPS information
def render_safety_services():
    current = st.session_state.get("drowsiness_processor")

    st.markdown(
        '<div class="panel-label">SAFETY SERVICES</div>',
        unsafe_allow_html=True,
    )

    if current is None:
        st.caption("GPS and emergency email status will appear during monitoring.")
        return

    data = current.snapshot()
    gps_col, email_col = st.columns(2)

    with gps_col:
        st.markdown("**GPS LOCATION**")
        location = data["location"]

        if location:
            lat = location["latitude"]
            lon = location["longitude"]

            st.write(f"{lat:.6f}, {lon:.6f}")
            st.markdown(
                f"[View location on Google Maps]"
                f"(https://maps.google.com/?q={lat},{lon})"
            )
        else:
            st.caption(data["location_status"])

    with email_col:
        st.markdown("**EMERGENCY EMAIL**")

        st.write(
            "Configured" if data["email_enabled"]
            else "Not configured"
        )

        if data["email_busy"]:
            st.caption("Email request processing")

        st.caption(
            f"Alert after {CRITICAL_SECONDS:.0f}s sustained critical risk"
        )

@st.fragment(run_every=1.0)
def render_dashboard_updates():
    render_primary_status()
    st.markdown("---")
    render_detection_metrics()
    st.markdown("---")
    render_safety_services()


render_dashboard_updates()


if active_processor is not None:
    error = active_processor.snapshot()["error"]
    if error:
        st.warning(f"System message: {error}")

st.markdown(
    '<div style="text-align:center;color:#71899b;'
    'font-size:11px;padding:18px;">'
    'DRIVEGUARD AI ? HYBRID DRIVER MONITORING'
    '</div>',
    unsafe_allow_html=True,
)



