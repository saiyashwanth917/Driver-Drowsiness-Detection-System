"""Hybrid driver drowsiness demo: MediaPipe FaceMesh + temporal CV + MobileNetV2."""
import tensorflow as tf
import os, time, math, threading
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from app.location import WindowsLocation


try:
    import winsound
except Exception:
    winsound = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


try:
    import requests
except Exception:
    requests = None


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv("MOBILENET_MODEL", str(ROOT / "outputs/training/nthu_loso/fold_1/mobilenetv2/phase2_best.keras"))

# FaceMesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 61, 291]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132,
    93, 234, 127, 162, 21, 54, 103, 67, 109
]
POSE_IDS = [1, 152, 33, 263, 61, 291]
POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)



def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))



def ear(points, ids):
    p = [points[i] for i in ids]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * max(dist(p[0], p[3]), 1e-6))



def get_dl_stage(prob):
    """Map the model's sigmoid drowsiness score to provisional display stages."""
    if prob < 0.30:
        return "NORMAL"
    elif prob < 0.50:
        return "WARNING"
    elif prob < 0.75:
        return "DANGER"
    return "EMERGENCY"



class EmergencyEmailAlert:
    """Send emergency alerts through Gmail SMTP."""

    def __init__(self, recipient=None):
        self.sender = os.getenv("SMTP_EMAIL", "").strip()
        self.password = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "").strip()
        self.recipient = (
            recipient if recipient is not None
            else os.getenv("EMERGENCY_EMAIL", "")
        ).strip()
        self.enabled = bool(self.sender and self.password and self.recipient)
        self.last_sent = 0.0
        self.cooldown = int(os.getenv("EMAIL_COOLDOWN_SECONDS", "300"))
        self.busy = False

        if not self.enabled:
            print("Emergency email disabled: check SMTP_EMAIL, SMTP_APP_PASSWORD and EMERGENCY_EMAIL.")
        else:
            print("Emergency Gmail alert configured.")

    def send(self, message, location=None):
        if not self.enabled or self.busy or time.time() - self.last_sent < self.cooldown:
            return

        if location:
            lat = location["latitude"]
            lon = location["longitude"]
            accuracy = location["accuracy"]
            age = max(0, int(time.time() - location["timestamp"]))
            maps_url = f"https://maps.google.com/?q={lat},{lon}"
            message += (
                f"\n\nLast available location (approximate):\n"
                f"Latitude: {lat:.6f}\nLongitude: {lon:.6f}\n"
                f"Accuracy: about {accuracy:.0f} meters\n"
                f"Location age: {age} seconds\n{maps_url}"
            )
        else:
            message += "\n\nDriver GPS location is currently unavailable."

        self.busy = True

        def worker():
            try:
                import smtplib
                import ssl
                from email.message import EmailMessage

                email = EmailMessage()
                email["Subject"] = "DRIVER EMERGENCY ALERT - CRITICAL"
                email["From"] = self.sender
                email["To"] = self.recipient
                email.set_content(
                    "HYBRID DRIVER DROWSINESS DETECTION SYSTEM\n\n"
                    + message
                    + f"\n\nAlert time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    "smtp.gmail.com", 465, context=context, timeout=20
                ) as server:
                    server.login(self.sender, self.password)
                    server.send_message(email)

                self.last_sent = time.time()
                print("Emergency email sent successfully.")

            except Exception as e:
                print("Emergency email failed:", e)
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()


# Backward-compatible name: Streamlit already imports EmergencyEmailAlert.



class OptionalModel:
    def __init__(self):
        self.model = None
        self.label = "DL: unavailable"
        try:
            if Path(MODEL_PATH).exists():
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                self.label = "DL: MobileNetV2 loaded"
                print("Loaded MobileNet checkpoint:", MODEL_PATH)
            else:
                print("MobileNet checkpoint not found; CV-only mode.", MODEL_PATH)
        except Exception as e:
            print("MobileNet load skipped:", e)


    def predict(self, face_bgr):
        if self.model is None:
            return None
        try:
            shape = self.model.input_shape
            h, w = int(shape[1]), int(shape[2])
            image = cv2.resize(face_bgr, (w, h))
            x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # Match training: MobileNetV2 preprocess_input expects RGB pixels in 0..255.
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            out = self.model.predict(x[None, ...], verbose=0)
            return np.asarray(out).reshape(-1)
        except Exception as e:
            self.label = "DL: inference skipped"
            print("Model inference skipped:", e)
            return None



def head_pose(points, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([points[i] for i in POSE_IDS], dtype=np.float64)
    focal = float(w)
    camera = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(POSE_MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rot, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot)
        return tuple(float(x) for x in angles)
    except Exception:
        return None, None, None



def main():
    mp_face = mp.solutions.face_mesh
    camera_index = int(os.getenv("CAMERA_INDEX", "0"))
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit("Webcam could not be opened. Check camera permissions or CAMERA_INDEX.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


    model = OptionalModel()
    email_alert = EmergencyEmailAlert()
    location_service = WindowsLocation(refresh_seconds=20)
    location_service.start()
    eye_closed = False
    eye_closed_start = 0.0
    closure_durations = deque(maxlen=40)
    blink_times = deque(maxlen=30)
    ear_history = deque()
    critical_since = None
    critical_email_sent = False
    medium_since = None
    last_alarm = 0.0
    risk = 0
    perclos = 0.0
    fps_t = time.time()
    fps = 0.0
    dl_text = "DL: not loaded"
    dl_every = 0
    dl_result = None
    face_missing_since = None
    FACE_LOST_SECONDS = float(os.getenv('FACE_LOST_SECONDS', '3'))
    recovery_since = None
    normal_since = None

    # Initial heuristic thresholds; calibrate for camera and person.
    EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.21"))
    MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", "0.55"))
    CLOSED_SECONDS = float(os.getenv("CLOSED_EYE_SECONDS", "1.5"))
    CRITICAL_SECONDS = float(os.getenv("CRITICAL_SECONDS", "8"))
    HEAD_DOWN_ANGLE = float(os.getenv("HEAD_DOWN_ANGLE", "25"))



    with mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    ) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            now = time.time()
            dt = now - fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1 / dt)
            fps_t = now
            state = "NO FACE"
            ear_avg = mar = eye_duration = 0.0
            pose = (None, None, None)
            face_crop = None
            if result.multi_face_landmarks:
                face_missing_since = None
                lm = result.multi_face_landmarks[0].landmark
                points = [(int(p.x * w), int(p.y * h)) for p in lm]
                for idx in LEFT_EYE + RIGHT_EYE + MOUTH + POSE_IDS:
                    x, y = points[idx]
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)
                for idx in FACE_OVAL:
                    x, y = points[idx]
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                left_ear, right_ear = ear(points, LEFT_EYE), ear(points, RIGHT_EYE)
                ear_avg = (left_ear + right_ear) / 2
                mar = dist(points[13], points[14]) / max(dist(points[61], points[291]), 1e-6)
                ear_history.append((now, ear_avg))
                while ear_history and now - ear_history[0][0] > 30.0:
                    ear_history.popleft()
                face_x = [p[0] for p in points]
                face_y = [p[1] for p in points]
                x1, x2 = max(0, min(face_x) - 15), min(w, max(face_x) + 15)
                y1, y2 = max(0, min(face_y) - 15), min(h, max(face_y) + 15)
                face_crop = frame[y1:y2, x1:x2]
                pose = head_pose(points, frame.shape)

                closed_now = ear_avg < EAR_THRESHOLD
                if closed_now and not eye_closed:
                    eye_closed = True
                    eye_closed_start = now
                elif not closed_now and eye_closed:
                    duration = now - eye_closed_start
                    closure_durations.append(duration)
                    if duration < 0.7:
                        blink_times.append(now)
                    eye_closed = False
                eye_duration = now - eye_closed_start if eye_closed else 0.0
                state = "EYES CLOSED" if closed_now else "FACE TRACKED"
                if mar > MAR_THRESHOLD:
                    state = "YAWN"

                recent = [(t, e) for t, e in ear_history if now - t <= 30]
                perclos = sum(1 for _, e in recent if e < EAR_THRESHOLD) / max(1, len(recent))
                recent_blinks = sum(1 for t in blink_times if now - t <= 60)
                long_closure = eye_duration >= CLOSED_SECONDS
                yawn = mar > MAR_THRESHOLD
                head_down = pose[0] is not None and abs(pose[0]) > HEAD_DOWN_ANGLE
                moderate_signs = sum((
                    long_closure,
                    yawn,
                    head_down,
                    perclos > 0.30,
                ))

                if long_closure or perclos > 0.40:
                    raw_risk = 4
                elif perclos > 0.30:
                    raw_risk = 3
                elif moderate_signs >= 2 or recent_blinks >= 25:
                    raw_risk = 2
                elif (
                    yawn
                    or eye_duration > 0.4
                    or perclos > 0.18
                    or recent_blinks >= 18
                    or head_down
                ):
                    raw_risk = 1
                else:
                    raw_risk = 0

                # Match Streamlit: escalate immediately and recover gradually.
                if raw_risk > risk:
                    risk = raw_risk
                    recovery_since = None
                elif raw_risk < risk:
                    if raw_risk == 0:
                        if recovery_since is None:
                            recovery_since = now
                        if now - recovery_since >= 8.0:
                            risk = max(0, risk - 1)
                            recovery_since = now
                    else:
                        risk = max(raw_risk, risk - 1)
                        recovery_since = None
                else:
                    recovery_since = None

                # Emergency email is reserved for sustained CRITICAL risk.
                if risk == 4:
                    if critical_since is None:
                        critical_since = now
                    if (
                        now - critical_since >= CRITICAL_SECONDS
                        and not critical_email_sent
                    ):
                        msg = (
                            "EMERGENCY DROWSINESS ALERT: The driver has shown "
                            "sustained critical drowsiness indicators. Please "
                            "contact/check the driver immediately. This is an "
                            "automated fatigue warning, not accident confirmation."
                        )
                        latest_location = location_service.get_latest(max_age=90)
                        email_alert.send(msg, location=latest_location)
                        critical_email_sent = True
                else:
                    critical_since = None

                # Re-arm the email only after 8 seconds at NORMAL risk.
                if risk == 0:
                    if normal_since is None:
                        normal_since = now
                    elif now - normal_since >= 8.0:
                        critical_email_sent = False
                else:
                    normal_since = None


                dl_every += 1
                if model.model is not None and dl_every % 15 == 0 and face_crop is not None and face_crop.size:
                    dl_result = model.predict(face_crop)
                    if dl_result is not None and len(dl_result) > 0:
                        drowsy_prob = float(np.asarray(dl_result).reshape(-1)[0])
                        drowsy_prob = max(0.0, min(1.0, drowsy_prob))
                        dl_stage = get_dl_stage(drowsy_prob)
                        dl_text = f"DL: {drowsy_prob:.3f} | {dl_stage}"
            else:
                if face_missing_since is None:
                    face_missing_since = now
                missing_duration = now - face_missing_since
                state = "NO FACE" if missing_duration < FACE_LOST_SECONDS else "MONITORING LOST"
                # Keep prior risk/critical timer when the face is temporarily missing.
                eye_duration = 0.0

            risk_names = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            risk_colors = [(60, 200, 60), (100, 220, 100), (0, 220, 255), (0, 165, 255), (0, 0, 255)]
            risk_color = risk_colors[risk]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 155), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)
            cv2.putText(frame, f"HYBRID DRIVER MONITOR | RISK: {risk_names[risk]}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, risk_color, 2)
            cv2.putText(frame, f"EAR {ear_avg:.3f} | MAR {mar:.3f} | PERCLOS {perclos:.2f} | Eye closed {eye_duration:.1f}s", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2)
            cv2.putText(frame, f"State: {state} | FPS {fps:.1f} | Head pose {pose}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1)
            cv2.putText(frame, dl_text, (20, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "Q: quit | R: reset risk history", (w - 350, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if risk >= 3 and winsound and now - last_alarm > 2:
                try:
                    winsound.Beep(1200 if risk == 4 else 800, 250)
                    last_alarm = now
                except Exception:
                    pass
            cv2.imshow("Hybrid Driver Drowsiness Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                ear_history.clear(); blink_times.clear(); closure_durations.clear(); critical_since = None; critical_email_sent = False; face_missing_since = None; recovery_since = None; normal_since = None; risk = 0
                dl_result = None
                dl_text = "DL: reset / awaiting prediction"
    location_service.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
