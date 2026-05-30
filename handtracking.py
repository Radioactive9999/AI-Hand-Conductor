%pip install mediapipe opencv-python
pip install sounddevice numpy
!pip install pygame
!pip install --upgrade pip
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import sounddevice as sd
# --------- VOLUME CONTROL (WINDOWS) ----------
import pygame.midi
from music_engine_test import MusicConductor
import music_engine_test as met
class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],[0, 1, 0, 0]], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        prediction = self.kf.predict()
        self.kf.correct(measurement)
        return int(prediction[0].item()), int(prediction[1].item())
sample_rate = 44100
duration = 0.1  # small chunk for real-time
conductor = MusicConductor()
ROOT_NOTE = 48
# Parameters initalisation and setting
model_path = "hand_landmarker.task"

#CREATE LANDMARKER
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
kf_left = Kalman2D()
kf_right = Kalman2D()
prev_time = 0
# Main Loop
global current_scale
current_scale = "chromatic"

with HandLandmarker.create_from_options(options) as landmarker:
    print("Camera started. Press 'q' to exit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        any_hand_found = False

        if result.hand_landmarks:
            conductor.hand_visible = True
            any_hand_found = True

            for i, hand_landmarks in enumerate(result.hand_landmarks):
                # MediaPipe labels are mirrored: "Right" label = user's LEFT hand
                label = result.handedness[i][0].category_name  # "Left" or "Right"
                is_left_hand = (label == "Right")   # flip if camera is not mirrored

                # Draw all landmarks (teal = volume/left, purple = pitch/right)
                color = (255, 200, 0) if is_left_hand else (200, 0, 255)
                for lm in hand_landmarks:
                    cx2, cy2 = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx2, cy2), 3, color, -1)

                # Index fingertip → Kalman smoothing
                lm = hand_landmarks[8]
                cx, cy = int(lm.x * w), int(lm.y * h)

                if is_left_hand:
                    # ---- LEFT HAND: volume + scale cycling ----
                    kx_l, ky_l = kf_left.update(cx, cy)
                    cv2.circle(frame, (cx, cy),    8, (0, 200, 255), -1)
                    cv2.circle(frame, (kx_l, ky_l), 10, (0, 255, 255), 2)
                    conductor.update_volume_hand(kx_l, ky_l, w, h, hand_landmarks)

                    # HUD — left side
                    vol_display = conductor.volume
                    cv2.putText(frame, f'Scale: {current_scale}', (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame, f'Vol:   {vol_display}',   (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 0), 2)

                    if met.is_fist(hand_landmarks):
                        cv2.putText(frame, "LEFT: FIST - Switch Scale",
                                    (10, h - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
                    else:
                        cv2.putText(frame, "LEFT: Volume",
                                    (10, h - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                else:
                    # ---- RIGHT HAND: pitch + tempo + chord/note ----
                    kx_r, ky_r = kf_right.update(cx, cy)
                    cv2.circle(frame, (cx, cy),     8, (255, 0, 100), -1)
                    cv2.circle(frame, (kx_r, ky_r), 10, (200, 0, 255), 2)
                    conductor.update_pitch_hand(kx_r, ky_r, w, h, hand_landmarks)

                    # HUD — right side
                    spread = met.get_finger_spread(hand_landmarks, w)
                    mode = "CHORD" if spread > 80 else "NOTE"
                    interval_ms = int(conductor.note_interval * 1000)
                    cv2.putText(frame, f'Mode:  {mode}',         (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    cv2.putText(frame, f'Tempo: {interval_ms}ms', (10, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

                    if spread > 80:
                        gesture_name  = "RIGHT: OPEN - Chord"
                        gesture_color = (255, 180, 0)
                    else:
                        gesture_name  = "RIGHT: Closed - Note"
                        gesture_color = (0, 255, 180)
                    cv2.putText(frame, gesture_name,
                                (w // 2 - 180, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, gesture_color, 2)

        else:
            if conductor.hand_visible:
                conductor.stop_all()
                conductor.hand_visible = False

        # ---- FPS ----
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Conductor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

conductor.stop_all()
cap.release()
cv2.destroyAllWindows()
pygame.midi.quit()
print("Done.")
