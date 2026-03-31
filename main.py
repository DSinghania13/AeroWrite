import cv2
import mediapipe as mp
import numpy as np
import os, urllib.request, ssl

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from writting import WritingHandler

# ===================== MODEL SETUP =====================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

# ===================== INIT =====================
writer = WritingHandler()

# ===================== UI =====================
def draw_status_bar(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def draw_bottom_panel(frame, text):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "OUTPUT: " + text,
                (20, h-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0), 2)

def draw_landmarks(frame, hand):
    h, w = frame.shape[:2]
    fingertips = [4, 8, 12, 16, 20]

    for i, lm in enumerate(hand):
        x = int(lm.x * w)
        y = int(lm.y * h)

        if i in fingertips:
            cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Initialize canvas inside writer
    writer.init_canvas(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    status_text = "IDLE"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        draw_landmarks(frame, hand)

        # 👉 CORE PIPELINE CALL
        status_text = writer.update(hand, frame)

    # Overlay drawing
    canvas = writer.get_canvas()
    if canvas is not None:
        frame = cv2.add(frame, canvas)

    # Get recognized text
    text = writer.get_text()

    # UI
    draw_status_bar(frame, status_text)
    draw_bottom_panel(frame, text)

    cv2.imshow("AeroWrite", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('Q'):
        break
    elif key == ord('C'):
        writer.clear_all()
    elif ord('a') <= key <= ord('z'):
        writer.calibrate(chr(key))

cap.release()
cv2.destroyAllWindows()