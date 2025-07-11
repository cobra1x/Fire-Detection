import cv2
import numpy as np
import threading
import playsound
from ultralytics import YOLO

# 🔊 Function to play alarm sound
def play_alarm_sound_function():
    playsound.playsound('alarm-sound.mp3', True)

# 🚨 Alarm status tracker
Alarm_Status = False

# 🧠 Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use yolov8s.pt or yolov8m.pt for stronger accuracy

# 🎥 Start webcam
cap = cv2.VideoCapture(0)
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🧼 Resize and blur for consistency
    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)

    # 🔍 HSV-based fire color detection
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 50, 50], dtype="uint8")
    upper = np.array([35, 255, 255], dtype="uint8")
    fire_mask = cv2.inRange(hsv, lower, upper)

    # 🎞️ Motion detection using frame diff
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    prev_frame = gray

    # 🔀 Combine fire color and motion mask
    combined = cv2.bitwise_and(fire_mask, thresh)

    # 🧍 Suppress human regions using YOLOv8
    results = model(frame)
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:  # Class 0 = person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                combined[y1:y2, x1:x2] = 0  # Exclude human region from fire mask

    # 🔬 Contour analysis for flame-like shapes
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_detected = False

    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            fire_detected = True
            break

    # 🔴 Highlight fire zones visually
    frame[fire_mask > 0] = [0, 0, 255]

    # 📺 Show video stream
    cv2.imshow("Fire Detector", frame)

    # 🚨 Alert logic
    if fire_detected:
        print("🔥 Fire Detected!")

        if not Alarm_Status:
            Alarm_Status = True
            threading.Thread(target=play_alarm_sound_function).start()

    # 🛑 Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Release resources
cap.release()
cv2.destroyAllWindows()
