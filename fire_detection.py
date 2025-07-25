import cv2
import numpy as np
from tensorflow import keras
import time
import os

class FireDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_path)
        self.input_shape = self.model.input_shape[1:3]
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()

    def _load_model(self, model_path):
        print(f"Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"Model loaded. Input shape: {model.input_shape[1:3]}")
        return model

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched

    def detect(self, frame):
        processed = self.preprocess_frame(frame)
        prediction = self.model.predict(processed, verbose=0)
        confidence = prediction[0][0]
        class_name = 'Fire' if confidence > self.confidence_threshold else 'Normal'
        return class_name, confidence          #This function needed to be called by backend for fire or normal based on confidence>0.5 is fire

    # The below whole code can be deleted on further development or a new function can be build o requirememt!!!!!!!!!
    def visualize(self, frame, class_name, confidence):
        h, w = frame.shape[:2]
        color = (0, 0, 255) if class_name == 'Fire' else (0, 255, 0)
        vis_frame = frame.copy()

        if class_name == 'Fire':
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)
            warning = "WARNING: FIRE DETECTED!"
            cv2.putText(vis_frame, warning, (int(w/2) - 200, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(vis_frame, (0, h-40), (w, h), (0, 0, 0), -1)
        status = f"{class_name}: {confidence:.2f}"
        fps_display = f"FPS: {self.fps}"
        cv2.putText(vis_frame, status, (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_frame, fps_display, (w-120, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_frame

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to access camera ID {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Fire detection running... Press 'q' to exit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera.")
                    break

                class_name, confidence = self.detect(frame)

                # FPS update
                self.frame_count += 1
                if time.time() - self.fps_start_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.fps_start_time = time.time()

                output = self.visualize(frame, class_name, confidence)
                cv2.imshow("Fire Detection", output)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Detection stopped.")

def main():
    model_path = os.path.join("model", "fire_model.h5")  # Update if needed
    try:
        detector = FireDetector(model_path=model_path, confidence_threshold=0.5)
        detector.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# feed should come from cctv and for that some requirements to be done in future stages
