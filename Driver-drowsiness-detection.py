import cv2
import numpy as np
import dlib
from imutils import face_utils
from tkinter import *
from PIL import Image, ImageTk
from playsound import playsound
import threading
import mediapipe as mp
import time

# Sound files
s1 = r'C:\Users\Souvik Deb\Downloads\audio\eyesclosed.mp3'
s2 = r'C:\Users\Souvik Deb\Downloads\audio\LOOKING AHEAD.mp3'
s3 = r'C:\Users\Souvik Deb\Downloads\audio\yawn.mp3'

# EAR and MAR calculation functions
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# Load dlib predictor
predictor_path = "C:\\Users\\Souvik Deb\\Downloads\\shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (48, 68)

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.75)

# GUI setup
root = Tk()
root.title("Eye and Yawn Status Detection")
root.geometry("600x500")

status_label = Label(root, text="Status: Waiting", font=("Helvetica", 16))
status_label.pack(pady=10)

video_label = Label(root)
video_label.pack()

# Flags and buffers
cap = None
stop_video = False
smoothed_ear = 0.0
smoothed_mar = 0.0
last_face_seen_time = time.time()
last_sound_time = {"s1": 0, "s2": 0, "s3": 0}
SOUND_COOLDOWN = 3  # seconds

# Thresholds (tuned for better reactivity)
EYE_CLOSED_THRESHOLD = 0.19  # Lower = faster response
YAWN_THRESHOLD = 0.6         # Adjusted for better yawn detection
CONSECUTIVE_FRAMES_THRESHOLD = 3
CONSECUTIVE_EYE_CLOSED_FRAMES_THRESHOLD = 4

consecutive_yawn_frames = 0
consecutive_eye_closed_frames = 0

def play_sound(audio_file, sound_key):
    now = time.time()
    if now - last_sound_time[sound_key] > SOUND_COOLDOWN:
        last_sound_time[sound_key] = now
        threading.Thread(target=playsound, args=(audio_file,), daemon=True).start()

def update_status(text):
    status_label.config(text=text)

def smooth_value(value, smoothed_value, alpha=0.8):
    return alpha * value + (1 - alpha) * smoothed_value

def start_detection():
    global cap, stop_video, consecutive_yawn_frames, consecutive_eye_closed_frames
    global smoothed_ear, smoothed_mar, last_face_seen_time
    stop_video = False
    cap = cv2.VideoCapture(0)

    def process_frame():
        global consecutive_yawn_frames, consecutive_eye_closed_frames, smoothed_ear, smoothed_mar, last_face_seen_time

        if stop_video:
            cap.release()
            return

        ret, frame = cap.read()
        if not ret:
            root.after(10, process_frame)
            return

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)

        found_face = False
        if results.detections:
            for detection in results.detections:
                if detection.score[0] < 0.75:
                    continue

                found_face = True
                last_face_seen_time = time.time()

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                startX = max(0, int(bboxC.xmin * iw))
                startY = max(0, int(bboxC.ymin * ih))
                endX = min(iw - 1, startX + int(bboxC.width * iw))
                endY = min(ih - 1, startY + int(bboxC.height * ih))

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                rect = dlib.rectangle(startX, startY, endX, endY)

                try:
                    shape = predictor(gray, rect)
                    if shape.num_parts == 0:
                        continue
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]

                    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                    smoothed_ear = smooth_value(ear, smoothed_ear)

                    mar = mouth_aspect_ratio(mouth)
                    smoothed_mar = smooth_value(mar, smoothed_mar)

                    if smoothed_ear < EYE_CLOSED_THRESHOLD:
                        consecutive_eye_closed_frames += 1
                        if consecutive_eye_closed_frames >= CONSECUTIVE_EYE_CLOSED_FRAMES_THRESHOLD:
                            update_status("Status: Sleepy - Eyes Closed")
                            play_sound(s1, "s1")
                    else:
                        consecutive_eye_closed_frames = 0
                        update_status("Status: Eyes Open")

                    if smoothed_mar > YAWN_THRESHOLD:
                        consecutive_yawn_frames += 1
                        if consecutive_yawn_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                            update_status("Status: Yawning")
                            play_sound(s3, "s3")
                    else:
                        consecutive_yawn_frames = 0

                    # Draw landmarks and values
                    cv2.polylines(frame, [np.array(leftEye)], True, (0, 255, 255), 1)
                    cv2.polylines(frame, [np.array(rightEye)], True, (0, 255, 255), 1)
                    cv2.polylines(frame, [np.array(mouth)], True, (255, 0, 0), 1)

                    cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (startX, startY - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    print(f"[WARN] Shape prediction error: {e}")

        # Trigger “not looking ahead” only if face not seen recently
        if not found_face:
            if time.time() - last_face_seen_time > 1.5:
                update_status("Status: Not Looking Ahead")
                play_sound(s2, "s2")

        # Display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        root.after(10, process_frame)

    root.after(0, process_frame)

def stop_detection():
    global stop_video
    stop_video = True
    update_status("Status: Detection Stopped")

# Buttons
start_button = Button(root, text="Start Detection", command=lambda: threading.Thread(target=start_detection, daemon=True).start())
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

root.mainloop()
