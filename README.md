# 🧠 Introduction

In today’s fast-paced world, long hours of driving or screen exposure can lead to fatigue, drowsiness, and ultimately result in accidents or loss of productivity. According to the World Health Organization (WHO), driver fatigue is one of the leading causes of road accidents globally. Drowsiness and yawning are early symptoms of fatigue, and timely detection can prevent potential hazards. This project presents a real-time computer vision-based system to detect these signs and alert the user, thereby ensuring safety and attentiveness.

# 📚 Background

Driver fatigue detection systems have gained significant attention in recent years due to the rise in road accidents caused by inattentiveness and microsleeps. Traditional systems relied on sensor-based approaches such as steering behavior or heart rate monitoring, which are often invasive and expensive. Recent advances in computer vision and deep learning have enabled the development of non-intrusive, camera-based alertness monitoring systems. By leveraging facial landmarks and behavioral cues like eye closure, yawning, and head orientation, real-time systems can now effectively gauge a person's alertness.

# 🔍 Motivation

The motivation behind this project is threefold:

1. **Road Safety**: Prevent accidents by providing timely alerts to drowsy or inattentive drivers.  
2. **Accessibility**: Build a low-cost, non-invasive system that runs on standard webcams and consumer hardware.  
3. **Human-Computer Interaction Research**: Contribute to ongoing research in behavioral monitoring using real-time computer vision.

# 🎯 Objectives

The main objectives of this project are:

- To detect **drowsiness** by analyzing the Eye Aspect Ratio (EAR).  
- To detect **yawning** using the Mouth Aspect Ratio (MAR).  
- To identify if the user's **face is not visible or not oriented forward**, indicating inattentiveness.  
- To trigger **real-time audio alerts** upon detection of unsafe states.  
- To display a **live annotated video feed** along with EAR/MAR statistics using a Tkinter-based GUI.

# 🛠️ Implementation Details

## 💻 Technologies Used

- **Python**: Main programming language  
- **OpenCV**: For real-time video capture and image processing  
- **Dlib**: For detecting 68 facial landmarks  
- **MediaPipe**: For fast and robust face detection  
- **Tkinter**: For GUI development  
- **Playsound**: For playing alert sounds  
- **Threading**: To handle video and alert playback asynchronously

## 📷 Facial Landmark Detection

- MediaPipe detects face bounding boxes.  
- Dlib is used to extract facial landmarks, specifically:
  - Eyes (for EAR)  
  - Mouth (for MAR)

## 👁️ Eye Aspect Ratio (EAR)

- EAR is calculated using distances between eye landmarks.  
- A sustained drop below a threshold (e.g., EAR < 0.25) for several frames triggers a drowsiness alert.

## 😮 Mouth Aspect Ratio (MAR)

- MAR is calculated using mouth landmarks.  
- A high MAR value (e.g., > 0.7) indicates yawning behavior.

## 🧠 Face Visibility

- If no face is detected for a threshold duration, a “not looking ahead” warning is given.

## 🔊 Alert System

- Alerts are triggered using `playsound`, played in a separate thread to prevent UI lag.  
- Cooldown logic prevents repeated alert sounds.

## 🖼️ User Interface

- A Tkinter-based GUI shows:
  - Live webcam feed  
  - Detected face box and landmarks  
  - EAR and MAR values  
  - Status messages (Drowsy, Yawning, Not Looking, Active)
