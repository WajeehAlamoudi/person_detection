# Person Detection with Parallel Processing

This project implements **real-time person detection** using **YOLO** and **OpenCV**, leveraging **parallel threading** for efficient sound alerts and video processing.

## 📌 Features
✅ **Real-time detection** using a trained YOLO model
✅ **Parallel sound alert system** to prevent blocking detections
✅ **Multi-threaded processing** for improved performance
✅ **Dynamic thresholding** to trigger alerts when a person crosses a line

## 🔧 How It Works
### 1️⃣ **Person Detection**
- Uses **YOLO** to detect people in a video stream.
- Each detected person’s bounding box and confidence score are extracted.
- A **threshold line** is drawn, and detection triggers an alert if a person crosses it.

### 2️⃣ **Parallel Sound Alert System**
- Uses a **separate thread** to manage sound playback.
- Sound alerts are stored in a **queue**, preventing multiple overlapping sounds.
- A **cooldown mechanism** ensures sounds are not replayed too frequently.

### 3️⃣ **Multi-Threaded Execution**
- The **main thread** handles video processing and person detection.
- A **background thread** manages sound alerts asynchronously.
- Prevents performance bottlenecks and ensures real-time responsiveness.

## 🚀 Running the Project
Ensure you have the required dependencies installed:
```bash
pip install ultralytics opencv-python numpy
```
Run the script:
```bash
python main.py
```

## 🛠️ How Parallel Processing is Used
### **Thread 1: Video Processing**
- Captures video frames.
- Runs YOLO detection.
- Draws bounding boxes and threshold lines.

### **Thread 2: Sound Management**
- Maintains a **queue** of alert sounds.
- Ensures sound cooldowns to prevent spamming.
- Runs independently, avoiding blocking the main process.

## 📄 Conclusion
This project demonstrates **efficient multi-threading** for AI-based video processing. By separating detection from sound alerts, it ensures smooth performance and real-time responsiveness without lag.

