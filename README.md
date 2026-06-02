Markdown
# Facial Recognition Attendance System

A lightweight, real-time biometric attendance logger built with Python. 

This project uses a webcam stream to detect human faces, calculate their unique 128-dimensional encodings, and match them against a database of known users. If a match is found, it automatically logs the user's name and arrival time into a CSV file.

Note: The foundational architecture for this project was inspired by CodeWithHarry's Python tutorials, built out to test local environment compilation and real-time video processing.

## 🛠️ How It Works
1. The Memory: The script loads images from a local directory and converts them into mathematical biometric fingerprints.
2. The Vision: OpenCV captures the live webcam feed frame-by-frame.
3. The Brain: The `dlib` engine scans the frames for faces, maps them, and compares the live matrix against the saved memory.
4. The Log: Once an identity is confirmed, the system checks the `Attendance.csv` file. If the person hasn't been logged yet today, it writes their name and the exact timestamp.

## 💻 Tech Stack
* Language: Python 3.x
* Computer Vision: OpenCV (`cv2`)
* Machine Learning: `dlib`, `face_recognition`
* Data Management: Native CSV & Datetime modules

## ⚙️ Setup & Installation
Warning: `dlib` requires C++ compilers to build correctly on Windows.

1. Install Visual Studio Build Tools (Select "Desktop development with C++").
2. Clone this repository.
3. Create a virtual environment and install the dependencies:
   ```bash
   pip install cmake
   pip install opencv-python numpy
   pip install setuptools==69.5.1
   pip install dlib
   pip install face_recognition
Place a clear .jpg photo of yourself in the faces/ directory.

Run the live_recognition.py script.
