<img width="1119" height="778" alt="image" src="https://github.com/user-attachments/assets/44319b7b-324d-4556-891c-4389f2dfe991" />

# 🚦 Traffic Violation Detection System using YOLOv8

## Overview

Traffic violations are one of the leading causes of road accidents and safety concerns. This project presents an AI-powered Traffic Violation Detection System that uses Computer Vision and Deep Learning techniques to automatically detect traffic violations from video footage.

The system leverages YOLOv8 for real-time object detection and tracking to identify:

* ✅ Helmet Detection
* ✅ No-Helmet Detection
* ✅ Triple Riding Detection
* ✅ Vehicle Monitoring
* ✅ Real-Time Video Processing
* ✅ Violation Analytics

This solution can assist traffic authorities in automating surveillance and improving road safety.

---

## Features

### Helmet Violation Detection

Detects whether a rider is wearing a helmet or not.

### Triple Riding Detection

Identifies motorcycles carrying more than two riders.

### Real-Time Object Detection

Uses YOLOv8 to detect and classify objects with high accuracy.

### Video-Based Monitoring

Processes uploaded or recorded traffic videos.

### Automated Violation Tracking

Tracks and highlights violations throughout the video stream.

---

## Tech Stack

| Technology  | Purpose               |
| ----------- | --------------------- |
| Python      | Core Development      |
| YOLOv8      | Object Detection      |
| OpenCV      | Video Processing      |
| Ultralytics | YOLO Framework        |
| NumPy       | Numerical Computation |
| Streamlit   | Web Interface         |

---

## Project Structure

```text
traffic-violation-detection-yolo/
│
├── main.py
├── helmet_test.py
├── readycode.py
├── data.yaml
├── README.md
├── .gitignore
├── .gitattributes
│
├── helmet model/
│   └── (Helmet Detection Weights)
│
├── triple riding model/
│   └── (Triple Riding Detection Weights)
│
├── triple riding model 2/
│   └── (Additional Model Weights)
│
├── demo videos/
│   └── (Project Demonstration Videos)
│
├── output videos/
│   └── (Processed Output Videos)
│
├── README.dataset.txt
├── README.roboflow.txt
└── yolov8s.pt
```

---

## Model Weights

Due to GitHub file size limitations, trained model files are hosted externally.

### Download Models

THE NO HELMET AND TRIPLE RIDING MODEL links is below:
https://drive.google.com/drive/folders/1DbLONQlFfTnTvAXk4dAbBSjg0RcwhX6k?usp=sharing

---

## Demo Videos

Project demonstration videos are available here:

🎥 Demo Video and OUTPUT video drive link :
https://drive.google.com/drive/folders/16RAli0cT4zq0avTpUVy2I6XLetjkq_uv?usp=sharing

---

## Installation

### Clone Repository

```bash
git clone https://github.com/sarankumarmortha40-dev/traffic-violation-detection-yolo.git

cd traffic-violation-detection-yolo
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Models

Download the trained models from the links above and place them in the appropriate project directories.

### Run Application

```bash
python main.py
```

or

```bash
streamlit run app.py
```

---

## Results

The system successfully:

* Detects helmet and no-helmet riders
* Detects triple riding violations
* Tracks vehicles across frames
* Processes real-world traffic videos
* Generates visual violation outputs

---

## Future Improvements

* Automatic Number Plate Recognition (ANPR)
* Traffic Signal Violation Detection
* Cloud-Based Dashboard
* Violation Report Generation
* Real-Time CCTV Integration
* Automated E-Challan Generation

---

## Screenshots

### Helmet Detection

<img width="674" height="503" alt="image" src="https://github.com/user-attachments/assets/a80cda7f-17c1-4855-bac1-94cff72cae1f" />


### Triple Riding Detection

<img width="512" height="503" alt="image" src="https://github.com/user-attachments/assets/ba24ed3b-31dc-46d1-a298-172a2fc4919e" />


### Violation Tracking

<img width="661" height="890" alt="image" src="https://github.com/user-attachments/assets/557425ac-46ec-494e-93b3-6df3825dee20" />

---

## Author

### Saran Kumar Mortha

Computer Vision | Artificial Intelligence | Machine Learning | Full Stack Development

LinkedIn:
MORTHA SARAN KUMAR 

GitHub:
https://github.com/sarankumarmortha40-dev

---

## License
This project is intended for educational, research, and demonstration purposes.
This project is intended for educational, research, and demonstration purposes.

