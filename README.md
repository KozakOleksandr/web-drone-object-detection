# Web-Based Drone Object Detection System

This repository contains the **diploma version** of a web-based information system for real-time object detection from drone footage using a trained YOLOv11 model.

## 🎯 Project Overview

The goal of this project is to build a web platform that can:

* Identify military or technical objects from images and videos captured by drones.
* Support detection from uploaded files and real-time video streams.
* Log and manage user access to the system.

The system was developed as part of a Bachelor's thesis in the field of information technologies.

## 🧠 Key Features

* ✅ YOLOv11-based object detection
* ✅ Real-time detection from webcam and drone streams
* ✅ Image and video file uploads
* ✅ User authentication (registration, login/logout)
* ✅ Secure access: detection available only to logged-in users
* ✅ Logging of all actions for auditing

## ⚙️ Tech Stack

* **Backend**: Django 4.x (Python)
* **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
* **Computer Vision**: OpenCV, PyTorch, YOLOv11 (custom-trained model)
* **Database**: SQLite (default), can be switched to PostgreSQL
* **Authentication**: Django's built-in auth system
* **Deployment-ready**: Configurable for production environments

## 📁 Project Structure

```
diploma_object_detection/
│
├── detect/                   # Detection module (YOLO logic, video/image processing)
├── users/                    # User management (login, registration, logging)
├── templates/                # HTML templates
├── static/                   # Static files (CSS, JS)
├── media/                    # Uploaded files (images/videos)
├── model/                    # Trained YOLOv11 model (.pt file)
├── manage.py                 # Django management script
└── requirements.txt          # Dependencies
```

## 🚀 Installation & Run

1. **Clone the repo**:

```bash
git clone https://github.com/KozakOleksandr/web-drone-object-detection.git
cd web-drone-object-detection/diploma_object_detection
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run migrations**:

```bash
python manage.py migrate
```

5. **Create superuser (optional)**:

```bash
python manage.py createsuperuser
```

6. **Run the server**:

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` in your browser.

## 📸 Example Use Cases

* Upload an image or video to detect military equipment.
* Start webcam or drone video stream to detect objects in real time.
* Review logged activity of users and detections.

## 📚 Academic Context

This system was developed as part of a Bachelor's qualification work titled:

> "Web-Oriented Information System for Identifying Objects from Unmanned Aerial Vehicles"

## 📜 License

This project is for academic use. Licensing terms can be added later if required.
