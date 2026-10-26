# Side Quest - UGV & UAV Drone System 🚁🚗

Welcome to the official repository of **Team Side Quest**! 
During this semester, we built a comprehensive, dual-drone system consisting of an Unmanned Ground Vehicle (UGV) and an Unmanned Aerial Vehicle (UAV). Despite having very little prior experience with drones, we overcame numerous challenges, learned a great deal, and proudly secured **5th place** in the drone competition!

---

## 📸 Gallery
*(Placeholders for your photos - replace the links with your actual image paths)*

<div align="center">
  <img src="link_do_zdjecia_zespolu.jpg" alt="Team Side Quest" width="400"/>
  <br>
  <i>Team Side Quest in action!</i>
</div>

<div align="center">
  <img src="link_do_zdjecia_ugv.jpg" alt="UGV Rolling Drone" width="45"/>
  <img src="link_do_zdjecia_uav.jpg" alt="UAV Flying Drone" width="45"/>
</div>

---

## 🚀 Project Overview

Our mission was to design, build, and program two distinct vehicles capable of detecting specific objects, reading QR codes, and manipulating items using custom-built grabbers. The entire ecosystem communicates via **ROS2**, streaming FPV video and synchronized telemetry to a ground station laptop.

### 🚗 UGV (Rolling Drone)
- **Vision System:** Equipped with an FPV camera (Picamera2) capturing frames at 60 FPS.
- **AI & Detection:** Runs a custom YOLO-based object detection model capable of recognizing 10 different target classes (e.g., military vehicles, civilian cars) and a QReader for scanning QR codes.
- **Hardware:** Features a custom stepper-motor grabber for physical interaction.
- **Control:** Fully keyboard-controlled with a multithreaded architecture for smooth, non-blocking driving and vision processing.

### 🚁 UAV (Flying Drone)
- **Flight Controller:** Operated using an ArduPilot-based control system communicating via MAVLink (`pymavlink`).
- **Design Philosophy:** Built to be as lightweight and agile as possible for maximum reliability and ease of control.
- **Payload:** Like the UGV, it features an integrated grabber and runs the same YOLO/QR detection stack, binding visual detections directly to GPS coordinates and telemetry data.

### 💻 Ground Station (Laptop Receiver)
- A custom OpenCV-based graphical interface (HUD).
- Subscribes to ROS2 topics to display real-time video feeds with annotated YOLO bounding boxes.
- Overlays critical telemetry in real-time (GPS Fix, Altitude, Attitude/Horizon, Battery, and Flight Mode).
- Displays a prominent alert banner when target objects or QR codes are detected, including their exact GPS location.

---

## 🧠 Software Architecture & Tech Stack

- **Framework:** ROS2 (Jazzy)
- **Languages:** Python 3
- **Computer Vision:** OpenCV, `libcamera` (Picamera2)
- **Machine Learning:** Ultralytics YOLOv8 (Custom trained model)
- **QR Decoding:** `qreader`
- **Flight Comm:** `pymavlink` (MAVLink protocol)
- **Hardware Control:** `gpiozero`, custom Stepper driver

### Main ROS2 Nodes
1. `DronJezdzacyAutko` (UGV Node): Handles multi-threaded camera capture, asynchronous YOLO+QR inference, motor PWM control, and gripper stepping.
2. `DronLatajacy` (UAV Node): Interfaces with the ArduPilot FC via UART/MAVLink, fetches telemetry at 10Hz, runs vision pipelines, and pairs detections with current GPS data.
3. `OdbiornikLaptop` (Ground Station Node): Receives compressed images, telemetry, and custom `Detection` messages, rendering a full FPV HUD with an artificial horizon and status overlays.

---

## 🏆 Conclusion
Taking part in this competition was an amazing adventure. We worked incredibly hard, overcame many setbacks, and built a highly complex, multi-agent robotic system from scratch. 

**Good job, Side Quest!** 🚀