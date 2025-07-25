import os
import cv2
import time
import numpy as np
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImagesn

# ✅ Proper gpiozero import and setup for Pi 5
import gpiozero
from gpiozero.pins.lgpio import LGPIOFactory
gpiozero.Device.pin_factory = LGPIOFactory()

from gpiozero import OutputDevice
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import onnxruntime as ort
from datetime import datetime
import socketio
import threading

# Initialize Socket.IO with stable connection
sio = socketio.Client(reconnection=True, reconnection_attempts=5)

# GPIO Setup using gpiozero
RELAY_PIN = 17
BUZZER_PIN = 27
relay = OutputDevice(RELAY_PIN, active_high=True, initial_value=False)
buzzer = OutputDevice(BUZZER_PIN, active_high=True, initial_value=False)

# Socket.IO Events
@sio.event
def connect():
    print("Connected to server with SID:", sio.sid)

@sio.event
def disconnect():
    print("Disconnected from server")

@sio.on('door_status')
def on_door_status(data):
    locked = data.get('locked', True)
    print(f"Door status received from {data.get('source')}: {'LOCKED' if locked else 'UNLOCKED'}")
    if locked:
        relay.off()
    else:
        relay.on()

def relock_and_emit():
    relay.off()
    print("Door auto-locked after 5 seconds")
    try:
        sio.emit('door_status', {'locked': True, 'source': 'camera'})
    except Exception as e:
        print("Emit failed:", e)

# Connect to server in background
def connect_socket():
    try:
        sio.connect(
            'https://server-door-lock.onrender.com',
            transports=['websocket'],
            socketio_path='/socket.io',
            headers={'Origin': 'https://server-door-lock.onrender.com'}
        )
    except Exception as e:
        print(f"Socket.IO connection failed: {e}")

# Face Recognition Setup
VEC_PATH = "face_vectors.npy"
MODEL_PATH = "arcface.onnx"
face_vectors = np.load(VEC_PATH, allow_pickle=True).item()

ort_session = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = ort_session.get_inputs()[0].name

mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# Face Processing Functions
def get_embedding(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if result.detections:
        bbox = result.detections[0].location_data.relative_bounding_box
        h, w = image.shape[:2]
        x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
        x2, y2 = x1 + int(bbox.width * w), y1 + int(bbox.height * h)
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        face = cv2.resize(face_crop, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face = (face - 0.5) / 0.5
        input_tensor = face.astype(np.float32)[np.newaxis, :]
        embedding = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]
        return embedding / np.linalg.norm(embedding)
    return None

def recognize(embedding):
    for name, vec in face_vectors.items():
        vec = np.array(vec)
        if embedding.shape != vec.shape:
            continue
        score = cosine_similarity([embedding], [vec])[0][0]
        if score > 0.7:
            return name
    return "unknown"

def log(name):
    with open("access_log.txt", "a") as f:
        f.write(f"{datetime.now()} - {name}\n")

# Main Application
class DetectApp(ctk.CTk):
    def _init_(self):
        super()._init_()
        self.title("Smart Door Detection")
        self.geometry("1024x600")
        self.status = ctk.StringVar(value="🔍 Scanning...")
        self.cap = cv2.VideoCapture(0)
        self.frame_skip = 0
        self.last_access_time = 0
        self.last_denied_time = 0
        self.cooldown = 10
        
        self.setup_ui()
        self.update_camera()
        self.after(2000, lambda: threading.Thread(target=connect_socket, daemon=True).start())

    def setup_ui(self):
        ctk.CTkLabel(self, textvariable=self.status, font=("Arial", 20), text_color="blue").pack(pady=10)
        self.cam_label = ctk.CTkLabel(self, width=960, height=480)
        self.cam_label.pack()

    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_camera)
            return

        if self.frame_skip == 0:
            embed = get_embedding(frame)
            now = time.time()

            if embed is not None:
                name = recognize(embed)
                if name != "unknown" and now - self.last_access_time > self.cooldown:
                    self.status.set(f"✅ Access Granted: {name}")
                    relay.on()
                    try:
                        sio.emit('door_status', {'locked': False, 'source': 'camera'})
                    except:
                        pass
                    self.after(5000, relock_and_emit)
                    self.last_access_time = now
                    log(f"✅ {name}")
                elif name == "unknown" and now - self.last_denied_time > self.cooldown:
                    self.status.set("❌ Unauthorized!")
                    buzzer.on()
                    self.after(3000, buzzer.off)
                    self.last_denied_time = now
                    log("❌ Unauthorized")

        self.frame_skip = (self.frame_skip + 1) % 5

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((960, 480))
        imgtk = CTkImage(light_image=img, size=(960, 480))
        self.cam_label.configure(image=imgtk)
        self.cam_label.image = imgtk

        self.after(10, self.update_camera)

    def _del_(self):
        self.cap.release()

if _name_ == "_main_":
    ctk.set_appearance_mode("light")
    app = DetectApp()
    app.mainloop()