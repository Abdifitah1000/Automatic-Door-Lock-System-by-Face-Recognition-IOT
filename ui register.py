import cv2, os, numpy as np
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
from customtkinter import CTkImage
import mediapipe as mp
import onnxruntime as ort

# === Paths ===
SAVE_DIR = "face_data"
os.makedirs(SAVE_DIR, exist_ok=True)
VEC_PATH = "face_vectors.npy"
MODEL_PATH = "arcface.onnx"

# === Load ONNX ArcFace Model ===
ort_session = ort.InferenceSession(MODEL_PATH)
INPUT_NAME = ort_session.get_inputs()[0].name  # dynamically get correct input name

# === Load existing face vectors ===
face_vectors = np.load(VEC_PATH, allow_pickle=True).item() if os.path.exists(VEC_PATH) else {}

# === MediaPipe Setup ===
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

# === Extract Face Embedding ===
def get_face_embedding(image):
    face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detector.process(face_rgb)
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        h, w = image.shape[:2]
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        face_crop = image[y1:y2, x1:x2]
        face = cv2.resize(face_crop, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.transpose(face, (2, 0, 1)) / 255.0
        face = (face - 0.5) / 0.5
        input_tensor = face.astype(np.float32)[np.newaxis, :]
        embedding = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]
        return embedding / np.linalg.norm(embedding)
    return None

# === CustomTkinter App ===
class RegisterApp(ctk.CTk):
    def _init_(self):
        super()._init_()
        self.title("Register Face")
        self.geometry("1024x600")
        self.name_var = ctk.StringVar()
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        top = ctk.CTkFrame(self)
        top.pack(pady=10)
        ctk.CTkLabel(top, text="Name:", font=("Arial", 18)).grid(row=0, column=0, padx=10)
        ctk.CTkEntry(top, textvariable=self.name_var, font=("Arial", 18), width=250).grid(row=0, column=1, padx=10)
        ctk.CTkButton(top, text="📷 Capture & Register", font=("Arial", 18), command=self.register).grid(row=0, column=2, padx=10)

        self.cam_label = ctk.CTkLabel(self, text="", width=960, height=480)
        self.cam_label.pack(pady=10)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((960, 480))
            imgtk = CTkImage(light_image=img, size=(960, 480))
            self.cam_label.configure(image=imgtk)
            self.cam_label.image = imgtk
        self.after(10, self.update_frame)

    def register(self):
        name = self.name_var.get().strip().lower()
        if not name or self.frame is None:
            messagebox.showerror("Error", "Name or camera frame missing.")
            return
        vec = get_face_embedding(self.frame)
        if vec is not None:
            face_vectors[name] = vec
            cv2.imwrite(os.path.join(SAVE_DIR, f"{name}.jpg"), self.frame)
            np.save(VEC_PATH, face_vectors)
            messagebox.showinfo("Success", f"{name} registered.")
        else:
            messagebox.showwarning("No Face", "Face not detected. Try again.")

if _name_ == "_main_":
    ctk.set_appearance_mode("light")
    RegisterApp().mainloop()