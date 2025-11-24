import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime
import os

class AdvancedFaceDetectionApp:
    def __init__(self):
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        
        self.detection_active = False
        self.cap = None
        self.features_enabled = {
            'faces': True,
            'eyes': True,
            'smiles': True,
            'blur': False,
            'landmarks': False,
            'emotion': False
        }
        self.face_count = 0
        self.detection_history = []
        self.snapshot_count = 0
        
        
        self.colors = {
            'faces': [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)],
            'eyes': [(255, 255, 255), (200, 200, 255)],
            'smiles': [(0, 165, 255), (0, 200, 255)],
            'landmarks': [(0, 255, 255), (255, 255, 0)]
        }
        
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Face Detection")
        self.root.geometry("800x700")
        self.root.configure(bg='#2c3e50')
        
        
        title_label = tk.Label(self.root, text="🎭 Advanced Face Detection", 
                              font=("Arial", 18, "bold"), 
                              bg='#2c3e50', fg='#ecf0f1')
        title_label.pack(pady=10)
        
        
        control_frame = tk.LabelFrame(self.root, text="Controls", 
                                     font=("Arial", 12, "bold"),
                                     bg='#34495e', fg='#ecf0f1')
        control_frame.pack(pady=10, padx=20, fill="x")
        
        
        btn_frame = tk.Frame(control_frame, bg='#34495e')
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="🚀 Start Detection", 
                                  command=self.toggle_detection,
                                  font=("Arial", 12, "bold"),
                                  bg='#27ae60', fg='white',
                                  width=15, height=2)
        self.start_btn.pack(side="left", padx=10)
        
        snapshot_btn = tk.Button(btn_frame, text="📸 Take Snapshot", 
                                command=self.take_snapshot,
                                font=("Arial", 12, "bold"),
                                bg='#3498db', fg='white',
                                width=15, height=2)
        snapshot_btn.pack(side="left", padx=10)
        
        
        features_frame = tk.LabelFrame(self.root, text="Detection Features", 
                                      font=("Arial", 12, "bold"),
                                      bg='#34495e', fg='#ecf0f1')
        features_frame.pack(pady=10, padx=20, fill="x")
        
        
        features_grid = tk.Frame(features_frame, bg='#34495e')
        features_grid.pack(pady=10)
        
        self.feature_vars = {}
        features = [
            ('Faces', 'faces', '#27ae60'),
            ('Eyes', 'eyes', '#3498db'),
            ('Smiles', 'smiles', '#e74c3c'),
            ('Face Blur', 'blur', '#9b59b6'),
            ('Facial Landmarks', 'landmarks', '#f39c12'),
            ('Emotion Zones', 'emotion', '#1abc9c')
        ]
        
        for i, (text, key, color) in enumerate(features):
            var = tk.BooleanVar(value=self.features_enabled[key])
            self.feature_vars[key] = var
            
            cb = tk.Checkbutton(features_grid, text=text, variable=var,
                               command=lambda k=key: self.toggle_feature(k),
                               font=("Arial", 10, "bold"),
                               bg='#34495e', fg=color,
                               selectcolor='#2c3e50')
            cb.grid(row=i//3, column=i%3, sticky="w", padx=10, pady=5)
        
        
        stats_frame = tk.LabelFrame(self.root, text="Real-time Statistics", 
                                   font=("Arial", 12, "bold"),
                                   bg='#34495e', fg='#ecf0f1')
        stats_frame.pack(pady=10, padx=20, fill="x")
        
        self.stats_text = tk.Text(stats_frame, height=6, width=70,
                                 font=("Courier", 10),
                                 bg='#2c3e50', fg='#ecf0f1',
                                 relief='flat')
        self.stats_text.pack(pady=10, padx=10)
        
        
        settings_frame = tk.LabelFrame(self.root, text="Detection Settings", 
                                      font=("Arial", 12, "bold"),
                                      bg='#34495e', fg='#ecf0f1')
        settings_frame.pack(pady=10, padx=20, fill="x")
        
        
        scale_frame = tk.Frame(settings_frame, bg='#34495e')
        scale_frame.pack(pady=5)
        
        tk.Label(scale_frame, text="Detection Sensitivity:", 
                bg='#34495e', fg='#ecf0f1').pack(side="left")
        
        self.scale_var = tk.DoubleVar(value=1.1)
        scale_slider = tk.Scale(scale_frame, from_=1.01, to=1.5, 
                               resolution=0.01, orient="horizontal",
                               variable=self.scale_var,
                               bg='#34495e', fg='#ecf0f1',
                               length=200)
        scale_slider.pack(side="left", padx=10)
        
        
        self.status_var = tk.StringVar(value="Ready to start face detection")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                             relief="sunken", anchor="w",
                             font=("Arial", 10),
                             bg='#34495e', fg='#ecf0f1')
        status_bar.pack(side="bottom", fill="x")
        
        
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")
    
    def toggle_feature(self, feature):
        self.features_enabled[feature] = self.feature_vars[feature].get()
    
    def toggle_detection(self):
        if not self.detection_active:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        self.detection_active = True
        self.start_btn.config(text="🛑 Stop Detection", bg='#e74c3c')
        self.status_var.set("Face detection active - Press 'Q' to stop")
        
        
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
    
    def stop_detection(self):
        self.detection_active = False
        self.start_btn.config(text="🚀 Start Detection", bg='#27ae60')
        self.status_var.set("Detection stopped")
        
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def draw_facial_landmarks(self, frame, face):
        x, y, w, h = face
        color = self.colors['landmarks'][0]
        
        
        center_x, center_y = x + w//2, y + h//2
        
        
        cv2.circle(frame, (center_x, y + h//6), 5, color, -1)
        
        
        cv2.circle(frame, (x + w//3, y + h//3), 5, color, -1)
        cv2.circle(frame, (x + 2*w//3, y + h//3), 5, color, -1)
        
        
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        
        cv2.circle(frame, (x + w//3, y + 2*h//3), 5, color, -1)
        cv2.circle(frame, (x + 2*w//3, y + 2*h//3), 5, color, -1)
        cv2.circle(frame, (center_x, y + 5*h//6), 5, color, -1)
    
    def draw_emotion_zones(self, frame, face):
        x, y, w, h = face
        colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
        
        
        zone_height = h // 3
        for i in range(3):
            zone_y = y + i * zone_height
            cv2.rectangle(frame, (x, zone_y), (x + w, zone_y + zone_height), 
                         colors[i], 2)
            cv2.putText(frame, f'Zone {i+1}', (x + 5, zone_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    def apply_face_blur(self, frame, faces):
        for (x, y, w, h) in faces:
            
            face_region = frame[y:y+h, x:x+w]
            
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            
            frame[y:y+h, x:x+w] = blurred_face
    
    def run_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam")
            self.stop_detection()
            return
        
        while self.detection_active:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_var.get(),
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            self.face_count = len(faces)
            
            
            if self.features_enabled['blur'] and len(faces) > 0:
                self.apply_face_blur(frame, faces)
            
            
            for i, (x, y, w, h) in enumerate(faces):
                face_color = self.colors['faces'][i % len(self.colors['faces'])]
                
                
                if self.features_enabled['faces']:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 3)
                    cv2.putText(frame, f'Face {i+1}', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, face_color, 2)
                
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                
                if self.features_enabled['eyes']:
                    eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                    for (ex, ey, ew, eh) in eyes:
                        eye_color = self.colors['eyes'][0]
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
                        cv2.circle(roi_color, (ex+ew//2, ey+eh//2), 2, eye_color, -1)
                
                
                if self.features_enabled['smiles']:
                    smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                    for (sx, sy, sw, sh) in smiles:
                        smile_color = self.colors['smiles'][0]
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), smile_color, 2)
                
                
                if self.features_enabled['landmarks']:
                    self.draw_facial_landmarks(frame, (x, y, w, h))
                
                
                if self.features_enabled['emotion']:
                    self.draw_emotion_zones(frame, (x, y, w, h))
            
           
            self.add_statistics_overlay(frame, len(faces))
            
            
            cv2.imshow('Advanced Face Detection', frame)
            
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.take_snapshot(frame)
        
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.stop_detection()
    
    def add_statistics_overlay(self, frame, face_count):
        
        current_time = datetime.now()
        self.detection_history.append((current_time, face_count))
        
        
        if len(self.detection_history) > 50:
            self.detection_history.pop(0)
        
        
        stats_text = [
            f"Faces Detected: {face_count}",
            f"Detection Scale: {self.scale_var.get():.2f}",
            f"Time: {current_time.strftime('%H:%M:%S')}",
            "Features: " + ", ".join([k for k, v in self.features_enabled.items() if v])
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, text, (10, 30 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        
        self.update_gui_stats(face_count)
    
    def update_gui_stats(self, face_count):
        stats = f"""
╔══════════════════════════════════════╗
║           REAL-TIME STATS           ║
╠══════════════════════════════════════╣
║ Faces Detected: {face_count:2d}                    ║
║ Active Features: {sum(self.features_enabled.values()):1d}/6              ║
║ Detection Scale: {self.scale_var.get():.2f}              ║
║ Snapshots Taken: {self.snapshot_count:2d}                 ║
║ Status: {'ACTIVE' if self.detection_active else 'READY'}              ║
╚══════════════════════════════════════╝
"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats)
    
    def take_snapshot(self, frame=None):
        if frame is None:
            if self.cap and self.detection_active:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
            else:
                messagebox.showwarning("Warning", "No active video feed")
                return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/snapshot_{timestamp}_{self.snapshot_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        self.snapshot_count += 1
        self.status_var.set(f"Snapshot saved: {filename}")
        
        
        cv2.putText(frame, "SNAPSHOT SAVED!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow('Advanced Face Detection', frame)
    
    def run(self):
        self.update_gui_stats(0)
        self.root.mainloop()
        
        
        self.stop_detection()

def main():
    app = AdvancedFaceDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
