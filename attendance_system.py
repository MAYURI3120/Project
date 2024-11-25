import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import datetime

# Initialize GUI
window = tk.Tk()
window.title("Face Recognition Based Attendance System")
window.geometry('500x600')
window.configure(background="white")

# Title
title = tk.Label(window, text="Face Recognition Attendance System", bg="black", fg="white",
                 font=('times', 20, 'bold'))
title.pack(fill=tk.X)

# Date and Time
date_time_label = tk.Label(window, text="", bg="black", fg="yellow", font=('times', 14, 'bold'))
date_time_label.pack(fill=tk.X)

def update_time():
    now = datetime.datetime.now().strftime("%d-%B-%Y | %H:%M:%S")
    date_time_label.config(text=now)
    window.after(1000, update_time)
update_time()

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def capture_images():
    user_id = txt_id.get()
    user_name = txt_name.get()
    if user_id == "" or user_name == "":
        messagebox.showerror("Error", "All fields are required!")
        return

    assure_path_exists("TrainingImages/")
    cam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    sample_count = 0

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sample_count += 1
            cv2.imwrite(f"TrainingImages/{user_name}.{user_id}.{sample_count}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Capturing Images", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", f"Images Captured for {user_name}")

def train_images():
    assure_path_exists("TrainedModel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []

    image_paths = [os.path.join("TrainingImages", img) for img in os.listdir("TrainingImages")]
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')
        img_np = np.array(img, 'uint8')
        user_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces.append(img_np)
        ids.append(user_id)

    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainedModel/FaceRecognizer.yml")
    messagebox.showinfo("Info", "Training Complete!")

def recognize_faces():
    assure_path_exists("Attendance/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainedModel/FaceRecognizer.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 50:
                user_name = f"User {user_id}"
                timestamp = datetime.datetime.now()
                df.loc[len(df)] = [user_id, user_name, timestamp.date(), timestamp.time()]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, user_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Recognizing Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    df.to_csv(f"Attendance/Attendance_{datetime.datetime.now().date()}.csv", index=False)
    cam.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Attendance Marked!")

lbl_id = tk.Label(window, text="Enter ID", font=('times', 14, 'bold'))
lbl_id.pack()
txt_id = tk.Entry(window)
txt_id.pack()

lbl_name = tk.Label(window, text="Enter Name", font=('times', 14, 'bold'))
lbl_name.pack()
txt_name = tk.Entry(window)
txt_name.pack()

btn_capture = tk.Button(window, text="Capture Images", command=capture_images, bg="green", fg="white", font=('times', 14, 'bold'))
btn_capture.pack()

btn_train = tk.Button(window, text="Train Images", command=train_images, bg="blue", fg="white", font=('times', 14, 'bold'))
btn_train.pack()

btn_recognize = tk.Button(window, text="Recognize Faces", command=recognize_faces, bg="orange", fg="white", font=('times', 14, 'bold'))
btn_recognize.pack()

# Start the GUI loop
window.mainloop()
