import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime
import pytz

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
current_date = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"Attendance/Attendance_{current_date}.csv"


# Date and Time Update Function
def update_time():
    now = datetime.now().strftime("%d-%B-%Y | %H:%M:%S")
    date_time_label.config(text=now)
    window.after(1000, update_time)  # Schedule the update every 1 second


# Update the time every second
update_time()


# Ensure the directory exists
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Capture Images
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


# Train Images
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


# Ensure the attendance directory exists
attendance_dir = r"C:\Users\mayur\PycharmProjects\face_rec_att_sys\Attendance"
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

ist = pytz.timezone('Asia/Kolkata')

# Define session ranges
SESSION_RANGES = {
    "Morning": (6, 12),  # 6 AM to 12 PM
    "Afternoon": (12, 18),  # 12 PM to 6 PM
    "Evening": (18, 24),  # 6 PM to 12 AM
}


def get_current_session():
    current_hour = datetime.now(ist).hour
    if 9 <= current_hour < 12:
        return "Morning"
    elif 12 <= current_hour < 17:
        return "Afternoon"
    else:
        return "Evening"


# Recognize Faces and Mark Attendance
def recognize_faces():
    # Set up recognizer and face cascade
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainedModel/FaceRecognizer.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Initialize DataFrame to store attendance
    df = pd.DataFrame(columns=["ID", "Name", "Date", "Time", "Session"])

    # Load existing attendance file if it exists
    attendance_dir = "Attendance"  # Replace with the actual directory path

    existing_attendance_file = os.path.join(attendance_dir, f"Attendance_{datetime.now().date()}.csv")

    if os.path.exists(existing_attendance_file):
        df = pd.read_csv(existing_attendance_file)

    # Open camera
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence < 50:  # Check if face recognition is successful
                user_name = f"User {user_id}"
                current_time = datetime.now(ist).strftime('%H:%M:%S')
                current_date = datetime.now(ist).date()

                # Get the current session (you should already have this variable)
                current_session = get_current_session()  # Assuming you have a function that retrieves the current session.

                # Ensure that 'Session' column exists in the DataFrame before proceeding
                if 'Session' in df.columns:
                    # The 'Session' column exists, so you can proceed with your logic
                    if (df['Session'] == current_session).any():
                        # Logic to process attendance when session matches
                        print("Session found!")
                        # Your attendance logic goes here (for example, mark attendance or update)
                        # Example:
                        # df.loc[df['Session'] == current_session, 'Attendance'] = 'Present'
                    else:
                        print("Session does not match current session.")
                        # Add any logic you need here when session doesn't match
                else:
                    # 'Session' column does not exist, handle this case
                    print("Session column is missing, adding it now.")
                    df['Session'] = None  # Add 'Session' column with default values (None)
                    print("Session column added.")

                    # Optionally, you can now set the session for the DataFrame
                    # Example: You could set all rows to the current session or add the specific session you need
                    df['Session'] = current_session  # Set the 'Session' column to current session

                    print("Session column has been updated with the current session.")
                current_session = get_current_session()

                # Check if the user has already marked attendance for the current session
                if not ((df['ID'] == user_id) & (df['Date'] == str(current_date)) & (
                        df['Session'] == current_session)).any():
                    # Add attendance for the user if not already marked for this session
                    new_row = {
                        'ID': user_id,
                        'Name': user_name,
                        'Time': current_time,
                        'Date': str(current_date),
                        'Session': current_session
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                    # Draw a green rectangle around recognized face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{user_name} - {current_session}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (255, 255, 255), 2)
                else:
                    # If already marked in this session, show "Already Marked"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(frame, "Already Marked", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            else:
                # Draw a red rectangle for unknown faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show video stream
        cv2.imshow("Recognizing Faces", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the updated attendance to the CSV file
    df.to_csv(existing_attendance_file, index=False)

    # Release camera and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

    # Confirm attendance is saved
    print(f"Attendance has been marked and saved to {existing_attendance_file}!")

def initialize_attendance_file(attendance_file):
    # Define the required columns
    required_columns = ["ID", "Name", "Date", "Time", "Session"]

    # Check if the file exists
    if os.path.exists(attendance_file):
        # Load the existing file
        df = pd.read_csv(attendance_file)

        # Add the missing columns (if any)
        for column in required_columns:
            if column not in df.columns:
                df[column] = None  # Add missing columns with default values
    else:
        # Create a new DataFrame with the required columns if the file doesn't exist
        df = pd.DataFrame(columns=required_columns)

    return df

# Use the function to initialize the DataFrame before starting attendance processing
attendance_file = f"Attendance/Attendance_{current_date}.csv"
df = initialize_attendance_file(attendance_file)

# GUI Elements
lbl_id = tk.Label(window, text="Enter ID", font=('times', 14, 'bold'))
lbl_id.pack()
txt_id = tk.Entry(window)
txt_id.pack()

lbl_name = tk.Label(window, text="Enter Name", font=('times', 14, 'bold'))
lbl_name.pack()
txt_name = tk.Entry(window)
txt_name.pack()

btn_capture = tk.Button(window, text="Capture Images", font=('times', 14, 'bold'), bg="lightblue",
                        command=capture_images)
btn_capture.pack(pady=10)

btn_train = tk.Button(window, text="Train Images", font=('times', 14, 'bold'), bg="lightgreen", command=train_images)
btn_train.pack(pady=10)

btn_recognize = tk.Button(window, text="Recognize Faces", font=('times', 14, 'bold'), bg="lightcoral",
                          command=recognize_faces)
btn_recognize.pack(pady=10)

# Start the GUI loop
window.mainloop()
