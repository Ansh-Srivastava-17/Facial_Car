import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import threading
import time
import pandas as pd
import random 

# Family database
family_database = {
    "Person 1": {"age": 45, "encoding": None, "authorized": True},
    "Person 2": {"age": 42, "encoding": None, "authorized": False},
    "Person 3": {"age": 18, "encoding": None, "authorized": False},
    "Person 4": {"age": 16, "encoding": None, "authorized": False},
    "Person 5": {"age": 14, "encoding": None, "authorized": False},
    "Person 6": {"age": 10, "encoding": None, "authorized": False}
}

# Create outside people database
outside_database = pd.DataFrame({
    "Name": [f"Outside Person {i}" for i in range(1, 11)],
    "Age": [random.randint(14, 30) for _ in range(10)],
    "Fingerprint": [f"FP{i}" for i in range(1, 11)],
    "Photo": [f"outside_person_{i}.jpeg" for i in range(1, 11)]
})

# Save outside database to Excel
outside_database.to_excel("outside_people_database.xlsx", index=False)

recording = False
out = None
car_moving = False
current_driver = None
last_auth_time = 0
AUTH_COOLDOWN = 30

def load_family_photos():
    for name, info in family_database.items():
        if info["authorized"]:
            print(f"{name} is pre-authorized. Skipping photo loading.")
            continue
        
        file_path = f"{name}.jpeg"
        if not os.path.exists(file_path):
            print(f"Warning: Photo for {name} not found at {file_path}")
            continue
        
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            print(f"Warning: No face detected in the photo of {name}")
            continue
        
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        family_database[name]["encoding"] = encoding
        print(f"Successfully loaded and encoded photo for {name}")

def start_recording(frame):
    global recording, out
    if not recording:
        recording = True
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"car_recording_{current_time}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        print(f"Started recording: {filename}")

def stop_recording():
    global recording, out
    if recording:
        recording = False
        out.release()
        print("Stopped recording")

def record_frame(frame):
    if recording:
        out.write(frame)

def simulate_car_motion():
    global car_moving
    car_moving = True
    print("Car started moving.")
    time.sleep(10) 
    car_moving = False
    print("Car stopped.")

def simulate_fingerprint_scan():
    print("Please place your finger on the scanner.")
    time.sleep(5) 
    return f"FP{random.randint(1, 10)}"  

def check_outside_person(fingerprint):
    match = outside_database[outside_database['Fingerprint'] == fingerprint]
    if not match.empty:
        person = match.iloc[0]
        if person['Age'] >= 18:
            return True, person['Name'], person['Age']
    return False, None, None

def scan_driver():
    global recording, car_moving, current_driver, last_auth_time
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Unable to access the camera.")
        return
    
    last_detection_time = time.time()
    motion_thread = None
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to capture frame from camera.")
            break
        
        current_time = time.time()
        

        if current_time - last_detection_time > 5:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if face_encodings:
                last_detection_time = current_time
                

                if not current_driver or (current_time - last_auth_time > AUTH_COOLDOWN):
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces([member["encoding"] for member in family_database.values() if member["encoding"] is not None], face_encoding)
                        
                        if True in matches:
                            matched_index = matches.index(True)
                            name = list(family_database.keys())[matched_index]
                            if family_database[name]["authorized"] or family_database[name]["age"] >= 18:
                                current_driver = name
                                last_auth_time = current_time
                                display_message(f"Welcome, {name}! The car is ready to move.")
                                start_car(motion_thread, frame)
                            else:
                                display_message(f"The car will not move. {name}, you are not an authorized user. Minimum age to drive is 18.")
                        else:
                            print("Person not recognized in family database. Initiating fingerprint scan.")
                            fingerprint = simulate_fingerprint_scan()
                            authorized, name, age = check_outside_person(fingerprint)
                            if authorized:
                                current_driver = name
                                last_auth_time = current_time
                                display_message(f"Welcome, {name}! You are authorized to drive. The car is ready to move.")
                                start_car(motion_thread, frame)
                            else:
                                display_message("Unauthorized user. The car will not move.")
            elif current_time - last_detection_time > 10 and car_moving:
                print("Driver not detected for 10 seconds. Stopping the car.")
                stop_car()
        
        if car_moving:
            record_frame(frame)
        elif recording:
            stop_recording()
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    stop_recording()
    if motion_thread and motion_thread.is_alive():
        motion_thread.join()

def start_car(motion_thread, frame):
    global car_moving
    if not car_moving:
        start_recording(frame)
        motion_thread = threading.Thread(target=simulate_car_motion)
        motion_thread.start()

def stop_car():
    global car_moving, current_driver
    car_moving = False
    current_driver = None
    print("Car stopped.")

def display_message(message):
    print(message)


def main():
    print("Initializing Enhanced Car Authentication System...")
    load_family_photos()
    
    if all(not member["authorized"] and member["encoding"] is None for member in family_database.values()):
        print("Error: No valid face encodings were created and no pre-authorized users. Please check your image files.")
        return
    
    print("System ready. Scanning for driver...")
    scan_driver()

if __name__ == "__main__":
    main()