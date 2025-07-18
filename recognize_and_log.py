import cv2
import face_recognition
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import numpy as np

# === Firebase ===
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Load all known faces from Firestore ===
employees_ref = db.collection('employees')
docs = employees_ref.stream()

known_face_encodings = []
known_face_names = []

for doc in docs:
    data = doc.to_dict()
    known_face_encodings.append(np.array(data['face_encoding']))  # Convert list to np.array
    known_face_names.append(data['name'])

print(f"Loaded {len(known_face_names)} employees from Firestore.")

# === RTSP / Webcam ===
rtsp_url = "rtsp://Jarvis2:Passw0rd2@192.168.1.7:554/stream1"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # === Log attendance ===
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db.collection('attendance_logs').add({
                'name': name,
                'timestamp': timestamp
            })
            print(f"✅ Logged attendance for {name} at {timestamp}")

        # Draw box and label
        top, right, bottom, left = face_location
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
