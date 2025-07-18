import cv2
from deepface import DeepFace
import os
import threading
import time
import firebase_admin
from firebase_admin import credentials, storage
import pandas as pd
from numpy import dot
from numpy.linalg import norm

# === INIT FIREBASE ===
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'senfrostfr.firebasestorage.app'
})

# === CONFIG ===
db_path = "known_faces"
model_name = "Facenet512"
rtsp_url = "rtsp://Jarvis2:Passw0rd2@192.168.1.7:554/stream1"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ System running. Press Q to quit.")

# === SYNC EMBEDDINGS ===
def sync_from_firebase():
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix="known_faces/")
    os.makedirs(db_path, exist_ok=True)
    for blob in blobs:
        if blob.name.endswith(".jpg") or blob.name.endswith(".png"):
            local_path = os.path.join(db_path, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
    print("✅ Synced from Firebase.")

sync_from_firebase()

db_representations = DeepFace.represent(
    img_path = db_path,
    model_name = model_name,
    detector_backend = 'opencv',
    enforce_detection = False
)
db_representations = pd.DataFrame(db_representations)

def find_cosine_distance(source, test):
    return 1 - dot(source, test) / (norm(source) * norm(test))

recognized_faces = []
lock = threading.Lock()

def recognize_face(frame):
    global recognized_faces
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    results = []
    try:
        faces = DeepFace.extract_faces(
            img_path=temp_path,
            detector_backend='opencv',
            enforce_detection=False
        )
        if len(faces) == 0:
            results.append(("No Face", None))
        else:
            for face in faces:
                coords = face['facial_area']
                face_img = face['face']
                cv2.imwrite("temp_face.jpg", face_img)
                embedding = DeepFace.represent(
                    img_path="temp_face.jpg",
                    model_name=model_name,
                    detector_backend='opencv',
                    enforce_detection=False
                )[0]["embedding"]

                min_dist = 1
                identity = "Unknown"

                for _, db_row in db_representations.iterrows():
                    db_embedding = db_row["embedding"]
                    dist = find_cosine_distance(embedding, db_embedding)
                    if dist < min_dist and dist < 0.4:
                        min_dist = dist
                        identity = os.path.basename(db_row["identity"])

                results.append((identity, coords))
        with lock:
            recognized_faces = results
    except Exception as e:
        print(f"❌ Error: {e}")

last_time = 0
interval = 1

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if time.time() - last_time > interval:
        last_time = time.time()
        threading.Thread(target=recognize_face, args=(frame.copy(),)).start()

    with lock:
        faces_to_draw = recognized_faces.copy()

    for name, coords in faces_to_draw:
        if coords is None:
            continue
        x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]
        color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
