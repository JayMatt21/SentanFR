import cv2
import os
from firebase_admin import storage, credentials, initialize_app

cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred, {
    'storageBucket': 'senfrostfr.firebasestorage.app'
})

cap = cv2.VideoCapture(0)
user_id = input("Enter User ID: ")
out_dir = f"registration/{user_id}"
os.makedirs(out_dir, exist_ok=True)

frame_count = 0

while frame_count < 30:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("Register Face", frame)
    cv2.imwrite(f"{out_dir}/frame_{frame_count}.jpg", frame)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

bucket = storage.bucket()
for fname in os.listdir(out_dir):
    blob = bucket.blob(f"known_faces/{user_id}/{fname}")
    blob.upload_from_filename(os.path.join(out_dir, fname))
print("✅ Uploaded to Firebase.")
