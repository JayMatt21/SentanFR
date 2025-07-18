from firebase_admin import storage, credentials, initialize_app
import os

cred = credentials.Certificate("serviceAccountKey.json")
initialize_app(cred, {
    'storageBucket': 'senfrostfr.firebasestorage.app'
})

bucket = storage.bucket()
blobs = bucket.list_blobs(prefix="known_faces/")
os.makedirs("known_faces", exist_ok=True)

for blob in blobs:
    if blob.name.endswith(".jpg"):
        local_path = os.path.join("known_faces", os.path.basename(blob.name))
        blob.download_to_filename(local_path)

print("✅ Synced known_faces from Firebase.")
