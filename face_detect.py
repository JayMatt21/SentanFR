import cv2
import face_recognition
import threading

rtsp_url = "rtsp://Jarvis2:Passw0rd2@192.168.1.7:554/stream1"

frame = None
running = True

def grab_frame():
    global frame, running
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

    if not cap.isOpened():
        print("❌ Cannot open RTSP stream")
        running = False
        return

    while running:
        ret, grabbed_frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            running = False
            break
        frame = grabbed_frame

    cap.release()

# Start thread
thread = threading.Thread(target=grab_frame)
thread.start()

while running:
    if frame is None:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    for top, right, bottom, left in face_locations:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Threaded Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
thread.join()
