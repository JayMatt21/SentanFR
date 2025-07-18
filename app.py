from flask import Flask, render_template, request, redirect, url_for, session, send_file
import firebase_admin
from firebase_admin import credentials, firestore, storage
from deepface import DeepFace
import cv2
import os
import time
from datetime import datetime
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'nooneelse'

# === Firebase ===
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'senfrostfr.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

# === Helpers ===
def cosine_dist(a, b):
    return 1 - dot(a, b) / (norm(a) * norm(b))

def log_error(msg):
    db.collection('logs').add({'time': datetime.now(), 'msg': msg})

# === Routes ===

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        role = request.form['role']
        doc = db.collection('users').document(user).get()
        if doc.exists:
            data = doc.to_dict()
            if data['password'] == pwd and data['role'] == role:
                session['username'] = user
                session['role'] = role
                if role == 'superadmin':
                    return redirect(url_for('superadmin_dashboard'))
                elif role == 'admin':
                    return redirect(url_for('admin_dashboard'))
                elif role == 'user':
                    return redirect(url_for('user_dashboard'))
            else:
                error = "Wrong password or role"
        else:
            error = "User not found"
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# === Dashboards ===

@app.route('/superadmin_dashboard')
def superadmin_dashboard():
    if session.get('role') != 'superadmin':
        return redirect(url_for('login'))
    logs = db.collection('logs').stream()
    return render_template('superadmin_dashboard.html', logs=logs)

@app.route('/admin_dashboard')
def admin_dashboard():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    return render_template('admin_dashboard.html')

@app.route('/user_dashboard')
def user_dashboard():
    if session.get('role') != 'user':
        return redirect(url_for('login'))
    attendance = db.collection('attendance').where('username', '==', session['username']).stream()
    return render_template('user_dashboard.html', attendance=attendance)

# === Registration ===

@app.route('/register_admin', methods=['GET','POST'])
def register_admin():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        db.collection('users').document(user).set({
            'username': user,
            'password': pwd,
            'role': 'admin'
        })
        return redirect(url_for('login'))
    return render_template('register_admin.html')

@app.route('/register_user', methods=['GET','POST'])
def register_user():
    if request.method == 'POST':
        user = request.form['username']
        pwd = request.form['password']
        embeddings = []
        cap = cv2.VideoCapture(0)
        for i in range(3):
            ret, frame = cap.read()
            fname = f"{user}_{i}.jpg"
            cv2.imwrite(fname, frame)
            emb = DeepFace.represent(img_path=fname, model_name='Facenet512')[0]['embedding']
            embeddings.append(emb)
            blob = bucket.blob(f'faces/{fname}')
            blob.upload_from_filename(fname)
        cap.release()
        cv2.destroyAllWindows()
        db.collection('users').document(user).set({
            'username': user,
            'password': pwd,
            'role': 'user',
            'embeddings': embeddings
        })
        return redirect(url_for('login'))
    return render_template('register_user.html')

# === Attendance ===

@app.route('/start_recognition')
def start_recognition():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    
    rtsp_url = "rtsp://Jarvis2:Passw0rd2@192.168.1.7:554/stream1"
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            faces = DeepFace.extract_faces(frame, enforce_detection=False)
            for face in faces:
                emb = DeepFace.represent(face['face'], model_name='Facenet512')[0]['embedding']
                match = None
                users = db.collection('users').where('role', '==', 'user').stream()
                for u in users:
                    data = u.to_dict()
                    for saved_emb in data['embeddings']:
                        dist = cosine_dist(emb, saved_emb)
                        if dist < 0.4:
                            match = data['username']
                            break
                if match:
                    db.collection('attendance').add({
                        'username': match,
                        'timestamp': datetime.now()
                    })
                    print(f"✅ {match} time-in recorded")
        except Exception as e:
            log_error(str(e))
    cap.release()
    cv2.destroyAllWindows()
    return "Recognition Stopped"

# === Payroll ===

@app.route('/generate_payslip')
def generate_payslip():
    if session.get('role') != 'admin':
        return redirect(url_for('login'))
    data = []
    users = db.collection('users').where('role', '==', 'user').stream()
    for u in users:
        uname = u.id
        att = db.collection('attendance').where('username', '==', uname).stream()
        days = len(list(att))
        gross = days * 500
        deductions = gross * 0.1
        net = gross - deductions
        data.append({
            'username': uname,
            'days': days,
            'gross': gross,
            'deductions': deductions,
            'net': net
        })
    df = pd.DataFrame(data)
    df.to_csv('payroll.csv', index=False)
    return send_file('payroll.csv', as_attachment=True)

