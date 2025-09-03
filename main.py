import os
import time
import uuid
import json
import shutil
import tempfile
import qrcode
import smtplib
import numpy as np
import pandas as pd
import cv2
import face_recognition
import pytesseract
from PIL import Image
from email.message import EmailMessage
import streamlit as st
import pyttsx3
import base64
import csv
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎓 Graduation Authentication System", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES (UI ONLY — NO LOGIC CHANGES)
# ──────────────────────────────────────────────────────────────────────────────
def add_custom_css():
    st.markdown("""
    <style>
        :root {
            --card-bg: #ffffff;
            --card-border: #e9eef3;
            --text-main: #2c3e50;
            --text-muted: #6b7c93;
            --accent: #1f6feb;
        }

        /* Keep typography compact & professional */
        html, body, [class*="css"] {
            font-size: 14px !important;
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif !important;
        }

        /* Make the main content area look like a clean sheet on top of any bg */
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1200px;
            background: rgba(255,255,255,0.86);
            border-radius: 14px;
            border: 1px solid rgba(233,238,243,0.7);
            box-shadow: 0 6px 22px rgba(0,0,0,0.04);
        }

        /* Title + subtitle */
        .page-title {
            color: var(--text-main);
            font-weight: 700;
            font-size: 28px !important;
            margin: 0 0 4px 0;
            text-align: center;
        }
        .page-subtitle {
            color: var(--text-muted);
            text-align: center;
            margin-bottom: 6px;
        }

        /* Section heading (step titles) */
        .section-header {
            color: var(--text-main);
            font-weight: 600;
            font-size: 22px !important;
            margin: 0 0 2px 0;
        }

        /* Subtle info line under section header */
        .section-hint {
            color: var(--text-muted);
            margin-bottom: 6px;
            font-size: 18px !important;
        }
        
        .section-info {
            color: var(--text-main);
            font-weight: 600;
            font-size: 18px !important;
            margin-bottom: 10px;
        }

        /* Generic card wrapper */
        .card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 14px;
        }

        /* Tighten Streamlit alerts spacing */
        .stAlert {
            padding: 10px 12px !important;
            border-radius: 10px !important;
            font-size: 14px !important;
        }

        /* Buttons — compact & rounded */
        button[kind="secondary"], button[kind="primary"], .stButton > button {
            font-size: 14px !important;
            padding: 8px 14px !important;
            border-radius: 10px !important;
        }

        /* Sidebar tidy */
        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #eef2f7;
        }
        section[data-testid="stSidebar"] .block-container {
            box-shadow: none;
            border: none;
            background: transparent;
        }

        /* Metric-like chips */
        .pill {
            display: inline-block;
            background: #f3f6fb;
            border: 1px solid #e7edf6;
            color: #334155;
            border-radius: 999px;
            padding: 6px 12px;
            margin: 2px 6px 6px 0;
            font-size: 13px;
        }

        /* Key-value table */
        .kv-table {
            width: 100%;
            border-collapse: collapse;
        }
        .kv-table td {
            border-bottom: 1px dashed #ecf0f5;
            padding: 8px 6px;
            vertical-align: top;
        }
        .kv-table td:first-child {
            color: #6b7c93;
            width: 34%;
        }

        /* Make images/video frames nicely rounded */
        img, video, canvas {
            border-radius: 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL BACKGROUND (kept, content card is opaque for readability)
# ──────────────────────────────────────────────────────────────────────────────
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

if os.path.exists("background.jpg"):
    set_bg("background.jpg")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">🎓 Graduation Authentication System</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">OCR • Face Recognition • QR Code • Email • Attendance • Stage Verification</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# YOUR EXISTING LOGIC — UNCHANGED
# ──────────────────────────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def record_attendance(sid, name, course):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "sid", "name", "course"])
        writer.writerow([timestamp, sid, name, course])

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_FILE = os.path.join(APP_DIR, "attendance.csv")
STUDENTS_XLSX = os.path.join(APP_DIR, "students.xlsx")
DATA_FILE = "face_db.json"
QR_DIR = "qr_codes"
ENCODINGS_DIR = "encodings"
TARGET_W, TARGET_H = 640, 480
BACKEND = cv2.CAP_DSHOW if os.name=="nt" else 0
SCAN_TIMEOUT_SEC = 600
SLEEP_BETWEEN_FRAMES = 0.12
STABLE_SEC = 3
CAPTURE_SEC = 3
ocr_name = None
ocr_sid = None
ocr_course = ""
ocr_email = ""
os.makedirs(ENCODINGS_DIR, exist_ok=True)

# initialize session state
if "FACE_DB" not in st.session_state:
    st.session_state.FACE_DB = {}

# add face_db.json to session_state
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            st.session_state.FACE_DB = json.load(f)
        except json.JSONDecodeError:
            st.session_state.FACE_DB = {}
else:
    st.session_state.FACE_DB = {}

def compare_single_face(stored_encoding_path, enc, tol=0.45):
    """
    Compare whether a single face encoding matches.
    """
    if not os.path.exists(stored_encoding_path):
        return False
    stored_enc = np.load(stored_encoding_path)
    matches = face_recognition.compare_faces([stored_enc], enc, tolerance=tol)
    return matches[0]

def img_to_face_encodings(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    encs = face_recognition.face_encodings(rgb, locs)
    
    # If multiple faces found, find the biggest one
    if len(locs) > 1:
        # Calculate areas of all faces
        face_areas = []
        for (top, right, bottom, left) in locs:
            area = (bottom - top) * (right - left)
            face_areas.append(area)
        
        # Find index of the biggest face
        biggest_face_idx = np.argmax(face_areas)
        
        # Return only the biggest face
        return [locs[biggest_face_idx]], [encs[biggest_face_idx]]
    
    return locs, encs

def ensure_single_face(locs):
    # This function now just checks if at least one face is detected
    # since we're already handling multiple faces in img_to_face_encodings
    return len(locs) > 0
    
def send_email_with_attachment(to_email, subject, body, attachment_path):
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "laupx-wm22@student.tarc.edu.my")
    SMTP_PASS = os.getenv("SMTP_PASS", "dnzw ulma eykm bsqv") 
    
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype="image", subtype="png", filename=os.path.basename(attachment_path))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

def generate_and_send_qr(sid: str, name: str, email: str, face_db, qr_dir="qr_codes"):
    import qrcode, os, time
    os.makedirs(qr_dir, exist_ok=True)

    info = face_db.get(sid, {})
    encp = info.get("encoding_path")
    if not encp or not os.path.exists(encp):
        st.error("The student's face code cannot be found")
        return

    uid = info.get("uuid") or str(uuid.uuid4())
    face_db[sid] = {**info, "uuid": uid}

    # generate QR
    qp = os.path.join(qr_dir, f"{sid}.png")
    if not os.path.exists(qp):
        qrcode.make(uid).save(qp)
    face_db[sid]["qr_path"] = qp

    # send email
    try:
        send_email_with_attachment(
            email,
            "Your Graduation QR Code",
            f"Dear {name},\n\nPlease use this QR during entrance. Keep it private.\n\nRegards,\nCeremony Team",
            qp
        )
        face_db[sid]["emailed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        st.error(f"邮件发送失败：{e}")

def _norm(s: str) -> str:
    s = "".join(ch for ch in (s or "") if ch.isalnum()).upper()
    return (s.replace("O","0").replace("I","1").replace("L","1")
            .replace("S","5").replace("Z","2").replace("B","8"))

def _lev(a,b):
    if a==b: return 0
    if not a: return len(b)
    if not b: return len(a)
    dp=[[i+j if i*j==0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]!=b[j-1]))
    return dp[-1][-1]

def correct_against_roster(raw_id, df, max_dist=2):
    if df is None or "ID" not in df.columns:
        return _norm(raw_id), None
    raw = _norm(raw_id)
    best_sid, best_d = None, 999
    for sid in df["ID"].astype(str):
        sidn = _norm(sid)
        d = _lev(raw, sidn)
        if d < best_d:
            best_sid, best_d = sidn, d
            if d==0: break
    if best_d<=max_dist: return best_sid, best_d
    return raw, best_d

def _order_points(pts):
    rect = np.zeros((4,2),dtype="float32")
    s=pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    diff=np.diff(pts,axis=1); rect[1]=pts[np.argmin(diff)]; rect[3]=pts[np.argmax(diff)]
    return rect

def perspective_warp_card(frame_bgr, W=900, H=600):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    card_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(card_contour)
    box = cv2.boxPoints(rect).astype("float32")
    dst_pts = np.array([[0,0],[W,0],[W,H],[0,H]], dtype="float32")
    M = cv2.getPerspectiveTransform(_order_points(box), dst_pts)
    warped = cv2.warpPerspective(frame_bgr,M,(W,H))
    return warped

def ocr_name_and_id_from_warped(warped_bgr):
    H,W = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    name_pct=(0.36,0.33,0.50,0.93); id_pct=(0.50,0.30,0.64,0.88)
    def roi(img,pct):
        y1=int(pct[0]*H); x1=int(pct[1]*W); y2=int(pct[2]*H); x2=int(pct[3]*W)
        return img[y1:y2,x1:x2]
    name_text = pytesseract.image_to_string(roi(gray,name_pct), config="--oem 3 --psm 7").strip()
    id_text   = _norm(pytesseract.image_to_string(roi(gray,id_pct), config="--oem 3 --psm 7").strip())
    return name_text, id_text

@st.cache_data
def load_roster(path):
    if not os.path.exists(path): return None
    df = pd.read_excel(path)
    df = df.rename(columns={c.lower():c for c in df.columns})
    required_cols = ["id","name","course","email"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = "" 
    df = df.rename(columns={
        c: "ID" if c.lower()=="id" else "Name" if c.lower()=="name" else 
           "Course" if c.lower()=="course" else "Email" if c.lower()=="email" else c
        for c in df.columns
    })
    return df[["ID","Name","Course","Email"]]

roster = load_roster(STUDENTS_XLSX)
if roster is None:
    st.error("students.xlsx must include id and name row")
    st.stop()

def enc_path_for(sid):
    return os.path.join(ENCODINGS_DIR,f"{sid}.npy")

def load_db():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(db):
    with open(DATA_FILE,"w",encoding="utf-8") as f:
        json.dump(db,f,indent=2)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR (UNCHANGED FLOW: THREE BUTTONS)
# ──────────────────────────────────────────────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "Registration"

st.sidebar.markdown("### Select mode")
if st.sidebar.button("Registration"):
    st.session_state.mode = "Registration"
if st.sidebar.button("Take Attendance"):
    st.session_state.mode = "Entrance"
if st.sidebar.button("Stage verification"):
    st.session_state.mode = "Stage verification"

mode = st.session_state.mode

# ---------------------------------------------------
# STEP 1 — REGISTRATION
# ---------------------------------------------------
if mode == "Registration":
    st.markdown('<div class="section-header">Step 1: Student Registration</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Scan your Student ID and capture your face to get your Attendance QR code.</div>', unsafe_allow_html=True)

    # Two-column layout: left = camera feed, right = status + student info
    col_left, col_right = st.columns([7, 5], gap="small")

    with col_left:
        start_scan = st.button("📸 Scan Student ID & Face Capture", use_container_width=True)
        frame_slot = st.empty()  # dynamic frame slot (only one)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-info">Status</div>', unsafe_allow_html=True)
        result_slot1 = st.empty()
        result_slot2 = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-info">Student Information</div>', unsafe_allow_html=True)
        info_slot = st.empty()  # fill after OCR success
        json_expander = st.expander("Show raw record (JSON)", expanded=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # Keep your original flow; just redirected the visual outputs to the new slots
    if "ocr_sid" not in st.session_state:
        st.session_state.ocr_sid = None

    if 'db_record_for_json' not in st.session_state:
        st.session_state.db_record_for_json = None

    if start_scan:
        # ---------------- OCR stage ----------------
        cap = cv2.VideoCapture(0, BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        result_slot1.info("Present your student ID/card to the camera for OCR…")

        ocr_name = ocr_course = ocr_email = None
        ocr_sid = None
        t0 = time.time()
        while time.time() - t0 < SCAN_TIMEOUT_SEC:
            ok, frame = cap.read()
            if not ok:
                continue

            # show frame
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            warped = perspective_warp_card(frame)
            if warped is None:
                time.sleep(SLEEP_BETWEEN_FRAMES)
                continue

            name_text, id_text = ocr_name_and_id_from_warped(warped)
            if not id_text:
                time.sleep(SLEEP_BETWEEN_FRAMES)
                continue

            ocr_sid, _ = correct_against_roster(id_text, roster)
            hit = roster[roster["ID"].astype(str).str.upper() == ocr_sid]
            if not hit.empty:
                row = hit.iloc[0]
                ocr_name = row["Name"]
                ocr_course = row.get("Course", "")
                ocr_email = row.get("Email", "")
                result_slot1.success(f"OCR matched: {ocr_name} ({ocr_sid})")
                with info_slot:
                    st.markdown(
                        f"""
                        <table class="kv-table">
                            <tr><td>Student Name</td><td><strong>{ocr_name}</strong></td></tr>
                            <tr><td>Student ID</td><td>{ocr_sid}</td></tr>
                            <tr><td>Course</td><td>{ocr_course}</td></tr>
                            <tr><td>Email</td><td>{ocr_email}</td></tr>
                        </table>
                        """,
                        unsafe_allow_html=True
                    )
                break

        cap.release()

        if not ocr_sid:
            result_slot1.error("No valid ID was identified. The process terminated.")
            st.stop()

        # save sid_key to session_state
        st.session_state.ocr_sid = ocr_sid.strip().upper()
        sid_key = st.session_state.ocr_sid

        # ---------------- Face capture ----------------
        cap = cv2.VideoCapture(1, BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        result_slot1.info("Move closer to the camera for face scanning...")

        db = load_db()
        captured = False
        stable_face_start = countdown_start = None
        t0 = time.time()

        while time.time() - t0 < SCAN_TIMEOUT_SEC:
            ok, frame = cap.read()
            if not ok:
                continue

            locs, encs = img_to_face_encodings(frame)
            
            # Draw bounding box only for the biggest face (if any)
            if locs:
                (top, right, bottom, left) = locs[0]
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, "Face", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if not encs:
                stable_face_start = countdown_start = None
                result_slot1.info("No face detected")
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                time.sleep(SLEEP_BETWEEN_FRAMES)
                continue

            # We don't need the multiple faces check anymore since img_to_face_encodings
            # already handles it and returns only the biggest face

            if stable_face_start is None:
                stable_face_start = time.time()
            if time.time() - stable_face_start < STABLE_SEC:
                result_slot1.info(f"Face stabilization: {int(STABLE_SEC - (time.time() - stable_face_start))}s")
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                time.sleep(SLEEP_BETWEEN_FRAMES)
                continue

            if countdown_start is None:
                countdown_start = time.time()
            remaining = max(0, CAPTURE_SEC - int(time.time() - countdown_start))
            cv2.putText(frame, f"Capturing in: {remaining}s", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            if remaining > 0:
                time.sleep(SLEEP_BETWEEN_FRAMES)
                continue

            # save face code
            np.save(enc_path_for(sid_key), encs[0])
            db[sid_key] = {
                "name": ocr_name.strip(),
                "course": ocr_course,
                "email": ocr_email,
                "encoding_path": enc_path_for(sid_key),
                "uuid": str(uuid.uuid4())
            }
            save_db(db)
            
            st.session_state.db_record_for_json = db[sid_key]  # keep for expander
            captured = True
            cap.release()
            break

        if not captured:
            result_slot1.error("Timeout. No valid face was captured. Please try again.")
            st.stop()

        # generate qr
        os.makedirs(QR_DIR, exist_ok=True)
        qr_file = os.path.join(QR_DIR, f"{sid_key}.png")
        uid = db[sid_key]["uuid"]
        qrcode.make(uid).save(qr_file)
        db[sid_key]["qr_path"] = qr_file
        save_db(db)

        # send
        if captured and ocr_email:
            generate_and_send_qr(sid_key, ocr_name, ocr_email, db, QR_DIR)
            result_slot2.success(f"✅ QR sent to {ocr_email}")

# ---------------------------------------------------
# STEP 2 — ENTRANCE (Rewritten)
# ---------------------------------------------------
elif mode == "Entrance":
    st.markdown('<div class="section-header">Step 2: Take Your Attendance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Present your registered face and unique QR code together for verification.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 5], gap="small")
    with col_left:
        start_btn = st.button("📸 Scan Face & QR Code", use_container_width=True)
        frame_slot = st.empty()

    with col_right:
        st.markdown('<div class="section-info">Status</div>', unsafe_allow_html=True)
        result_slot = st.empty()

    # Load DB from disk and sync session
    db = load_db()
    st.session_state.FACE_DB = db  # ensure session DB is up to date

    if start_btn:
        cap = cv2.VideoCapture(0, BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)

        result_slot.info("Move close to the camera for face scanning and scan QR Code…")

        qr_detector = cv2.QRCodeDetector()
        scanned_qr = False
        matched_face = False
        t0 = time.time()

        sid_from_qr = None
        student_info = None  # will store info of QR owner

        while time.time() - t0 < SCAN_TIMEOUT_SEC:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ----- QR Detection -----
            retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)
            if retval and decoded_info and not scanned_qr:
                for data in decoded_info:
                    # Find student by QR uuid
                    for sid, info in db.items():
                        if info.get("uuid") == data:
                            sid_from_qr = sid
                            student_info = info
                            scanned_qr = True
                            result_slot.success(f"✅ QR verification passed: {info['name']} ({sid_from_qr})")
                            break
                    if scanned_qr:
                        break

                # Draw QR bounding boxes
                if points is not None:
                    for qr_pts in points:
                        pts = qr_pts.astype(int)
                        for j in range(4):
                            cv2.line(frame, tuple(pts[j]), tuple(pts[(j+1)%4]), (255,0,0), 3)

            # ----- Face Detection -----
            locs, encs = img_to_face_encodings(frame)
            if locs:
                for (top, right, bottom, left) in locs:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(frame, "Face", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # ----- Face Matching -----
            if encs and scanned_qr and sid_from_qr:
                enc_path = db[sid_from_qr].get("encoding_path")
                if enc_path and compare_single_face(enc_path, encs[0], tol=0.45):
                    result_slot.success(f"✅ Attendance taken for: {student_info['name']} ({sid_from_qr})")
                    matched_face = True
                    # Record attendance
                    record_attendance(sid_from_qr, student_info['name'], student_info['course'])
                else:
                    result_slot.warning("Face does not match QR owner")

            elif encs and not scanned_qr:
                result_slot.info("Face detected, but QR code not yet scanned")

            elif not encs:
                result_slot.info("No face detected")

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(SLEEP_BETWEEN_FRAMES)

            if scanned_qr and matched_face:
                break

        cap.release()
        if not (scanned_qr and matched_face):
            result_slot.warning("Timeout. No valid face or QR Code detected.")

# ---------------------------------------------------
# STEP 3 — STAGE VERIFICATION
# ---------------------------------------------------
elif mode == "Stage verification":
    st.markdown('<div class="section-header">Step 3: Stage Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">Final verification as the student approaches the stage.</div>', unsafe_allow_html=True)

    st.session_state.FACE_DB = load_db()
    if not st.session_state.FACE_DB:
        st.warning("The local FACE_DB is empty. Please complete the registration of at least one student first.")
        st.stop()

    col_left, col_right = st.columns([7, 5], gap="small")
    with col_left:
        start_stage = st.button("📸 Start Stage Verification", use_container_width=True)
        frame_slot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-info">Status</div>', unsafe_allow_html=True)
        status_slot = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    TARGET_W, TARGET_H = 640, 480
    if start_stage:
        cap = cv2.VideoCapture(0, BACKEND)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        status_slot.info("Take a photo close to the camera for face recognition")

        MAX_FRAMES = 10
        MATCH_THRESHOLD = 0.6
        tol = 0.45

        matched_counts = {}
        stable_face_start = None
        t0 = time.time()
        verified = False

        while time.time() - t0 < SCAN_TIMEOUT_SEC and not verified:
            ok, frame = cap.read()
            if not ok:
                continue

            locs, encs = img_to_face_encodings(frame)
            if not encs:
                stable_face_start = None
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                continue

            (top, right, bottom, left) = locs[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame, "Face", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if stable_face_start is None:
                stable_face_start = time.time()
            
            elapsed = time.time() - stable_face_start
            if elapsed < 2:
                cv2.putText(frame, f"Hold still: {int(2 - elapsed)}s", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                continue

            enc = encs[0]
            for sid, info in st.session_state.FACE_DB.items():
                enc_path = info.get("encoding_path")
                if enc_path and os.path.exists(enc_path):
                    if compare_single_face(enc_path, enc, tol=tol):
                        matched_counts[sid] = matched_counts.get(sid, 0) + 1

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(SLEEP_BETWEEN_FRAMES)

            for sid, count in matched_counts.items():
                if count >= MAX_FRAMES * MATCH_THRESHOLD:
                    info = st.session_state.FACE_DB[sid]
                    status_slot.success(f"✅ Matched: {info.get('name')} – {info.get('course')} ({sid})")
                    verified = True

                    try:
                        engine = pyttsx3.init()
                        engine.setProperty("rate", 160)
                        engine.setProperty("volume", 1.0)
                        announcement = f"Ladies and gentlemen, please welcome {info.get('name')}, graduating from {info.get('course')}. Congratulations!"
                        engine.say(announcement)
                        engine.runAndWait()
                        try:
                            engine.stop()
                        except:
                            pass
                    except Exception as e:
                        status_slot.warning(f"TTS failed: {e}")
                    break

        cap.release()
        if not verified:
            status_slot.error("No faces were matched. Please try again.")