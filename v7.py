import os
import sys
import csv
import time
import sqlite3
import threading
from datetime import datetime

try:
    from tkinter import (
        Tk, Frame, Label, Button, Entry, StringVar, BooleanVar, DoubleVar,
        Toplevel, messagebox, filedialog, ttk
    )
except ImportError as e:
    sys.stderr.write(
        "\n[Error] Python 'tkinter' is missing.\n"
        "Even inside an Arch Linux venv, 'tk' must be installed on the host system:\n"
        "  sudo pacman -S tk\n\n"
    )
    raise SystemExit(1) from e

try:
    import numpy as np
    import cv2
    from PIL import Image, ImageTk
except ImportError as e:
    sys.stderr.write(
        f"\n[Error] Missing required venv package: {e}\n"
        "Activate your virtual environment and run:\n"
        "  pip install numpy opencv-python pillow\n\n"
    )
    raise SystemExit(1) from e

# Optional hardware & recognition libraries with graceful fallbacks
try:
    try:
        import pkg_resources
    except ImportError:
        # Python 3.12+ / 3.14 removed pkg_resources; shim resource_filename safely
        import types
        import importlib.resources
        def resource_filename(package_or_requirement, resource_name):
            return str(importlib.resources.files(package_or_requirement).joinpath(resource_name))
        dummy_pkg = types.ModuleType("pkg_resources")
        dummy_pkg.resource_filename = resource_filename
        sys.modules["pkg_resources"] = dummy_pkg

    import face_recognition
    HAS_FACE_REC = True
except (ImportError, SystemExit, Exception):
    HAS_FACE_REC = False

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False

class AttendanceDatabase:
    """
    Handles SQLite storage for registered people (128-D face encodings or histogram signatures)
    and daily attendance records.
    """
    def __init__(self, db_path="pi_attendance.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    encoding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    date_str TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    UNIQUE(user_id, date_str)
                )
            """)
            conn.commit()

    def register_user(self, user_id: str, name: str, encoding: np.ndarray) -> bool:
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (user_id, name, encoding, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    user_id.strip(),
                    name.strip(),
                    encoding.astype(np.float32).tobytes(),
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"[DB Error] User ID '{user_id}' already exists.")
            return False
        except Exception as e:
            print(f"[DB Error] Failed to register user: {e}")
            return False

    def load_all_users(self):
        users = []
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, name, encoding FROM users")
            for row in cursor.fetchall():
                uid, name, enc_blob = row
                enc = np.frombuffer(enc_blob, dtype=np.float32)
                users.append({
                    "user_id": uid,
                    "name": name,
                    "encoding": enc
                })
        return users

    def delete_user(self, user_id: str) -> bool:
        """
        Removes a registered user from the database by their user_id.
        """
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id.strip(),))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"[DB Error] Failed to delete user: {e}")
            return False

    def mark_attendance(self, user_id: str, name: str, confidence: float) -> bool:
        """
        Marks attendance once per calendar day per user. Returns True if newly marked.
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        try:
            with self.lock, sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO attendance (user_id, name, timestamp, date_str, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, name, time_str, date_str, round(float(confidence), 2)))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"[DB Error] Attendance log failed: {e}")
            return False

    def get_today_attendance(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, name, timestamp, confidence
                FROM attendance
                WHERE date_str = ?
                ORDER BY id DESC
            """, (date_str,))
            return cursor.fetchall()

    def clear_today_attendance(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance WHERE date_str = ?", (date_str,))
            conn.commit()


class LowLightAndCrowdEngine:
    """
    Provides real-time low-light image enhancement (CLAHE in LAB color space + Gamma LUT)
    and multi-face detection optimized strictly for face_recognition (dlib HOG).
    """
    def __init__(self, max_batch_size=100):
        self.max_batch_size = max_batch_size
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._current_gamma = 1.0
        self._gamma_lut = np.arange(256, dtype=np.uint8)

    def _update_gamma_lut(self, gamma: float):
        if abs(gamma - self._current_gamma) > 0.02:
            self._current_gamma = max(0.1, gamma)
            inv_gamma = 1.0 / self._current_gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            self._gamma_lut = table

    def enhance_low_light(self, bgr_frame: np.ndarray, apply_clahe=True, gamma=1.5, denoise=False) -> np.ndarray:
        """
        Enhances dark or IR-lit frames on Raspberry Pi without blowing out highlights.
        """
        enhanced = bgr_frame
        if apply_clahe:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_enhanced = self.clahe.apply(l_channel)
            merged_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        if abs(gamma - 1.0) > 0.05:
            self._update_gamma_lut(gamma)
            enhanced = cv2.LUT(enhanced, self._gamma_lut)

        if denoise:
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def compute_embedding(self, bgr_frame: np.ndarray, box_xywh) -> np.ndarray:
        """
        Computes a 128-D face encoding using `face_recognition` strictly.
        """
        if not HAS_FACE_REC:
            return np.zeros(128, dtype=np.float32)

        x, y, w, h = box_xywh
        h_img, w_img = bgr_frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        face_crop = bgr_frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(
            rgb_crop, known_face_locations=[(0, rgb_crop.shape[1], rgb_crop.shape[0], 0)]
        )
        if encodings:
            return np.array(encodings[0], dtype=np.float32)
            
        return np.zeros(128, dtype=np.float32)

    def detect_faces_batch(self, bgr_frame: np.ndarray, min_face_px=24):
        """
        Detects faces using face_recognition (HOG model) on a 4x downscaled frame for maximum CPU performance.
        """
        boxes = []
        if not HAS_FACE_REC:
            return boxes

        # Downscale by 4x (0.25) to drastically improve dlib HOG performance on CPU
        scale_factor = 0.25
        small_frame = cv2.resize(bgr_frame, (0, 0), fx=1,fy=1)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=1, model="hog")
        
        # Scale bounding boxes back up to original frame resolution
        inv_scale = int(1 / scale_factor)
        for (top, right, bottom, left) in locs[:self.max_batch_size]:
            top *= inv_scale
            right *= inv_scale
            bottom *= inv_scale
            left *= inv_scale
            boxes.append((left, top, right - left, bottom - top))

        return boxes

class VideoStream:
    """
    Dedicated thread for reading camera frames to eliminate OpenCV buffer latency.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            self.stream = cv2.VideoCapture(src)
            
        # Hardware optimizations for FPS
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.stopped = True
        self.stream.release()


class CrowdSimulator:
    """
    Generates a realistic low-light auditorium/classroom scene with up to 100 synthetic faces
    (arranged in a 10x10 grid) so users can test 100-person batch detection & attendance logging
    even when sitting alone at their desk.
    """
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.tick = 0

    def generate_frame_and_boxes(self, registered_users, num_people=20, dark_noise=True):
        self.tick += 1
        frame = np.full((self.height, self.width, 3), 18, dtype=np.uint8)

        # Draw subtle classroom grid background
        cols = 10
        rows = int(np.ceil(num_people / cols))
        pad_x, pad_y = 60, 50
        cell_w = (self.width - 2 * pad_x) // cols
        cell_h = (self.height - 2 * pad_y) // max(1, rows)

        simulated_detections = []

        for idx in range(min(num_people, 100)):
            r = idx // cols
            c = idx % cols

            # Subtle natural movement oscillation
            jitter_x = int(3 * np.sin((self.tick + idx * 7) * 0.15))
            jitter_y = int(2 * np.cos((self.tick + idx * 5) * 0.15))

            cx = pad_x + c * cell_w + cell_w // 2 + jitter_x
            cy = pad_y + r * cell_h + cell_h // 2 + jitter_y
            fw = min(cell_w - 16, 48)
            fh = min(cell_h - 14, 52)
            x1 = cx - fw // 2
            y1 = cy - fh // 2

            # Draw synthetic low-light face silhouette
            skin_tone = (42 + (idx % 15), 52 + (idx % 20), 65 + (idx % 25))
            cv2.ellipse(frame, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, skin_tone, -1)
            # Eyes
            eye_y = cy - fh // 6
            cv2.circle(frame, (cx - fw // 5, eye_y), 3, (15, 15, 15), -1)
            cv2.circle(frame, (cx + fw // 5, eye_y), 3, (15, 15, 15), -1)
            # Mouth
            cv2.ellipse(frame, (cx, cy + fh // 5), (fw // 5, 3), 0, 0, 180, (25, 25, 35), 1)

            # Assign identity from registered users or temporary batch ID
            if idx < len(registered_users):
                u = registered_users[idx]
                uid = u["user_id"]
                name = u["name"]
                conf = 0.94
            else:
                uid = f"BATCH-{idx + 1:03d}"
                name = f"Attendee #{idx + 1:03d}"
                conf = 0.88

            simulated_detections.append({
                "box": (x1, y1, fw, fh),
                "user_id": uid,
                "name": name,
                "confidence": conf
            })

        if dark_noise:
            noise = np.random.randint(0, 12, (self.height, self.width, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            if len(detected) > 0:
                for (x, y, w, h) in detected[:self.max_batch_size]:
                    boxes.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))

        return boxes


class PiAttendanceApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Attendance app")
        self.root.geometry("1280x780")
        self.root.minsize(1024, 680)
        self.root.configure(bg="#0f172a")

        # Core modules
        self.db = AttendanceDatabase()
        self.engine = LowLightAndCrowdEngine(max_batch_size=100)
        self.known_users = self.db.load_all_users()

        # State variables
        self.is_running = True
        self.enable_clahe = BooleanVar(value=True)
        self.enable_denoise = BooleanVar(value=False)
        self.gamma_val = DoubleVar(value=1.6)

        self.fps_text = StringVar(value="FPS: 0.0")
        self.detected_count_text = StringVar(value="Faces: 0")
        self.present_count_text = StringVar(value="Count: 0")
        self.status_banner_text = StringVar(value="CLAHE ON")

        # Video capture & threading state
        self.vs = None
        self.picam2 = None
        self.latest_raw_frame = None
        self.latest_display_frame = None
        self.current_detections = []
        self.frame_lock = threading.Lock()

        self._setup_styles()
        self._build_ui()
        self._init_camera()
        self._refresh_attendance_table()

        # Start background processing thread
        self.worker_thread = threading.Thread(target=self._capture_and_process_loop, daemon=True)
        self.worker_thread.start()

        # Start UI video refresh loop
        self._update_gui_frame()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#1e293b",
            foreground="#f8fafc",
            fieldbackground="#1e293b",
            rowheight=28,
            font=("sans-serif", 10)
        )
        style.configure(
            "Treeview.Heading",
            background="#334155",
            foreground="#38bdf8",
            font=("sans-serif", 10, "bold")
        )
        style.map("Treeview", background=[("selected", "#0284c7")])

    def _build_ui(self):
        # Top Header Bar
        header = Frame(self.root, bg="#1e293b", height=60, padx=16, pady=10)
        header.pack(fill="x", side="top")

        title_lbl = Label(
            header,
            text="🍓 PiVision Low-Light Crowd Attendance (Up to 100 Faces/Batch)",
            bg="#1e293b",
            fg="#f8fafc",
            font=("sans-serif", 14, "bold")
        )
        title_lbl.pack(side="left")

        # Live KPI badges in header
        kpi_frame = Frame(header, bg="#1e293b")
        kpi_frame.pack(side="right")

        for var, bg_col, fg_col in [
            (self.fps_text, "#0f172a", "#94a3b8"),
            (self.detected_count_text, "#0369a1", "#e0f2fe"),
            (self.present_count_text, "#047857", "#d1fae5")
        ]:
            badge = Label(
                kpi_frame,
                textvariable=var,
                bg=bg_col,
                fg=fg_col,
                font=("sans-serif", 10, "bold"),
                padx=12,
                pady=6
            )
            badge.pack(side="left", padx=6)

        # Main Body Container
        main_body = Frame(self.root, bg="#0f172a", padx=12, pady=12)
        main_body.pack(fill="both", expand=True)

        # Left Column: Video Feed & Low-Light Controls
        left_col = Frame(main_body, bg="#0f172a")
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))

        video_card = Frame(left_col, bg="#1e293b", bd=1, relief="solid", width=800, height=480)
        video_card.pack_propagate(False) # Stop dynamic zooming feedback loop
        video_card.pack(fill="both", expand=True)

        self.video_label = Label(
            video_card,
            bg="#090d16",
            fg="#64748b",
            text="Initializing Camera Stream (V4L2 / PiCam)...",
            font=("sans-serif", 13)
        )
        self.video_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Low-Light Control Deck under Video
        controls_card = Frame(left_col, bg="#1e293b", pady=10, padx=14)
        controls_card.pack(fill="x", pady=(10, 0))

        # Row 1: Low-Light Enhancement Controls
        row1 = Frame(controls_card, bg="#1e293b")
        row1.pack(fill="x")

        ttk.Checkbutton(
            row1,
            text="CLAHE toggle",
            variable=self.enable_clahe
        ).pack(side="left", padx=(0, 14))

        ttk.Checkbutton(
            row1,
            text="Low-Light Denoise",
            variable=self.enable_denoise
        ).pack(side="left", padx=(0, 14))

        Label(row1, text="Gamma Boost:", bg="#1e293b", fg="#cbd5e1", font=("sans-serif", 9)).pack(side="left")
        gamma_slider = ttk.Scale(row1, from_=0.8, to=2.8, variable=self.gamma_val, orient="horizontal", length=120)
        gamma_slider.pack(side="left", padx=6)

        # Right Column: User Registration & Live Attendance Roster
        right_col = Frame(main_body, bg="#1e293b", width=410, padx=14, pady=12)
        right_col.pack(side="right", fill="y")
        right_col.pack_propagate(False)

        # Registration Card
        Label(
            right_col,
            text="Enrol New Person",
            bg="#1e293b",
            fg="#38bdf8",
            font=("sans-serif", 12, "bold")
        ).pack(anchor="w", pady=(0, 8))

        form_frame = Frame(right_col, bg="#1e293b")
        form_frame.pack(fill="x", pady=(0, 10))

        Label(form_frame, text="ID / Roll No:", bg="#1e293b", fg="#cbd5e1", font=("sans-serif", 9)).grid(row=0, column=0, sticky="w", pady=3)
        self.entry_id = Entry(form_frame, bg="#0f172a", fg="white", insertbackground="white", relief="flat", font=("sans-serif", 10))
        self.entry_id.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3, ipady=3)

        Label(form_frame, text="Full Name:", bg="#1e293b", fg="#cbd5e1", font=("sans-serif", 9)).grid(row=1, column=0, sticky="w", pady=3)
        self.entry_name = Entry(form_frame, bg="#0f172a", fg="white", insertbackground="white", relief="flat", font=("sans-serif", 10))
        self.entry_name.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3, ipady=3)
        form_frame.columnconfigure(1, weight=1)

        btn_register = Button(
            right_col,
            text="➕ Capture Face from Frame & Register",
            command=self._register_face_from_stream,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            relief="flat",
            pady=6,
            font=("sans-serif", 10, "bold")
        )
        btn_register.pack(fill="x", pady=(0, 6))

        btn_delete = Button(
            right_col,
            text="❌ Delete User by ID",
            command=self._delete_user,
            bg="#b91c1c",
            fg="white",
            activebackground="#991b1b",
            relief="flat",
            pady=4,
            font=("sans-serif", 9, "bold")
        )
        btn_delete.pack(fill="x", pady=(0, 14))

        # Divider
        Frame(right_col, bg="#334155", height=1).pack(fill="x", pady=6)

        # Attendance Log Section
        log_header = Frame(right_col, bg="#1e293b")
        log_header.pack(fill="x", pady=(4, 6))

        Label(
            log_header,
            text="Today's Attendance Log",
            bg="#1e293b",
            fg="#38bdf8",
            font=("sans-serif", 12, "bold")
        ).pack(side="left")

        Label(
            log_header,
            text=f"Registered: {len(self.known_users)}",
            bg="#1e293b",
            fg="#94a3b8",
            font=("sans-serif", 9)
        ).pack(side="right")
        self.reg_count_label = log_header.winfo_children()[-1]

        # Treeview Table
        table_container = Frame(right_col, bg="#1e293b")
        table_container.pack(fill="both", expand=True)

        columns = ("id", "name", "time", "conf")
        self.tree = ttk.Treeview(table_container, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("time", text="Time")
        self.tree.heading("conf", text="Conf")

        self.tree.column("id", width=85, anchor="w")
        self.tree.column("name", width=135, anchor="w")
        self.tree.column("time", width=80, anchor="center")
        self.tree.column("conf", width=60, anchor="center")

        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bottom Action Buttons (Export CSV / Reset)
        bottom_btns = Frame(right_col, bg="#1e293b", pady=8)
        bottom_btns.pack(fill="x")

        Button(
            bottom_btns,
            text="📥 Export CSV Report",
            command=self._export_csv,
            bg="#0d9488",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            font=("sans-serif", 9, "bold")
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        Button(
            bottom_btns,
            text="🗑️ Clear Today",
            command=self._clear_today_logs,
            bg="#dc2626",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
            font=("sans-serif", 9, "bold")
        ).pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Status Bar at Bottom of Window
        status_bar = Frame(self.root, bg="#020617", height=28, padx=12)
        status_bar.pack(fill="x", side="bottom")
        Label(
            status_bar,
            textvariable=self.status_banner_text,
            bg="#020617",
            fg="#38bdf8",
            font=("sans-serif", 9)
        ).pack(side="left", pady=4)

    def _init_camera(self):
        """
        Attempts to initialize Picamera2 first, then falls back to the threaded V4L2 VideoStream.
        """
        if HAS_PICAMERA2:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
                self.picam2.configure(config)
                self.picam2.start()
                self.status_banner_text.set("Connected to Raspberry Pi CSI Camera — Low-Light Mode Ready")
                return
            except Exception as e:
                print(f"[Camera] Picamera2 init failed, falling back to V4L2 Webcam: {e}")
                self.picam2 = None

        # Start multi-threaded video stream
        self.vs = VideoStream(src=0).start()
        
        # Brief pause to let camera warm up and capture a frame
        time.sleep(0.5)
        if self.vs.read() is not None:
            self.status_banner_text.set("Connected to Threaded V4L2/USB Camera — Max FPS Mode")
        else:
            self.status_banner_text.set("No physical camera detected — Please plug in a camera.")

    def _capture_and_process_loop(self):
        """
        Runs in a dedicated background thread so the GUI never freezes
        during heavy detection processing.
        """
        frame_counter = 0
        t_start = time.time()
        cached_detections = []

        while self.is_running:
            raw_frame = None
            detections_list = []

            # Read from Picamera2 or threaded VideoStream
            if self.picam2 is not None:
                try:
                    rgb_arr = self.picam2.capture_array()
                    raw_frame = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                except Exception:
                    raw_frame = None
            elif self.vs is not None:
                raw_frame = self.vs.read()

            if raw_frame is None:
                time.sleep(0.01)
                continue

            # Resize raw frame immediately using fastest algorithm to cut CPU load by >50%
            raw_frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_NEAREST)

            enhanced = self.engine.enhance_low_light(
                raw_frame,
                apply_clahe=self.enable_clahe.get(),
                gamma=self.gamma_val.get(),
                denoise=self.enable_denoise.get()
            )

            # --- OPTIMIZATION START ---
            # 1. Detect bounding boxes every 3 frames for a smooth UI
            if frame_counter % 3 == 0:
                boxes = self.engine.detect_faces_batch(enhanced)
                
                detections_list = []
                for box in boxes:
                    emb = self.engine.compute_embedding(enhanced, box)
                    uid, name, conf = self._match_embedding(emb)
                    detections_list.append({
                        "box": box,
                        "user_id": uid,
                        "name": name,
                        "confidence": conf
                    })
                cached_detections = detections_list
            else:
                detections_list = cached_detections
            # --- OPTIMIZATION END ---

            # Draw overlays and log recognized users
            annotated = enhanced.copy()
            newly_marked = 0

            for det in detections_list:
                x, y, w, h = det["box"]
                uid = det["user_id"]
                name = det["name"]
                conf = det["confidence"]

                is_recognized = uid != "UNKNOWN"
                color = (34, 197, 94) if is_recognized else (56, 189, 248)

                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label_str = f"{name[:12]}" if is_recognized else "Unregistered"
                cv2.putText(
                    annotated,
                    label_str,
                    (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA
                )

                if is_recognized:
                    if self.db.mark_attendance(uid, name, conf):
                        newly_marked += 1

            # HUD Banner on top-left of video
            hud_text = (
                f"CLAHE: {'ON' if self.enable_clahe.get() else 'OFF'} | "
                f"Gamma: {self.gamma_val.get():.1f} | "
                f"Faces Detected: {len(detections_list)}"
            )
            cv2.rectangle(annotated, (8, 8), (540, 38), (15, 23, 42), -1)
            cv2.putText(annotated, hud_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

            with self.frame_lock:
                self.latest_raw_frame = enhanced
                self.latest_display_frame = annotated
                self.current_detections = detections_list

            # Update FPS and UI counters
            frame_counter += 1
            elapsed = time.time() - t_start
            if elapsed >= 0.5:
                fps = frame_counter / elapsed
                self.fps_text.set(f"FPS: {fps:.1f}")
                self.detected_count_text.set(f"Faces in Frame: {len(detections_list)}")
                frame_counter = 0
                t_start = time.time()

            if newly_marked > 0:
                self.root.after(0, self._refresh_attendance_table)

            # Removed the time.sleep() delay to allow the loop to run as fast as the CPU permits
            # CPU pacing is naturally handled by the camera's hardware FPS limit

    def _match_embedding(self, query_emb: np.ndarray):
        """
        Matches a face embedding against all registered users in memory using fast vectorized L2 distance.
        """
        if not self.known_users:
            return "UNKNOWN", "Unknown", 0.0

        best_uid = "UNKNOWN"
        best_name = "Unknown"
        best_dist = 999.0

        for user in self.known_users:
            dist = float(np.linalg.norm(query_emb - user["encoding"]))
            if dist < best_dist:
                best_dist = dist
                best_uid = user["user_id"]
                best_name = user["name"]

        threshold = 0.58 if HAS_FACE_REC else 0.42
        if best_dist <= threshold:
            confidence = max(0.50, 1.0 - (best_dist / (threshold * 1.5)))
            return best_uid, best_name, confidence
        return "UNKNOWN", "Unknown", 0.0

    def _update_gui_frame(self):
        if not self.is_running:
            return

        frame_to_show = None
        with self.frame_lock:
            if self.latest_display_frame is not None:
                frame_to_show = self.latest_display_frame.copy()

        if frame_to_show is not None:
            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
            # Default constraints to stop continuous zoom-in if UI hasn't settled
            if lbl_w < 100 or lbl_h < 100:
                lbl_w, lbl_h = 800, 480

            h, w = frame_to_show.shape[:2]
            scale = min(lbl_w / max(1, w), lbl_h / max(1, h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

            # Use INTER_NEAREST for faster rendering
            resized = cv2.resize(frame_to_show, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk, text="")
            self.video_label.imgtk = imgtk

        self.root.after(40, self._update_gui_frame)

    def _register_face_from_stream(self):
        user_id = self.entry_id.get().strip()
        name = self.entry_name.get().strip()

        if not user_id or not name:
            self.status_banner_text.set("⚠️ Please enter both ID/Roll No and Full Name before registering.")
            return

        # 1. Prevent Duplicate ID / Roll No
        if any(u["user_id"] == user_id for u in self.known_users):
            self.status_banner_text.set(f"⚠️ Registration blocked: ID '{user_id}' already exists!")
            messagebox.showerror("Duplicate ID", f"A user with ID '{user_id}' is already registered.\nPlease delete them first if you want to overwrite.")
            return

        with self.frame_lock:
            frame = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None
            detections = list(self.current_detections)

        if frame is None:
            self.status_banner_text.set("⚠️ Camera stream not ready yet.")
            return

        # Select the largest detected face in frame, or center region if none detected
        if detections:
            largest = max(detections, key=lambda d: d["box"][2] * d["box"][3])
            box = largest["box"]
        else:
            h, w = frame.shape[:2]
            box = (w // 4, h // 4, w // 2, h // 2)

        emb = self.engine.compute_embedding(frame, box)
        
        # 2. Prevent Duplicate Biometrics (Same face, different ID)
        matched_uid, matched_name, _ = self._match_embedding(emb)
        if matched_uid != "UNKNOWN":
            self.status_banner_text.set(f"⚠️ Registration blocked: Face already registered to '{matched_name}'.")
            messagebox.showerror("Duplicate Face", f"This face is already registered to {matched_name} (ID: {matched_uid}).")
            return

        if self.db.register_user(user_id, name, emb):
            self.known_users = self.db.load_all_users()
            self.reg_count_label.configure(text=f"Registered: {len(self.known_users)}")
            self.db.mark_attendance(user_id, name, 0.99)
            self._refresh_attendance_table()
            self.entry_id.delete(0, "end")
            self.entry_name.delete(0, "end")
            self.status_banner_text.set(f"✅ Enrolled '{name}' ({user_id}) and marked present!")

    def _delete_user(self):
        user_id = self.entry_id.get().strip()
        if not user_id:
            self.status_banner_text.set("⚠️ Please enter an ID / Roll No to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to permanently delete user ID '{user_id}'?"):
            if self.db.delete_user(user_id):
                self.known_users = self.db.load_all_users()
                self.reg_count_label.configure(text=f"Registered: {len(self.known_users)}")
                self.entry_id.delete(0, "end")
                self.entry_name.delete(0, "end")
                self.status_banner_text.set(f"🗑️ Successfully deleted user ID '{user_id}'.")
            else:
                self.status_banner_text.set(f"⚠️ User ID '{user_id}' not found in database.")

    def _refresh_attendance_table(self):
        rows = self.db.get_today_attendance()
        for item in self.tree.get_children():
            self.tree.delete(item)

        for uid, name, ts, conf in rows:
            self.tree.insert("", "end", values=(uid, name, ts, f"{int(conf * 100)}%"))

        self.present_count_text.set(f"Total Present Today: {len(rows)}")

    def _export_csv(self):
        rows = self.db.get_today_attendance()
        if not rows:
            self.status_banner_text.set("⚠️ No attendance records today to export.")
            return

        default_name = f"attendance_{datetime.now().strftime('%Y_%m_%d')}.csv"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["User ID", "Full Name", "Check-In Time", "Date", "Confidence"])
                today_str = datetime.now().strftime("%Y-%m-%d")
                for uid, name, ts, conf in rows:
                    writer.writerow([uid, name, ts, today_str, f"{conf:.2f}"])
            self.status_banner_text.set(f"📥 Exported {len(rows)} attendance records to {os.path.basename(file_path)}")
        except Exception as e:
            self.status_banner_text.set(f"⚠️ CSV Export failed: {e}")

    def _clear_today_logs(self):
        self.db.clear_today_attendance()
        self._refresh_attendance_table()
        self.status_banner_text.set("🗑️ Cleared today's attendance log.")

    def _on_close(self):
        self.is_running = False
        if self.vs is not None:
            self.vs.stop()
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = PiAttendanceApp(root)
    root.mainloop()