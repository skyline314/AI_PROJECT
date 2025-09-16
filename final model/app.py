import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import math
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ==================================================================================================
# KONFIGURASI DAN PEMUATAN MODEL (DARI final.py)
# ==================================================================================================

# Threshold
HEAD_DOWN_FRAMES_THRESHOLD = 20
EYES_CLOSED_FRAMES_THRESHOLD = 15
PITCH_THRESHOLD_DEG = 30 

# Model HeadPose 
try:
    HEAD_POSE_MODELS = {
        'pitch': joblib.load('model/headposeModel/xgb_pitch_model.joblib'),
        'yaw': joblib.load('model/headposeModel/xgb_yaw_model.joblib'),
        'roll': joblib.load('model/headposeModel/xgb_roll_model.joblib')
    }
except FileNotFoundError:
    st.error("File Model Head Pose tidak ditemukan. Pastikan path model benar.")
    st.stop()

# Model EyeStatus
try:
    EYE_STATUS_MODEL = load_model('model/eyeModel/eye_status_model.h5')
except (FileNotFoundError, IOError):
    st.error("File Model Eye Status tidak ditemukan. Pastikan path model benar.")
    st.stop()
EYE_IMG_SIZE = 24
EYE_CLASS_LABELS = ['Mata Tertutup', 'Mata Terbuka']
LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


# Model Yawn Detection
try:
    YAWN_SVM_MODEL = joblib.load('model/yawnModel/svm_yawn_detector.joblib')
    YAWN_SCALER = joblib.load('model/yawnModel/scaler.joblib')
except FileNotFoundError:
    st.error("File Yawn Detection tidak ditemukan. Pastikan path model benar.")
    st.stop()
MAR_WINDOW_SIZE = 20
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

#  MediaPipe Face Mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==================================================================================================
# FUNGSI-FUNGSI BANTUAN (DARI final.py)
# ==================================================================================================
def create_feature_vector(face_landmarks):
    anchor_point = face_landmarks.landmark[1]
    p_left = face_landmarks.landmark[359]
    p_right = face_landmarks.landmark[130]
    scale_distance = np.linalg.norm([p_left.x - p_right.x, p_left.y - p_right.y])
    if scale_distance < 1e-6: return None
    feature_vector = []
    for i in range(468):
        if i == 1: continue
        landmark = face_landmarks.landmark[i]
        feature_vector.extend([(landmark.x - anchor_point.x) / scale_distance, (landmark.y - anchor_point.y) / scale_distance, (landmark.z - anchor_point.z) / scale_distance])
    return np.array(feature_vector)

def draw_axes(img, pitch, yaw, roll, nose_2d, size=100):
    if nose_2d is None: return img
    pitch_rad = pitch * np.pi / 180
    yaw_rad = -(yaw * np.pi / 180)
    roll_rad = roll * np.pi / 180
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    Ry = np.array([[math.cos(yaw_rad), 0, math.sin(yaw_rad)], [0, 1, 0], [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])
    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0], [math.sin(roll_rad), math.cos(roll_rad), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    axis = np.array([[size, 0, 0], [0, size, 0], [0, 0, size]])
    rotated_axis = R @ axis
    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2_yaw = (int(nose_2d[0] + rotated_axis[0, 0]), int(nose_2d[1] + rotated_axis[1, 0])); cv2.line(img, p1, p2_yaw, (255, 0, 0), 3)
    p2_pitch = (int(nose_2d[0] + rotated_axis[0, 1]), int(nose_2d[1] + rotated_axis[1, 1])); cv2.line(img, p1, p2_pitch, (0, 255, 0), 3)
    p2_roll = (int(nose_2d[0] + rotated_axis[0, 2]), int(nose_2d[1] + rotated_axis[1, 2])); cv2.line(img, p1, p2_roll, (0, 0, 255), 3)
    return img

def preprocess_eye_for_predict(eye_img):
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    normalized_eye = resized_eye / 255.0
    return np.expand_dims(normalized_eye, axis=-1)

def calculate_mar(face_landmarks):
    p_vertical1 = face_landmarks.landmark[13]
    p_vertical2 = face_landmarks.landmark[14]
    p_horizontal1 = face_landmarks.landmark[78]
    p_horizontal2 = face_landmarks.landmark[308]
    vertical_dist = np.linalg.norm([p_vertical1.x - p_vertical2.x, p_vertical1.y - p_vertical2.y])
    horizontal_dist = np.linalg.norm([p_horizontal1.x - p_horizontal2.x, p_horizontal1.y - p_horizontal2.y])
    return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.0


# ==================================================================================================
# KELAS PROSESOR VIDEO UNTUK STREAMLIT-WEBRTC
# ==================================================================================================

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        # Inisialisasi variabel status dan counter
        self.head_down_counter = 0
        self.eyes_closed_counter = 0
        self.mar_buffer = deque(maxlen=MAR_WINDOW_SIZE)
        
        self.PITCH_THRESHOLD_RAD = -PITCH_THRESHOLD_DEG * np.pi / 180
        
        # Variabel untuk frame skipping
        self.frame_counter = 0
        self.PROCESSING_INTERVAL = 3  
        
        # Variabel untuk menyimpan status terakhir
        self.drowsiness_alert = False
        self.alert_reason = ""
        self.yawn_status = "NORMAL"
        self.current_mar = 0.0
        self.pitch_deg, self.yaw_deg, self.roll_deg = 0, 0, 0
        self.eye_statuses = ["N/A", "N/A"] 

    def _update_status_display(self):
        """Fungsi untuk memperbarui placeholder di UI Streamlit."""
        with eye_status_placeholder.container():
            st.markdown(f"**Status Mata Kiri:** `{self.eye_statuses[0]}`")
            st.markdown(f"**Status Mata Kanan:** `{self.eye_statuses[1]}`")

        with mar_status_placeholder.container():
            st.markdown(f"**Status Mulut:** `{self.yawn_status}`")
            st.metric(label="MAR (Mouth Aspect Ratio)", value=f"{self.current_mar:.2f}")

        with head_pose_placeholder.container():
            st.metric(label="Pitch (Hijau)", value=f"{self.pitch_deg:.1f}°")
            st.metric(label="Yaw (Biru)", value=f"{self.yaw_deg:.1f}°")
            st.metric(label="Roll (Merah)", value=f"{self.roll_deg:.1f}°")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        h_orig, w_orig, _ = img.shape
        new_w = 640
        new_h = int(new_w * (h_orig / w_orig))
        frame = cv2.resize(img, (new_w, new_h))
        h, w = new_h, new_w

        frame = cv2.flip(frame, 1)
        self.frame_counter += 1

        # Variabel sementara untuk digambar di frame
        nose_2d = None
        eye_draw_info = []
        
        if self.frame_counter % self.PROCESSING_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            self.drowsiness_alert = False
            self.alert_reason = ""
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # 1. Analisis Pose Kepala
                feature_vector = create_feature_vector(face_landmarks)
                if feature_vector is not None:
                    feature_vector = feature_vector.reshape(1, -1)
                    pitch_rad = HEAD_POSE_MODELS['pitch'].predict(feature_vector)[0]
                    yaw_rad = HEAD_POSE_MODELS['yaw'].predict(feature_vector)[0]
                    roll_rad = HEAD_POSE_MODELS['roll'].predict(feature_vector)[0]

                    self.pitch_deg = pitch_rad * 180 / np.pi
                    self.yaw_deg = yaw_rad * 180 / np.pi
                    self.roll_deg = roll_rad * 180 / np.pi
                    
                    if pitch_rad < self.PITCH_THRESHOLD_RAD: self.head_down_counter += 1
                    else: self.head_down_counter = 0
                    
                    if self.head_down_counter >= HEAD_DOWN_FRAMES_THRESHOLD:
                        self.drowsiness_alert = True
                        self.alert_reason = "Kepala Terkulai"

                    nose_2d = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)

                # 2. Analisis Status Mata
                padding = 5
                eye_crops_to_predict = []
                eye_positions_for_drawing = []

                # Mata Kiri
                left_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE_IDXS]).astype(int)
                lx_min, ly_min = np.min(left_eye_points, axis=0); lx_max, ly_max = np.max(left_eye_points, axis=0)
                if ly_max > ly_min and lx_max > lx_min:
                    left_eye_crop = frame[ly_min-padding:ly_max+padding, lx_min-padding:lx_max+padding]
                    if left_eye_crop.size > 0:
                        eye_crops_to_predict.append(preprocess_eye_for_predict(left_eye_crop))
                        eye_positions_for_drawing.append({'label_pos': (lx_min-padding, ly_min-15)})
                
                # Mata Kanan
                right_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE_IDXS]).astype(int)
                rx_min, ry_min = np.min(right_eye_points, axis=0); rx_max, ry_max = np.max(right_eye_points, axis=0)
                if ry_max > ry_min and rx_max > rx_min:
                    right_eye_crop = frame[ry_min-padding:ry_max+padding, rx_min-padding:rx_max+padding]
                    if right_eye_crop.size > 0:
                        eye_crops_to_predict.append(preprocess_eye_for_predict(right_eye_crop))
                        eye_positions_for_drawing.append({'label_pos': (rx_min-padding, ry_min-15)})

                if eye_crops_to_predict:
                    predictions = EYE_STATUS_MODEL.predict(np.array(eye_crops_to_predict), verbose=0)
                    
                    temp_eye_statuses = []
                    for i, pred in enumerate(predictions):
                        status = EYE_CLASS_LABELS[1] if pred[0] > 0.5 else EYE_CLASS_LABELS[0]
                        temp_eye_statuses.append(status)
                        color = (0, 255, 0) if status == 'Mata Terbuka' else (0, 0, 255)
                        eye_draw_info.append({'status': status, 'color': color, 'pos': eye_positions_for_drawing[i]})
                    
                    self.eye_statuses = temp_eye_statuses if len(temp_eye_statuses) == 2 else ["N/A", "N/A"]
                    
                    if len(temp_eye_statuses) == 2 and all(s == 'Mata Tertutup' for s in temp_eye_statuses): self.eyes_closed_counter += self.PROCESSING_INTERVAL
                    else: self.eyes_closed_counter = 0
                
                if self.eyes_closed_counter >= EYES_CLOSED_FRAMES_THRESHOLD:
                    self.drowsiness_alert = True
                    self.alert_reason = "Mata Tertutup"

                # 3. Analisis Menguap 
                self.current_mar = calculate_mar(face_landmarks)
                self.mar_buffer.append(self.current_mar)

                if len(self.mar_buffer) == MAR_WINDOW_SIZE:
                    features = np.array([[np.mean(self.mar_buffer), np.max(self.mar_buffer), np.std(self.mar_buffer)]])
                    prediction = YAWN_SVM_MODEL.predict(YAWN_SCALER.transform(features))[0]
                    self.yawn_status = "MENGUAP" if prediction == 1 else "NORMAL"
                
                if self.yawn_status == "MENGUAP":
                    self.drowsiness_alert = True
                    self.alert_reason = "Menguap"
        
        # Gambar visualisasi
        if nose_2d:
            draw_axes(frame, self.pitch_deg, self.yaw_deg, self.roll_deg, nose_2d)

        for info in eye_draw_info:
            cv2.putText(frame, info['status'], info['pos']['label_pos'], cv2.FONT_HERSHEY_SIMPLEX, 0.6, info['color'], 2)

        if self.drowsiness_alert:
            cv2.putText(frame, "!!! PERINGATAN KANTUK !!!", (int(w/2) - 250, int(h/2) - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Penyebab: {self.alert_reason}", (int(w/2) - 120, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Perbarui UI di thread utama
        self._update_status_display()

        return frame

# ==================================================================================================
# UI STREAMLIT
# ==================================================================================================

st.set_page_config(layout="wide")
st.title("Sistem Deteksi Kantuk 😴")
st.markdown("Aplikasi ini mendeteksi tanda-tanda kantuk melalui webcam Anda secara *real-time*.")

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Visualisasi Kamera")
    webrtc_streamer(
        key="drowsiness-detection",
        video_transformer_factory=DrowsinessTransformer,
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.header("Status Deteksi")
    
    with st.expander("👁️ Status Mata", expanded=True):
        eye_status_placeholder = st.empty()
        with eye_status_placeholder.container():
            st.markdown("**Status Mata Kiri:** `N/A`")
            st.markdown("**Status Mata Kanan:** `N/A`")

    with st.expander("👄 Status Mulut (MAR)", expanded=True):
        mar_status_placeholder = st.empty()
        with mar_status_placeholder.container():
            st.markdown("**Status Mulut:** `NORMAL`")
            st.metric(label="MAR (Mouth Aspect Ratio)", value="0.00")

    with st.expander("🧠 Status Posisi Kepala", expanded=True):
        head_pose_placeholder = st.empty()
        with head_pose_placeholder.container():
            st.metric(label="Pitch (Tundukan)", value="0.0°")
            st.metric(label="Yaw (Tolehan)", value="0.0°")
            st.metric(label="Roll (Miring)", value="0.0°")

st.info("Arahkan wajah Anda ke kamera dan klik 'START' untuk memulai deteksi. Berikan izin akses kamera jika diminta.")