import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import math
from playsound import playsound

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
    print("Model Head Pose berhasil dimuat")
except FileNotFoundError:
    print("File Model Head Pose tidak ditemukan")
    exit()

# Model EyeStatus
try:
    EYE_STATUS_MODEL = load_model('model/eyeModel/eye_status_model.h5')
    print("Model Eye Status berhasil dimuat")
except (FileNotFoundError, IOError):
    print("File Model Eye Status tidak ditemukan")
    exit()
EYE_IMG_SIZE = 24
EYE_CLASS_LABELS = ['Mata Tertutup', 'Mata Terbuka']
LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


# Model Yawn Detection
try:
    YAWN_SVM_MODEL = joblib.load('model/yawnModel/svm_yawn_detector.joblib')
    YAWN_SCALER = joblib.load('model/yawnModel/scaler.joblib')
    print("Model Yawn Detection berhasil dimuat")
except FileNotFoundError:
    print("File Yawn Detection tidak ditemukan")
    exit()
MAR_WINDOW_SIZE = 20
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

#  MediaPipe Face Mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # landmark mata yang detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# FUNGSI-FUNGSI BANTUAN 

# Fungsi untuk HeadPose
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
    p2_yaw = (int(nose_2d[0] + rotated_axis[0, 0]), int(nose_2d[1] + rotated_axis[1, 0])); cv2.line(img, p1, p2_yaw, (255, 0, 0), 3) # Yaw (Biru)
    p2_pitch = (int(nose_2d[0] + rotated_axis[0, 1]), int(nose_2d[1] + rotated_axis[1, 1])); cv2.line(img, p1, p2_pitch, (0, 255, 0), 3) # Pitch (Hijau)
    p2_roll = (int(nose_2d[0] + rotated_axis[0, 2]), int(nose_2d[1] + rotated_axis[1, 2])); cv2.line(img, p1, p2_roll, (0, 0, 255), 3) # Roll (Merah)
    return img

# Fungsi untuk Eyes Status
def preprocess_eye_for_predict(eye_img):
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    normalized_eye = resized_eye / 255.0
    # Ubah dimensi untuk batch prediction
    return np.expand_dims(normalized_eye, axis=-1)

# Fungsi untuk Yawn Detection 
def calculate_mar(face_landmarks):
    p_vertical1 = face_landmarks.landmark[13]
    p_vertical2 = face_landmarks.landmark[14]
    p_horizontal1 = face_landmarks.landmark[78]
    p_horizontal2 = face_landmarks.landmark[308]
    vertical_dist = np.linalg.norm([p_vertical1.x - p_vertical2.x, p_vertical1.y - p_vertical2.y])
    horizontal_dist = np.linalg.norm([p_horizontal1.x - p_horizontal2.x, p_horizontal1.y - p_horizontal2.y])
    return vertical_dist / horizontal_dist if horizontal_dist > 0 else 0.0

# Jalankan semuanya 

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera")
        return
    print("Webcam berhasil dibuka. Tekan 'q' untuk keluar")

    # Inisialisasi variabel status dan counter
    head_down_counter = 0
    eyes_closed_counter = 0
    mar_buffer = deque(maxlen=MAR_WINDOW_SIZE)
    alarm_playing = False
    
    PITCH_THRESHOLD_RAD = -PITCH_THRESHOLD_DEG * np.pi / 180

    # ===== OPTIMISASI: Inisialisasi variabel untuk frame skipping =====
    frame_counter = 0
    PROCESSING_INTERVAL = 3  # Hanya proses setiap 3 frame
    
    # Variabel untuk menyimpan status terakhir
    drowsiness_alert = False
    alert_reason = ""
    yawn_status = "NORMAL"
    current_mar = 0.0
    pitch_deg, yaw_deg, roll_deg = 0, 0, 0
    nose_2d = None
    face_box = None
    mouth_box = None
    eye_draw_info = [] # Untuk menyimpan info gambar mata

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # ===== OPTIMISASI 1: MENGURANGI RESOLUSI FRAME =====
        h_orig, w_orig, _ = frame.shape
        new_w = 640
        new_h = int(new_w * (h_orig / w_orig))
        frame = cv2.resize(frame, (new_w, new_h))
        h, w = new_h, new_w
        # ===================================================

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # ===== OPTIMISASI 2: FRAME SKIPPING =====
        # Semua proses berat (deteksi) hanya berjalan pada interval yang ditentukan
        if frame_counter % PROCESSING_INTERVAL == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            # Reset status per frame yang diproses
            drowsiness_alert = False
            alert_reason = ""
            eye_draw_info = [] # Reset info gambar mata

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # 1. Analisis Pose Kepala 
                feature_vector = create_feature_vector(face_landmarks)
                if feature_vector is not None:
                    feature_vector = feature_vector.reshape(1, -1)
                    pitch_rad = HEAD_POSE_MODELS['pitch'].predict(feature_vector)[0]
                    yaw_rad = HEAD_POSE_MODELS['yaw'].predict(feature_vector)[0]
                    roll_rad = HEAD_POSE_MODELS['roll'].predict(feature_vector)[0]

                    pitch_deg = pitch_rad * 180 / np.pi
                    yaw_deg = yaw_rad * 180 / np.pi
                    roll_deg = roll_rad * 180 / np.pi

                    if pitch_rad < PITCH_THRESHOLD_RAD:
                        head_down_counter += 1
                    else:
                        head_down_counter = 0
                    
                    if head_down_counter >= HEAD_DOWN_FRAMES_THRESHOLD:
                        drowsiness_alert = True
                        alert_reason = "Kepala Terkulai"

                    nose_2d = (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h)
                
                # 2. Analisis Status Mata (Persiapan Batch)
                padding = 5
                eye_crops_to_predict = []
                eye_positions_for_drawing = []

                # Mata Kiri
                left_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE_IDXS]).astype(int)
                lx_min, ly_min = np.min(left_eye_points, axis=0)
                lx_max, ly_max = np.max(left_eye_points, axis=0)
                if ly_max > ly_min and lx_max > lx_min:
                    left_eye_crop = frame[ly_min-padding:ly_max+padding, lx_min-padding:lx_max+padding]
                    if left_eye_crop.size > 0:
                        processed_eye = preprocess_eye_for_predict(left_eye_crop)
                        eye_crops_to_predict.append(processed_eye)
                        eye_positions_for_drawing.append({
                            'box': (lx_min-padding, ly_min-padding, lx_max+padding, ly_max+padding),
                            'label_pos': (lx_min-padding, ly_min-15)
                        })
                
                # Mata Kanan
                right_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE_IDXS]).astype(int)
                rx_min, ry_min = np.min(right_eye_points, axis=0)
                rx_max, ry_max = np.max(right_eye_points, axis=0)
                if ry_max > ry_min and rx_max > rx_min:
                    right_eye_crop = frame[ry_min-padding:ry_max+padding, rx_min-padding:rx_max+padding]
                    if right_eye_crop.size > 0:
                        processed_eye = preprocess_eye_for_predict(right_eye_crop)
                        eye_crops_to_predict.append(processed_eye)
                        eye_positions_for_drawing.append({
                            'box': (rx_min-padding, ry_min-padding, rx_max+padding, ry_max+padding),
                            'label_pos': (rx_min-padding, ry_min-15)
                        })

                # ===== OPTIMISASI 3: BATCH INFERENCE UNTUK MATA =====
                if eye_crops_to_predict:
                    batch_input = np.array(eye_crops_to_predict)
                    predictions = EYE_STATUS_MODEL.predict(batch_input, verbose=0)
                    
                    eye_statuses = []
                    for i, pred in enumerate(predictions):
                        status = EYE_CLASS_LABELS[1] if pred[0] > 0.5 else EYE_CLASS_LABELS[0]
                        eye_statuses.append(status)
                        color = (0, 255, 0) if status == 'Mata Terbuka' else (0, 0, 255)
                        # Simpan info untuk digambar di luar if-block
                        eye_draw_info.append({'status': status, 'color': color, 'pos': eye_positions_for_drawing[i]})
                    
                    # Cek kondisi mata tertutup jika kedua mata terdeteksi
                    if len(eye_statuses) == 2 and all(s == 'Mata Tertutup' for s in eye_statuses):
                        eyes_closed_counter += PROCESSING_INTERVAL # Tambah sesuai interval
                    else:
                        eyes_closed_counter = 0
                
                if eyes_closed_counter >= EYES_CLOSED_FRAMES_THRESHOLD:
                    drowsiness_alert = True
                    alert_reason = "Mata Tertutup"

                # 3. Analisis Menguap 
                current_mar = calculate_mar(face_landmarks)
                mar_buffer.append(current_mar)

                if len(mar_buffer) == MAR_WINDOW_SIZE:
                    features = np.array([[np.mean(mar_buffer), np.max(mar_buffer), np.std(mar_buffer)]])
                    scaled_features = YAWN_SCALER.transform(features)
                    prediction = YAWN_SVM_MODEL.predict(scaled_features)[0]
                    yawn_status = "MENGUAP" if prediction == 1 else "NORMAL"
                
                if yawn_status == "MENGUAP":
                    drowsiness_alert = True
                    alert_reason = "Menguap"
                
                # Simpan Bounding Box untuk digambar nanti
                all_x = [lm.x * w for lm in face_landmarks.landmark]
                all_y = [lm.y * h for lm in face_landmarks.landmark]
                face_box = (int(min(all_x)), int(min(all_y)), int(max(all_x)), int(max(all_y)))
                
                mouth_x = [face_landmarks.landmark[i].x * w for i in MOUTH_INDICES]
                mouth_y = [face_landmarks.landmark[i].y * h for i in MOUTH_INDICES]
                mouth_box = (int(min(mouth_x)), int(min(mouth_y)), int(max(mouth_x)), int(max(mouth_y)))
        
        # --- Bagian menggambar dan menampilkan selalu berjalan setiap frame ---
        # Ini membuat video terlihat mulus, menggunakan data dari frame terakhir yang diproses
        
        # Gambar Face Box
        if face_box:
            cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
            
        # Gambar Axes Head Pose
        if nose_2d:
            draw_axes(frame, pitch_deg, yaw_deg, roll_deg, nose_2d)
        
        # Tampilkan teks Head Pose
        cv2.putText(frame, f"Pitch: {pitch_deg:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw_deg:.1f}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Roll: {roll_deg:.1f}", (w - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Gambar Info Mata
        for info in eye_draw_info:
            pos = info['pos']
            box = pos['box']
            label_pos = pos['label_pos']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), info['color'], 2)
            cv2.putText(frame, info['status'], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, info['color'], 2)
        
        # Gambar Info Mulut
        if mouth_box:
            cv2.rectangle(frame, (mouth_box[0], mouth_box[1]), (mouth_box[2], mouth_box[3]), (255, 0, 0), 1)
        cv2.putText(frame, f"Mulut: {yawn_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {current_mar:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Tampilkan Peringatan jika terdeteksi kantuk
        if drowsiness_alert:
            cv2.putText(frame, "!!! PERINGATAN KANTUK !!!", (int(w/2) - 250, int(h/2) - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Penyebab: {alert_reason}", (int(w/2) - 120, int(h/2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if not alarm_playing:
                try:
                    playsound('alarm3.mp3', block=False) 
                    alarm_playing = True
                    print("ALARM DIMULAI!")
                except Exception as e:
                    print(f"Error memainkan suara: {e}")
        else:
            if alarm_playing:
                alarm_playing = False
                print("Alarm direset")

        cv2.imshow('Sistem Deteksi Kantuk Terintegrasi', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Aplikasi ditutup.")

if __name__ == '__main__':
    main()