from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model
import base64
import math

app = Flask(__name__)

try:
    print("⏳ [Startup] Memuat model...")
    HEAD_POSE_MODELS = {
        'pitch': joblib.load('model/headposeModel/xgb_pitch_model.joblib'),
        'yaw': joblib.load('model/headposeModel/xgb_yaw_model.joblib'),
        'roll': joblib.load('model/headposeModel/xgb_roll_model.joblib')
    }
    EYE_STATUS_MODEL = load_model('model/eyeModel/eye_status_model.h5')
    YAWN_SVM_MODEL = joblib.load('model/yawnModel/svm_yawn_detector.joblib')
    YAWN_SCALER = joblib.load('model/yawnModel/scaler.joblib')
    print("✅ [Startup] Semua model berhasil dimuat.")
except Exception as e:
    print(f"❌ [Startup] Fatal: Error saat memuat model: {e}")
    exit()

EYE_IMG_SIZE = 24
EYE_CLASS_LABELS = ['Tertutup', 'Terbuka']
LEFT_EYE_IDXS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_IDXS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def create_feature_vector(face_landmarks):
    anchor_point = face_landmarks.landmark[1]
    p_left, p_right = face_landmarks.landmark[359], face_landmarks.landmark[130]
    scale_distance = np.linalg.norm([p_left.x - p_right.x, p_left.y - p_right.y])
    if scale_distance < 1e-6: return None
    feature_vector = []
    for i in range(468):
        if i == 1: continue
        landmark = face_landmarks.landmark[i]
        feature_vector.extend([(landmark.x - anchor_point.x) / scale_distance, (landmark.y - anchor_point.y) / scale_distance, (landmark.z - anchor_point.z) / scale_distance])
    return np.array(feature_vector)

def preprocess_eye_for_predict(eye_img):
    if eye_img is None or eye_img.size == 0: return None
    gray_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    normalized_eye = resized_eye / 255.0
    return np.expand_dims(normalized_eye, axis=-1)

def calculate_mar(face_landmarks):
    p_v1, p_v2 = face_landmarks.landmark[13], face_landmarks.landmark[14]
    p_h1, p_h2 = face_landmarks.landmark[78], face_landmarks.landmark[308]
    v_dist = np.linalg.norm([p_v1.x - p_v2.x, p_v1.y - p_v2.y])
    h_dist = np.linalg.norm([p_h1.x - p_h2.x, p_h1.y - p_h2.y])
    return v_dist / h_dist if h_dist > 0 else 0.0


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detector')
def detector():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    
    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    h, w = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    response_data = {"face_found": False}

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        response_data["face_found"] = True

        feature_vector = create_feature_vector(face_landmarks)
        if feature_vector is not None:
            pitch = HEAD_POSE_MODELS['pitch'].predict(feature_vector.reshape(1, -1))[0] * 180 / np.pi
            yaw = HEAD_POSE_MODELS['yaw'].predict(feature_vector.reshape(1, -1))[0] * 180 / np.pi
            roll = HEAD_POSE_MODELS['roll'].predict(feature_vector.reshape(1, -1))[0] * 180 / np.pi
            response_data['head_pose'] = {'pitch': pitch, 'yaw': yaw, 'roll': roll}

        all_x = [lm.x * w for lm in face_landmarks.landmark]
        all_y = [lm.y * h for lm in face_landmarks.landmark]
        response_data['face_box'] = [min(all_x), min(all_y), max(all_x), max(all_y)]

        eye_boxes = []
        eye_statuses = []
        padding = 5
        for idxs in [LEFT_EYE_IDXS, RIGHT_EYE_IDXS]:
            points = np.array([[lm.x * w, lm.y * h] for lm in [face_landmarks.landmark[i] for i in idxs]]).astype(int)
            x, y, pw, ph = cv2.boundingRect(points)
            eye_boxes.append([x - padding, y - padding, x + pw + padding, y + ph + padding])
            
            eye_crop = frame[y-padding:y+ph+padding, x-padding:x+pw+padding]
            processed = preprocess_eye_for_predict(eye_crop)
            if processed is not None:
                prediction = EYE_STATUS_MODEL.predict(np.array([processed]), verbose=0)
                status = EYE_CLASS_LABELS[1] if prediction[0][0] > 0.5 else EYE_CLASS_LABELS[0]
                eye_statuses.append(status)

        response_data['eye_boxes'] = eye_boxes
        response_data['eye_statuses'] = eye_statuses
        
        response_data['mar'] = calculate_mar(face_landmarks)

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)