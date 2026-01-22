import cv2
import joblib
import numpy as np
from collections import deque, Counter

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== PATHLER ==================
MODEL_PATH = "sign_language/asl_model.pkl"
LANDMARKER_PATH = "sign_language/models/hand_landmarker.task"

# ================== KAMERA ==================
CAM_INDEX = 1
CAM_BACKEND = cv2.CAP_DSHOW
ROTATE_MODE = cv2.ROTATE_90_COUNTERCLOCKWISE
MIRROR = False

# Performans:
PREDICT_EVERY_N_FRAMES = 2   # 1=her frame, 2=her 2 frame (önerilir)
FRAME_WIDTH = 640            # daha küçük = daha hızlı
BUFFER_SIZE = 1              # gecikmeyi azaltır

# ================== KARAR / STABİLİZASYON ==================
WINDOW = 10              # çoğunluk oyu penceresi
MIN_VOTES = 6            # bu kadar oy alan label ekrana yazılır

PROBA_THRESHOLD = 0.55   # çok Unknown olursa 0.50 dene
MARGIN_THRESHOLD = 0.08  # top1-top2 farkı küçükse kararsız

HOLD_LAST_STABLE = True  # kararsızda son stabil harfi tut

SHOW_TOPK = 2

# ================== MODEL ==================
model = joblib.load(MODEL_PATH)
CLASSES = list(model.classes_) if hasattr(model, "classes_") else None

# ================== MEDIAPIPE ==================
base_options = python.BaseOptions(model_asset_path=LANDMARKER_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ================== FEATURE ==================
def extract_raw_features(result):
    if not result.hand_landmarks:
        return None
    feats = []
    for lm in result.hand_landmarks[0]:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=np.float32).reshape(1, -1)

def normalize_row(x63):
    pts = x63.reshape(21, 3).astype(np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9])  # wrist->middle_mcp
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.reshape(1, -1)

def topk_from_proba(proba, k=2):
    idx = np.argsort(proba)[::-1][:k]
    out = []
    for i in idx:
        out.append((CLASSES[i], float(proba[i])))
    return out

def decide_label_from_proba(proba):
    topk = topk_from_proba(proba, k=max(2, SHOW_TOPK))
    top1_label, top1_p = topk[0]
    top2_p = topk[1][1] if len(topk) > 1 else 0.0

    ok_conf = top1_p >= PROBA_THRESHOLD
    ok_margin = (top1_p - top2_p) >= MARGIN_THRESHOLD

    if ok_conf and ok_margin:
        return top1_label, topk
    return "Unknown", topk

# ================== MAIN ==================
def main():
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, CAM_BACKEND)

    # Kamera gecikmesi azaltma
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

    history = deque(maxlen=WINDOW)
    shown_label = "Unknown"
    last_topk = []
    frame_id = 0

    last_pred = "Unknown"  # her frame tahmin etmeyince bunu tutacağız

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.rotate(frame, ROTATE_MODE)
        if MIRROR:
            frame = cv2.flip(frame, 1)

        # Sadece her N frame'de bir tahmin
        do_predict = (frame_id % PREDICT_EVERY_N_FRAMES == 0)
        frame_id += 1

        if do_predict:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = hand_landmarker.detect(mp_image)

            raw = extract_raw_features(result)
            if raw is not None and hasattr(model, "predict_proba"):
                X = normalize_row(raw)
                proba = model.predict_proba(X)[0]
                last_pred, last_topk = decide_label_from_proba(proba)
            elif raw is not None:
                X = normalize_row(raw)
                last_pred = model.predict(X)[0]
                last_topk = []
            else:
                last_pred = "Unknown"
                last_topk = []

        # History'e son tahmini ekle
        history.append(last_pred)

        # Çoğunluk oyu
        counts = Counter([h for h in history if h != "Unknown"])
        if counts:
            best_label, votes = counts.most_common(1)[0]
            if votes >= MIN_VOTES:
                shown_label = best_label
            else:
                if not HOLD_LAST_STABLE:
                    shown_label = "Unknown"
        else:
            if not HOLD_LAST_STABLE:
                shown_label = "Unknown"

        # UI
        cv2.putText(frame, f"Prediction: {shown_label}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.putText(frame, f"Win:{WINDOW}  MinVotes:{MIN_VOTES}  Every:{PREDICT_EVERY_N_FRAMES}f",
                    (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Th:{PROBA_THRESHOLD:.2f}  Margin:{MARGIN_THRESHOLD:.2f}",
                    (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if last_topk:
            y = 170
            cv2.putText(frame, f"Top-{SHOW_TOPK}:", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y += 28
            for lab, p in last_topk[:SHOW_TOPK]:
                cv2.putText(frame, f"{lab}: {p:.2f}", (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                y += 26

        cv2.putText(frame, "q=quit  c=clear", (30, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("ASL Live Prediction (Fast+Stable)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            history.clear()
            shown_label = "Unknown"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
