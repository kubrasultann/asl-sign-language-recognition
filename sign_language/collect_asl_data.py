import cv2
import csv
import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== AYARLAR ==================
DATASET_FILE = "sign_language/dataset_asl.csv"
MODEL_PATH = "sign_language/models/hand_landmarker.task"

CAM_INDEX = 1
CAM_BACKEND = cv2.CAP_DSHOW

ROTATE_MODE = cv2.ROTATE_90_COUNTERCLOCKWISE
MIRROR = False

# ASL (J ve Z hariç - hareketli)
ASL_LETTERS = [
    "A","B","C","D","E","F","G","H","I",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y"
]
LABELS = {ch.lower(): ch for ch in ASL_LETTERS}

# ================== MEDIAPIPE TASKS ==================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ================== CSV ==================
os.makedirs(os.path.dirname(DATASET_FILE), exist_ok=True)

csv_file = open(DATASET_FILE, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

def features_one_hand(result):
    """63 özellik: (x,y,z)*21. El yoksa None."""
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    feats = []
    for lm in hand:
        feats.extend([lm.x, lm.y, lm.z])
    return feats

def draw_points(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

def is_up_arrow(key):
    return key in (82, 2490368)

def is_down_arrow(key):
    return key in (84, 2621440)

def main():
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    if not cap.isOpened():
        print("CAM_INDEX=1 açılmadı. CAM_INDEX=0 deneniyor...")
        cap = cv2.VideoCapture(0, CAM_BACKEND)

    if not cap.isOpened():
        print("Kamera açılamadı. DroidCam Client açık mı? Start yaptın mı?")
        return

    print("=== ASL Veri Toplama ===")
    print("Harf tuşları (a,b,c...): label seçer")
    print("Kontroller:")
    print("  ↑  : kayıt aç/kapat")
    print("  ↓  : kaydı kapat")
    print("  ESC: çık\n")

    selected_label = None
    recording = False

    # sayaçlar
    total_rows = 0
    label_rows = 0  # seçili label için yazılan satır sayısı

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame alınamadı. DroidCam bağlantısını kontrol et.")
            break

        frame = cv2.rotate(frame, ROTATE_MODE)
        if MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = hand_landmarker.detect(mp_image)

        if result.hand_landmarks:
            draw_points(frame, result.hand_landmarks[0])

        feats = features_one_hand(result)

        if recording and selected_label and feats is not None:
            writer.writerow(feats + [selected_label])
            total_rows += 1
            label_rows += 1

        # ekran yazıları
        cv2.putText(frame, f"Label: {selected_label if selected_label else 'None'}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        cv2.putText(frame, f"Recording: {recording}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, f"Label frames: {label_rows}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(frame, f"Total rows: {total_rows}",
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)

        cv2.putText(frame, "UP=rec toggle | DOWN=rec off | ESC=quit",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("ASL Data Collector", frame)

        key = cv2.waitKeyEx(1)

        if key == 27:  # ESC
            break

        # kayıt kontrolü
        if is_up_arrow(key):
            recording = not recording
            print("Recording:", recording)
            continue

        if is_down_arrow(key):
            if recording:
                recording = False
                print("Recording: False")
            continue

        # label seçimi (label seçince kayıt otomatik kapanır + label sayacı sıfırlanır)
        if key != -1 and key != 255:
            try:
                ch = chr(key).lower()
                if ch in LABELS:
                    selected_label = LABELS[ch]
                    recording = False
                    label_rows = 0
                    print("Seçilen label:", selected_label)
            except:
                pass

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
