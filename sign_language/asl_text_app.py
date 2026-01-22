import cv2
import joblib
import numpy as np
from collections import deque, Counter
import os
import time

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

# ================== PERFORMANS ==================
PREDICT_EVERY_N_FRAMES = 2
FRAME_WIDTH = 900
BUFFER_SIZE = 1

# Kamera gecikmesini azaltmak için (UI değişmez)
FLUSH_GRABS_PER_LOOP = 1  # bazen gereksiz overhead; 1 daha stabil

# Tahmin için frame küçültme (UI aynı kalır; sadece model/mediapipe daha hızlı çalışır)
PROCESS_SCALE = 0.85  

# ================== KARAR/STABİLİZASYON ==================
WINDOW = 12
MIN_VOTES = 8

PROBA_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.08

HOLD_LAST_STABLE = True

SHOW_TOPK = 2
NO_HAND_RESET_FRAMES = 3  # SADECE "EL YOK" ise reset

# ================== AUTO YAZMA ==================
AUTO_HOLD_FRAMES = 12
COOLDOWN_FRAMES = 14
REQUIRE_CHANGE_BEFORE_REPEAT = True

# ================== UI (Modern) ==================
FONT = cv2.FONT_HERSHEY_SIMPLEX

WIN_W = 1280
WIN_H = 720

PAD = 18
GAP = 14
HEADER_H = 64
FOOTER_H = 70

CAM_W = 760
CAM_H = WIN_H - HEADER_H - FOOTER_H - 2 * PAD

RIGHT_W = WIN_W - (2 * PAD) - CAM_W - GAP

STATUS_H = 200
TOPK_H = 170
TEXT_H = CAM_H - STATUS_H - TOPK_H - 2 * GAP

COL_BG = (18, 18, 20)
COL_PANEL = (28, 28, 32)
COL_BORDER = (70, 70, 78)
COL_TITLE = (240, 240, 240)
COL_TEXT = (220, 220, 220)
COL_MUTED = (160, 160, 170)
COL_ACCENT = (80, 190, 120)   # yeşil
COL_ACCENT2 = (255, 170, 70)  # turuncu
COL_WARN = (90, 200, 255)     # mavi

# ================== AUTOCOMPLETE ==================
SUGGEST_K = 3
WORDS_FILE = "sign_language/words_en.txt"  # her satır 1 kelime

BUILTIN_WORDS = [
    "hello","help","home","how","are","you","yes","no","please","thanks","thank","good","bad",
    "morning","night","name","my","your","what","where","when","why","who","sorry",
    "i","we","they","he","she","it","love","like","want","need","go","come",

    "school","work","water","food","today","tomorrow","friend","family","phone","message",
    "computer","project","camera","model","training","dataset","letter","word","translate","translator",
    "open","close","start","stop","again","try","correct","wrong","right","left","up","down",

    "a","an","the","and","or","but","because","to","from","with","without","in","on","at","for",
    "is","am","are","was","were","be","been","have","has","had","do","does","did","can","could","will","would",
    "this","that","these","those","here","there","now","later","soon",

    "listen","speak","talk","write","read","understand","again","slow","fast","repeat",
    "call","text","send","receive","meet","see","look","watch","learn","teach",

    "happy","sad","angry","tired","excited","scared","fine","okay","great",

    "room","library","class","office","house","door","book","pen","paper","table","chair",
    "car","bus","train","street","city"
]

def load_words_upper():
    words = []
    if os.path.exists(WORDS_FILE):
        try:
            with open(WORDS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if not w:
                        continue
                    w2 = "".join([c for c in w if c.isalpha()])
                    if w2:
                        words.append(w2.upper())
        except:
            words = []
    if not words:
        words = [w.upper() for w in BUILTIN_WORDS if w]
    return sorted(list(set(words)))

WORDS = load_words_upper()

def current_prefix_upper(text):
    t = text.rstrip("\n")
    if not t:
        return ""
    i = t.rfind(" ")
    prefix = t[i+1:] if i >= 0 else t
    prefix = prefix.upper()
    if not prefix.isalpha():
        return ""
    return prefix

def get_suggestions_upper(prefix, k=3):
    if not prefix:
        return []
    cand = [w for w in WORDS if w.startswith(prefix)]
    cand.sort(key=lambda w: (len(w), w))
    return cand[:k]

def apply_suggestion_upper(text, suggestion):
    if not suggestion:
        return text
    t = text.rstrip("\n")
    i = t.rfind(" ")
    if i >= 0:
        base = t[:i+1]
        return base + suggestion.upper() + " "
    return suggestion.upper() + " "

# ================== MANUAL ONAY VERİ TOPLAMA (CSV) ==================
LOG_MANUAL_CONFIRMS = True
LOG_PATH = "sign_language/user_samples/user_manual_samples.csv"

def append_manual_sample_csv(x63_norm_row, label):
    """
    x63_norm_row: shape (1, 63)
    label: str (örn 'A')
    CSV format:
      ts,label,f0,f1,...,f62
    """
    if not LOG_MANUAL_CONFIRMS:
        return
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    except:
        pass

    ts = f"{time.time():.3f}"
    feats = x63_norm_row.reshape(-1).tolist()
    line = ts + "," + str(label).upper() + "," + ",".join([f"{v:.6f}" for v in feats]) + "\n"

    need_header = (not os.path.exists(LOG_PATH)) or (os.path.getsize(LOG_PATH) == 0)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            if need_header:
                header = "ts,label," + ",".join([f"f{i}" for i in range(63)]) + "\n"
                f.write(header)
            f.write(line)
    except:
        pass

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

# ================== FEATURES ==================
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
    return [(CLASSES[i], float(proba[i])) for i in idx]

def decide_label_from_proba(proba):
    topk = topk_from_proba(proba, k=max(2, SHOW_TOPK))
    top1_label, top1_p = topk[0]
    top2_p = topk[1][1] if len(topk) > 1 else 0.0

    if (top1_p >= PROBA_THRESHOLD) and ((top1_p - top2_p) >= MARGIN_THRESHOLD):
        return top1_label, topk
    return "Unknown", topk

def majority_vote(history):
    counts = Counter([h for h in history if h != "Unknown"])
    if not counts:
        return None, 0
    return counts.most_common(1)[0]

# ================== UI HELPERS ==================
def put_text(img, text, x, y, scale=0.7, color=COL_TEXT, thickness=2):
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)

def draw_panel(canvas, x, y, w, h, title, accent=None, filled=True):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_PANEL if filled else COL_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)

    put_text(canvas, title.upper(), x + 14, y + 30, scale=0.85, color=COL_TITLE, thickness=2)

    if accent is None:
        accent = COL_ACCENT
    cv2.line(canvas, (x, y + 44), (x + w, y + 44), accent, 3)

def fit_into_box(frame, box_w, box_h):
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        out = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        return out
    scale = min(box_w / w, box_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 0)
    x0 = (box_w - nw) // 2
    y0 = (box_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def wrap_text_by_chars(text, max_chars):
    lines = []
    cur = ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        cur += ch
        if len(cur) >= max_chars:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines

def compute_chars_per_line(panel_w):
    usable = max(10, panel_w - 24)
    return max(12, usable // 14)

def draw_text_output(canvas, x, y, w, h, typed_text, suggestions):
    draw_panel(canvas, x, y, w, h, "TEXT OUTPUT", accent=COL_WARN)

    max_chars = compute_chars_per_line(w)
    lines = wrap_text_by_chars(typed_text, max_chars)

    usable_lines = 8
    if len(lines) > usable_lines:
        lines = lines[-usable_lines:]
        lines[0] = "… " + lines[0]

    start_y = y + 78
    line_gap = 26
    for i, line in enumerate(lines):
        put_text(canvas, line, x + 14, start_y + i * line_gap, scale=0.78, color=COL_TEXT, thickness=2)

    last_line = lines[-1] if lines else ""
    cx = x + 14 + int(14 * len(last_line))
    cy = start_y + (len(lines) - 1) * line_gap
    put_text(canvas, "_", cx, cy + 2, scale=0.85, color=COL_TEXT, thickness=2)

    sug_y = y + h - 44
    put_text(canvas, "SUGGESTIONS:", x + 14, sug_y, scale=0.68, color=COL_MUTED, thickness=2)
    if suggestions:
        s = "   ".join([f"{i+1}) {w}" for i, w in enumerate(suggestions)])
        put_text(canvas, s, x + 14, sug_y + 22, scale=0.72, color=COL_TEXT, thickness=2)
    else:
        put_text(canvas, "N/A", x + 170, sug_y, scale=0.68, color=COL_MUTED, thickness=2)

def draw_status(canvas, x, y, w, h, auto_mode, shown_label, window, every_n, cooldown, votes, min_votes):
    draw_panel(canvas, x, y, w, h, "STATUS", accent=COL_ACCENT2)

    chip_text = "AUTO" if auto_mode else "MANUAL"
    chip_col = COL_ACCENT2 if auto_mode else COL_MUTED
    chip_w = 100 if auto_mode else 120
    chip_h = 28
    chip_x = x + 14
    chip_y = y + 60
    cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), (40, 40, 46), -1)
    cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + chip_w, chip_y + chip_h), chip_col, 2)
    put_text(canvas, f"MODE: {chip_text}", chip_x + 10, chip_y + 20, scale=0.62, color=COL_TITLE, thickness=2)

    put_text(canvas, "PREDICTION", x + 14, y + 120, scale=0.75, color=COL_MUTED, thickness=2)
    pred_col = COL_ACCENT if shown_label != "Unknown" else (120, 120, 130)
    put_text(canvas, shown_label.upper() if shown_label != "Unknown" else "UNKNOWN",
             x + 14, y + 170, scale=1.6, color=pred_col, thickness=3)

    info_x = x + w - 240
    put_text(canvas, f"WINDOW: {window}", info_x, y + 80, scale=0.70, color=COL_TEXT, thickness=2)
    put_text(canvas, f"EVERYN: {every_n}", info_x, y + 110, scale=0.70, color=COL_TEXT, thickness=2)
    put_text(canvas, f"VOTES: {votes}/{min_votes}", info_x, y + 140, scale=0.70, color=COL_TEXT, thickness=2)
    put_text(canvas, f"COOLDOWN: {cooldown}", info_x, y + 170, scale=0.70, color=COL_TEXT, thickness=2)

def draw_topk(canvas, x, y, w, h, topk_list):
    draw_panel(canvas, x, y, w, h, f"TOP-{SHOW_TOPK}", accent=COL_ACCENT)

    if not topk_list:
        put_text(canvas, "N/A", x + 14, y + 88, scale=0.85, color=COL_MUTED, thickness=2)
        return

    start_y = y + 80
    for i, (lab, p) in enumerate(topk_list[:SHOW_TOPK]):
        lab = str(lab).upper()
        bar_x = x + 14
        bar_y = start_y + i * 44
        bar_w = w - 28
        bar_h = 18

        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 52), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, p)))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COL_ACCENT, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COL_BORDER, 1)

        put_text(canvas, f"{lab}", bar_x, bar_y - 6, scale=0.78, color=COL_TEXT, thickness=2)
        put_text(canvas, f"{p:.2f}", x + w - 72, bar_y - 6, scale=0.78, color=COL_TEXT, thickness=2)

def draw_header(canvas, title="ASL Translator (Demo)"):
    cv2.rectangle(canvas, (0, 0), (WIN_W, HEADER_H), (22, 22, 26), -1)
    cv2.line(canvas, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), COL_BORDER, 1)
    put_text(canvas, title.upper(), PAD, 42, scale=1.2, color=COL_TITLE, thickness=3)

def draw_footer(canvas, auto_mode):
    y0 = WIN_H - FOOTER_H
    cv2.rectangle(canvas, (0, y0), (WIN_W, WIN_H), (22, 22, 26), -1)
    cv2.line(canvas, (0, y0), (WIN_W, y0), COL_BORDER, 1)

    put_text(canvas, "TAB: AUTO | ENTER: ADD | SPACE: SPACE | BACKSPACE: DELETE | C: CLEAR | 1-3: SUGGEST | ESC: QUIT",
             PAD, y0 + 44, scale=0.72, color=COL_TEXT, thickness=2)

    tag = "AUTO ON" if auto_mode else "AUTO OFF"
    col = COL_ACCENT2 if auto_mode else COL_MUTED
    box_w = 140
    box_h = 30
    x = WIN_W - PAD - box_w
    y = y0 + 20
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), (40, 40, 46), -1)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), col, 2)
    put_text(canvas, tag, x + 18, y + 22, scale=0.68, color=COL_TITLE, thickness=2)

# ================== MAIN ==================
def main():
    try:
        cv2.setNumThreads(1)
    except:
        pass

    auto_mode = False

    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, CAM_BACKEND)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)

    history = deque(maxlen=WINDOW)
    shown_label = "Unknown"
    last_topk = []
    last_pred = "Unknown"

    frame_id = 0
    typed_text = ""
    cooldown = 0

    stable_run_label = None
    stable_run_count = 0
    last_committed = None
    need_change = False

    last_votes = 0
    no_hand_streak = 0

    # MANUAL veri kaydı için: en son tahminde kullanılan feature
    last_X_norm = None

    while True:
        for _ in range(FLUSH_GRABS_PER_LOOP):
            cap.grab()

        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.rotate(frame, ROTATE_MODE)
        if MIRROR:
            frame = cv2.flip(frame, 1)

        do_predict = (frame_id % PREDICT_EVERY_N_FRAMES == 0)
        frame_id += 1

        hand_present = True

        if do_predict:
            # Tahmin için küçültülmüş frame (UI aynı kalır)
            if PROCESS_SCALE is not None and 0.1 < PROCESS_SCALE < 1.0:
                proc = cv2.resize(frame, None, fx=PROCESS_SCALE, fy=PROCESS_SCALE, interpolation=cv2.INTER_AREA)
            else:
                proc = frame

            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = hand_landmarker.detect(mp_image)

            raw = extract_raw_features(result)
            if raw is None:
                hand_present = False
                last_pred = "Unknown"
                last_topk = []
                last_X_norm = None
            else:
                X = normalize_row(raw)
                last_X_norm = X.copy()
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0]
                    last_pred, last_topk = decide_label_from_proba(proba)
                else:
                    last_pred = model.predict(X)[0]
                    last_topk = []
                    # predict_proba yoksa CSV kaydı yine X ile yapılır

        # ======= UNKNOWN / NO HAND RESET (SADECE EL YOKSA) =======
        # Buradaki en önemli değişiklik:
        # no_hand_streak artık "last_pred == Unknown" iken artmıyor.
        # Sadece el gerçekten yoksa artıyor.
        if not hand_present:
            no_hand_streak += 1
        else:
            no_hand_streak = 0

        if no_hand_streak >= NO_HAND_RESET_FRAMES:
            history.clear()
            shown_label = "Unknown"
            last_topk = []      # TOP-2 => N/A
            last_votes = 0
        else:
            history.append(last_pred)

            voted_label, votes = majority_vote(history)
            last_votes = votes

            if voted_label is not None and votes >= MIN_VOTES:
                shown_label = voted_label
            else:
                if not HOLD_LAST_STABLE:
                    shown_label = "Unknown"

        # cooldown
        if cooldown > 0:
            cooldown -= 1

        # spam kilidi çöz
        if REQUIRE_CHANGE_BEFORE_REPEAT and need_change:
            if shown_label == "Unknown" or (last_committed is not None and shown_label != last_committed):
                need_change = False

        # AUTO write
        if auto_mode and cooldown == 0 and shown_label != "Unknown":
            if stable_run_label == shown_label:
                stable_run_count += 1
            else:
                stable_run_label = shown_label
                stable_run_count = 1

            if stable_run_count >= AUTO_HOLD_FRAMES:
                if not (REQUIRE_CHANGE_BEFORE_REPEAT and last_committed == shown_label and need_change):
                    typed_text += str(shown_label).upper()
                    last_committed = shown_label
                    need_change = True if REQUIRE_CHANGE_BEFORE_REPEAT else False
                    cooldown = COOLDOWN_FRAMES
                stable_run_count = 0
        else:
            stable_run_label = None
            stable_run_count = 0

        # AUTOCOMPLETE
        prefix = current_prefix_upper(typed_text)
        suggestions = get_suggestions_upper(prefix, SUGGEST_K)

        # UI canvas
        ui = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        ui[:] = COL_BG

        draw_header(ui, "ASL Translator (Demo)")
        draw_footer(ui, auto_mode)

        content_x = PAD
        content_y = HEADER_H + PAD

        cam_x = content_x
        cam_y = content_y

        right_x = cam_x + CAM_W + GAP
        right_y = content_y

        draw_panel(ui, cam_x, cam_y, CAM_W, CAM_H, "CAMERA", accent=COL_ACCENT)

        cam_inner_x = cam_x + 14
        cam_inner_y = cam_y + 58
        cam_inner_w = CAM_W - 28
        cam_inner_h = CAM_H - 72

        fitted = fit_into_box(frame, cam_inner_w, cam_inner_h)
        ui[cam_inner_y:cam_inner_y + cam_inner_h, cam_inner_x:cam_inner_x + cam_inner_w] = fitted

        draw_status(ui, right_x, right_y, RIGHT_W, STATUS_H,
                    auto_mode, shown_label, WINDOW, PREDICT_EVERY_N_FRAMES, cooldown, last_votes, MIN_VOTES)

        draw_topk(ui, right_x, right_y + STATUS_H + GAP, RIGHT_W, TOPK_H, last_topk)

        draw_text_output(ui, right_x, right_y + STATUS_H + GAP + TOPK_H + GAP, RIGHT_W, TEXT_H, typed_text, suggestions)

        cv2.imshow("ASL Translator (Demo)", ui)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == 9:  # TAB
            auto_mode = not auto_mode
            cooldown = 0
            stable_run_label = None
            stable_run_count = 0
            need_change = False

        if key == ord('c'):
            typed_text = ""
            last_committed = None
            need_change = False

        # MANUAL add + MANUAL TRAIN LOG
        if key == 13:  # ENTER
            # Sadece MANUAL modda ve geçerli harf varsa
            if (not auto_mode) and (shown_label != "Unknown"):
                typed_text += str(shown_label).upper()
                last_committed = shown_label
                need_change = True if REQUIRE_CHANGE_BEFORE_REPEAT else False
                cooldown = COOLDOWN_FRAMES

                # >>> eğitim verisine ekle (sadece ENTER ile, auto değil)
                if last_X_norm is not None:
                    append_manual_sample_csv(last_X_norm, shown_label)

        if key == 32:  # SPACE
            typed_text += " "
            last_committed = None
            need_change = False
            cooldown = COOLDOWN_FRAMES

        if key == 8:  # BACKSPACE
            typed_text = typed_text[:-1] if len(typed_text) > 0 else typed_text
            need_change = False

        # AUTOCOMPLETE accept: 1/2/3 (BURADA veri toplama yok)
        if key in (ord('1'), ord('2'), ord('3')):
            idx = int(chr(key)) - 1
            if 0 <= idx < len(suggestions):
                typed_text = apply_suggestion_upper(typed_text, suggestions[idx])
                last_committed = None
                need_change = False
                cooldown = COOLDOWN_FRAMES

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
