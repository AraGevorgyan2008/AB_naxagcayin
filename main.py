import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────
# GTSRB — 43 դասերի անունները
# ─────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Արագության սահմանափակում (20)",
    1:  "Արագության սահմանափակում (30)",
    2:  "Արագության սահմանափակում (50)",
    3:  "Արագության սահմանափակում (60)",
    4:  "Արագության սահմանափակում (70)",
    5:  "Արագության սահմանափակում (80)",
    6:  "Արագության սահմանափակում 80 — վերջ",
    7:  "Արագության սահմանափակում (100)",
    8:  "Արագության սահմանափակում (120)",
    9:  "Անցնելն արգելված",
    10: "Բեռնատարների անցնելն արգելված",
    11: "Առաջնահերթ խաչմերուկ",
    12: "Առաջնահերթ ճանապարհ",
    13: "Ճանապարհ տուր",
    14: "Կանգ առ",
    15: "Երթևեկությունն արգելված",
    16: "Բեռնատարների երթևեկն արգելված",
    17: "Ներս մուտք արգելված",
    18: "Ուշադրություն",
    19: "Ձախ կոր",
    20: "Աջ կոր",
    21: "Կրկնակի կոր",
    22: "Ճանապարհի ելուստ",
    23: "Սայթաքուն ճանապարհ",
    24: "Նեղացող ճանապարհ (աջ)",
    25: "Ճանապարհային աշխատանքներ",
    26: "Լուսացույց",
    27: "Հետիոտն",
    28: "Երեխաների անցում",
    29: "Հեծանվորդ",
    30: "Սառույց / ձյուն",
    31: "Կենդանի",
    32: "Սահմանափակումներ — վերջ",
    33: "Աջ պարտադիր",
    34: "Ձախ պարտադիր",
    35: "Ուղիղ պարտադիր",
    36: "Ուղիղ կամ աջ",
    37: "Ուղիղ կամ ձախ",
    38: "Աջ կողմից շրջանցիր",
    39: "Ձախ կողմից շրջանցիր",
    40: "Կլոր երթևեկություն",
    41: "Անցնելն — վերջ",
    42: "Բեռնատարների անցնելն — վերջ",
}

# ─────────────────────────────────────────────
# Կարգավորումներ — փոխիր ըստ քո ֆայլի
# ─────────────────────────────────────────────
MODEL_PATH = "my_model1.keras"
IMG_SIZE   = (64, 64)
CONFIDENCE_THRESHOLD = 0.70

# ─────────────────────────────────────────────
# Font — Unicode աջակցությամբ
# ─────────────────────────────────────────────
def load_font(size=22):
    font_paths = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()

FONT_LARGE = load_font(24)
FONT_SMALL = load_font(18)

# ─────────────────────────────────────────────
# Տեքստ գրել հայերեն (Pillow-ով)
# ─────────────────────────────────────────────
def put_armenian_text(frame, text, pos, font, color_rgb=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
# Մոդել բեռնել
# ─────────────────────────────────────────────
print("Մոդելը բեռնվում է...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Մոդելը հաջողությամբ բեռնվեց։")
except Exception as e:
    print(f"Սխալ մոդելը բեռնելիս: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────
# Կամեռա
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Կամեռան չի բացվում։")
    sys.exit(1)

print("Կամեռան աշխատում է — ելնելու համար սեղմիր Q կամ ESC")

def preprocess(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def draw_ui(frame, label, confidence, color_rgb):
    h, w = frame.shape[:2]
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB → BGR OpenCV-ի համար

    # ── Ստորին մուգ վահանակ ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Confidence բար ──
    bar_w = int((w - 40) * confidence)
    cv2.rectangle(frame, (20, h - 22), (w - 20, h - 7), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, h - 22), (20 + bar_w, h - 7), color_bgr, -1)

    # ── Հայերեն label ──
    frame = put_armenian_text(frame, label,
                               (15, h - 80), FONT_LARGE, (255, 255, 255))
    frame = put_armenian_text(frame, f"{confidence*100:.1f}%",
                               (15, h - 50), FONT_SMALL, color_rgb)

    # ── ROI շրջանակ կենտրոնում ──
    cx, cy = w // 2, h // 2
    size   = min(w, h) // 3
    cv2.rectangle(frame,
                  (cx - size, cy - size),
                  (cx + size, cy + size),
                  color_bgr, 3)

    # ── Վերի հուշ ──
    frame = put_armenian_text(frame,
                               "Ցույց տուր նշանը շրջանակի մեջ",
                               (cx - size, cy - size - 32),
                               FONT_SMALL, (0, 230, 80))
    return frame

# ─────────────────────────────────────────────
# Գլխավոր loop
# ─────────────────────────────────────────────
frame_count   = 0
PREDICT_EVERY = 5  # Ամեն 5 ֆրեյմին մեկ prediction

current_label = "Ցույց տուր նշան..."
current_conf  = 0.0
current_color = (180, 180, 180)  # RGB

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PREDICT_EVERY == 0:
        preds      = model.predict(preprocess(frame), verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id])

        if confidence >= CONFIDENCE_THRESHOLD:
            current_label = CLASS_NAMES.get(class_id, f"Դաս {class_id}")
            current_color = (0, 210, 80)     # Կանաչ (RGB)
        else:
            current_label = "Անհայտ — ցածր վստահություն"
            current_color = (255, 140, 0)    # Նարնջական (RGB)

        current_conf = confidence

    frame = draw_ui(frame, current_label, current_conf, current_color)
    cv2.imshow("Ճանապարհային նշանի ճանաչում", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("Ծրագիրն ավարտվեց։")