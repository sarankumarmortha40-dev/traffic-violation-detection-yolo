from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# =========================
# 🔥 LOAD MODELS
# =========================
helmet_model = YOLO(r"C:\traffic violation project\helmet model\detect\train3\weights\best.pt")
triple_model = YOLO(r"C:\traffic violation\train-4\weights\best.pt")

cap = cv2.VideoCapture(r"C:\Users\Saran Kumar Mortha\Downloads\WhatsApp Video 2026-06-12 at 12.47.04 PM.mp4")

# =========================
# 🔧 SETTINGS (tune if needed)
# =========================
CONF_HELMET = 0.30
CONF_TRIPLE = 0.45          # ↑ reduces false positives
ASPECT_MAX = 1.2            # h/w > this → likely walking person
MIN_SIZE = 80               # ignore tiny boxes
NMS_DIST = 60               # merge nearby duplicate boxes
STABLE_FRAMES = 3           # temporal smoothing

triple_history = deque(maxlen=5)

# =========================
# 🔧 HELPER: remove duplicates (simple NMS by distance)
# =========================
def filter_close_boxes(boxes, dist_thresh=60):
    kept = []
    for (x1,y1,x2,y2,conf) in boxes:
        cx, cy = (x1+x2)//2, (y1+y2)//2
        good = True
        for (kx1,ky1,kx2,ky2,_) in kept:
            kcx, kcy = (kx1+kx2)//2, (ky1+ky2)//2
            if abs(cx-kcx) < dist_thresh and abs(cy-kcy) < dist_thresh:
                good = False
                break
        if good:
            kept.append((x1,y1,x2,y2,conf))
    return kept

# =========================
# 🎬 LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    helmet_results = helmet_model(frame, conf=CONF_HELMET)[0]
    triple_results = triple_model(frame, conf=CONF_TRIPLE)[0]

    no_helmet_count = 0
    triple_boxes = []

    # =========================
    # 🚨 NO HELMET
    # =========================
    for box in helmet_results.boxes:
        cls = int(box.cls[0])
        label = helmet_model.names[cls].lower()

        if "no" in label:
            no_helmet_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(frame, f"NO HELMET {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,255), 2)

    # =========================
    # 🚨 TRIPLE RIDING (FILTERED)
    # =========================
    for box in triple_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        w = x2 - x1
        h = y2 - y1

        if w < MIN_SIZE or h < MIN_SIZE:
            continue

        ratio = h / w if w > 0 else 0

        # ❌ remove walking persons (tall boxes)
        if ratio > ASPECT_MAX:
            continue

        triple_boxes.append((x1,y1,x2,y2,conf))

    # 🔥 remove duplicates
    triple_boxes = filter_close_boxes(triple_boxes, NMS_DIST)

    # 🔥 temporal stability
    triple_history.append(len(triple_boxes))
    stable = sum(1 for v in triple_history if v > 0) >= STABLE_FRAMES

    triple_count = len(triple_boxes) if stable else 0

    # =========================
    # 🎯 DRAW TRIPLE BOXES
    # =========================
    if stable:
        for (x1,y1,x2,y2,conf) in triple_boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(frame, f"TRIPLE RIDING {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,0,0), 2)

    # =========================
    # 🖥️ PROFESSIONAL UI PANEL
    # =========================
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (420,90), (0,0,0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

    cv2.putText(frame, "Traffic Violation Detection",
                (10,25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    cv2.putText(frame, f"No Helmet: {no_helmet_count}",
                (10,55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,255), 2)

    cv2.putText(frame, f"Triple Riding: {triple_count}",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,0,0), 2)

    # =========================
    # SHOW
    # =========================
    cv2.imshow("Final Presentation 🚨", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()