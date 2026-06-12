from ultralytics import YOLO
import cv2

# 🔥 Load your trained helmet model
model = YOLO(r"C:\traffic violation project\runs\detect\train3\weights\best.pt")

# 🔥 Print class names
print("Class names:", model.names)

# 🔥 Open video
cap = cv2.VideoCapture(r"C:\traffic violation\testing videos\Video Project 1.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 🔥 Run detection
    results = model(frame, conf=0.20)[0]

    # Counters
    helmet_count = 0
    no_helmet_count = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = model.names[cls].lower()

        # 🔴 No Helmet
        if "no" in label:
            color = (0, 0, 255)
            text = f"NO HELMET {conf:.2f}"
            no_helmet_count += 1

        # 🟢 Helmet
        else:
            color = (0, 255, 0)
            text = f"HELMET {conf:.2f}"
            helmet_count += 1

        # Draw bounding box
        cv2.rectangle(frame,
                      (x1, y1),
                      (x2, y2),
                      color,
                      2)

        cv2.putText(frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

    # ==========================
    # 📊 COUNTER PANEL
    # ==========================
    cv2.rectangle(frame, (10, 10), (330, 100), (0, 0, 0), -1)

    cv2.putText(frame,
                f"Helmet: {helmet_count:02d}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"No Helmet: {no_helmet_count:02d}",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2)

    # ==========================
    # 🎥 SHOW OUTPUT
    # ==========================
    cv2.imshow("Helmet Detection Dashboard", frame)

    # Press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
