import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
import cv2
import streamlit as st
import tempfile
import time

# =========================
# 🔥 LOAD MODELS
# =========================
helmet_model = YOLO(r"C:\traffic violation project\helmet model\detect\train3\weights\best.pt")
triple_model = YOLO(r"C:\traffic violation project\triple riding model 2\weights\best.pt")

# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="Traffic Violation Detection", layout="wide")

st.title("🚨 Traffic Violation Detection System")
st.markdown("### Helmet & Triple Riding Detection (Tracked)")

uploaded_file = st.file_uploader("📤 Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:

    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    # 🔥 FULL WIDTH (FIXED)
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    progress = st.progress(0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    prev_time = time.time()
    current_frame = 0

    helmet_ids = set()
    triple_ids = set()

    # =========================
    # 🔁 MAIN LOOP
    # =========================
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # =========================
        # 🚨 HELMET TRACKING
        # =========================
        helmet_results = helmet_model.track(frame, conf=0.4, persist=True)[0]

        if helmet_results.boxes is not None:
            for box in helmet_results.boxes:
                cls = int(box.cls[0])
                label = helmet_model.names[cls].lower()

                if "no" in label:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])

                    track_id = int(box.id[0]) if box.id is not None else None
                    if track_id is not None:
                        helmet_ids.add(track_id)

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.putText(frame,"NO HELMET",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        # =========================
        # 🚨 TRIPLE RIDING TRACKING
        # =========================
        triple_results = triple_model.track(frame, conf=0.5, persist=True)[0]

        if triple_results.boxes is not None:
            for box in triple_results.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1

                if w < 80 or h < 80:
                    continue

                ratio = h / w if w > 0 else 0
                if ratio > 1.2:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    triple_ids.add(track_id)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame,"TRIPLE RIDING",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

        # =========================
        # 🔥 OVERLAY COUNTS ON VIDEO (FIXED)
        # =========================
        cv2.rectangle(frame, (10, 50), (360, 120), (0, 0, 0), -1)

        cv2.putText(frame, f"No Helmet: {len(helmet_ids)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Triple Riding: {len(triple_ids)}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # =========================
        # ⚡ FPS
        # =========================
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        # =========================
        # 📊 SIDE STATS
        # =========================
        stats_placeholder.markdown(f"""
        ## 📊 Live Stats  
        🔴 No Helmet: **{len(helmet_ids)}**  
        🔵 Triple Riding: **{len(triple_ids)}**
        """)

        # =========================
        # 🎥 DISPLAY (FULL WIDTH FIXED)
        # =========================
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, use_container_width=True)

        # =========================
        # 💾 SAVE
        # =========================
        out.write(frame)

        # =========================
        # 📈 PROGRESS
        # =========================
        progress.progress(min(current_frame / frame_count, 1.0))

    cap.release()
    out.release()

    # =========================
    # 📥 DOWNLOAD FIX
    # =========================
    time.sleep(1)

    st.success("✅ Processing Complete!")

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.download_button(
        label="📥 Download Processed Video",
        data=video_bytes,
        file_name="traffic_output.mp4",
        mime="video/mp4"
    )