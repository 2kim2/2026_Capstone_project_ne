from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from ultralytics import YOLO
from collections import Counter

import torch
import cv2
import numpy as np
import tempfile
import os

from place_estimator import estimate_place

# =========================================
# FastAPI
# =========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# 설정
# =========================================

CONF_THRES = 0.4
VIDEO_FRAME_INTERVAL = 30

# =========================================
# 모델 로드
# =========================================

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8m.pt")

print("=================================")
print("YOLO Server Started")
print("GPU Available:", torch.cuda.is_available())
print("Device:", device)
print("=================================")

# =========================================
# 객체 검출
# =========================================

def get_detected_counts(frame):

    results = model.predict(
        frame,
        conf=CONF_THRES,
        device=device,
        verbose=False
    )

    counts = Counter()

    r = results[0]

    if r.boxes is not None:

        for box in r.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            counts[class_name] += 1

    return counts

# =========================================
# 서버 상태 확인
# =========================================

@app.get("/")
def health_check():

    return {
        "success": True,
        "message": "YOLO Server Running",
        "gpu": torch.cuda.is_available(),
        "device": str(device)
    }

# =========================================
# 이미지 분석
# =========================================

@app.post("/detect-image")
async def detect_image(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return {
            "success": False,
            "message": "이미지를 읽을 수 없습니다."
        }

    counts = get_detected_counts(image)

    place, scores = estimate_place(counts)

    return {
        "success": True,
        "place": place,
        "objects": dict(counts),
        "scores": scores
    }

# =========================================
# 영상 분석
# =========================================

@app.post("/detect-video")
async def detect_video(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(await file.read())

        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    history = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # 30프레임마다 분석
        if frame_idx % VIDEO_FRAME_INTERVAL != 0:
            continue

        counts = get_detected_counts(frame)

        place, _ = estimate_place(counts)

        history.append(place)

    cap.release()

    os.remove(video_path)

    if history:

        final_place = Counter(history).most_common(1)[0][0]

    else:

        final_place = "unknown"

    return {
        "success": True,
        "place": final_place,
        "history": history
    }

# =========================================
# 이미지 + 객체 + 장소 한번에 조회
# =========================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    np_arr = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        np_arr,
        cv2.IMREAD_COLOR
    )

    if image is None:

        return {
            "success": False,
            "message": "이미지 읽기 실패"
        }

    counts = get_detected_counts(image)

    place, scores = estimate_place(counts)

    detected_objects = []

    for obj_name, count in counts.items():

        detected_objects.append(
            {
                "name": obj_name,
                "count": count
            }
        )

    return {
        "success": True,
        "place": place,
        "objects": detected_objects,
        "scores": scores
    }