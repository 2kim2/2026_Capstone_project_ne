# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware


from ultralytics import YOLO
from collections import Counter
from typing import Optional

import torch
import cv2
import numpy as np
import tempfile
import os
import io 
from PIL import Image, ImageOps

from place_estimator import estimate_place
from clip_estimator import load_model, predict_space


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

# YOLO 객체 탐지 confidence 기준
# 2026.09.30 정확도 변경 : 0.35 -> 0.5
CONF_THRES = 0.5

# 영상에서 몇 프레임마다 분석할지
# VIDEO_FRAME_INTERVAL = 30

# -----------------------------------------
# YOLO 공간 점수 확정 기준
# -----------------------------------------
# place_estimator.py의
# weight * object_confidence 합산 점수 기준입니다.
# weights.py의 가중치에 따라 조정해야 합니다.

YOLO_PLACE_THRESHOLD = 2.5

# YOLO 1위 공간 점수와 2위 공간 점수의 최소 차이
YOLO_MARGIN_THRESHOLD = 0.5

# -----------------------------------------
# CLIP 공간 점수 확정 기준
# -----------------------------------------

# CLIP 상위 1위 confidence 최소값
CLIP_CONFIDENCE_THRESHOLD = 0.45

# CLIP 1위와 2위 confidence 최소 차이
CLIP_MARGIN_THRESHOLD = 0.08


# =========================================
# 모델 로드
# =========================================

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# YOLO 모델 로드
model = YOLO("yolov8m.pt")

print("=================================")
print("[ YOLO Server Started ]")
print("GPU Available:", torch.cuda.is_available())
print("Device:", device)
print("=================================")


# =========================================
# 선택 사항: 서버 시작 시 CLIP 사전 로드
# =========================================
# 첫 CLIP 요청에서 모델을 로드하면 처음 한 번 응답이 늦어질 수 있습니다.
# 서버 시작 때 미리 CLIP을 GPU/CPU 메모리에 올리려면 사용합니다.
#
# GPU 메모리가 부족하면 이 함수 전체를 주석 처리하세요.
# 그러면 YOLO가 애매한 이미지가 처음 들어왔을 때만 CLIP이 로드됩니다.
# =========================================

@app.on_event("startup")
def startup_event():
    print("=================================")
    print("main: CLIP 모델 사전 로딩 시작")

    load_model()

    print("main: CLIP 모델 사전 로딩 완료")
    print("=================================")


# =========================================
# 객체 검출
# =========================================

def get_detected_objects(frame):
# """
#     objects: 객체 상세 목록 (동일 클래스의 중복은 최대 2개까지만 출력)
#     counts: 객체명별 개수 (전체 개수 유지)
# """

    results = model.predict(
        frame,
        conf=CONF_THRES,
        device=device,
        verbose=False
    )

    objects = []
    counts = Counter()
# 동일 클래스의 출력 개수를 제한하기 위한 임시 카운트
    included_per_class = Counter()

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

        # 같은 클래스의 출력이 2개 이상이면 추가하지 않음
            if included_per_class[class_name] >= 2:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            counts[class_name] += 1
            included_per_class[class_name] += 1

            objects.append({
                "name": class_name,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    return objects, counts


# =========================================
# 이미지 bytes를 OpenCV 이미지로 변환
# =========================================

def decode_image(image_bytes):
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# =========================================
# 점수 관련 함수
# =========================================

def get_score_info(scores):

    if not scores:
        return 0.0, 0.0, 0.0

    sorted_scores = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    # 1위 점수
    best_score = sorted_scores[0][1]
    
    # 2위 점수
    second_score = (
        sorted_scores[1][1]
        if len(sorted_scores) > 1
        else 0.0
    )
    # 1위와 2위의 점수 차이
    margin = best_score - second_score

    return best_score, second_score, margin


# 점수 상위 top_k(3)개만 dictionary로 반환합니다.
def get_top_scores(scores, top_k=3):

    return dict(
        sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:top_k]
    )


# =========================================
# YOLO + CLIP 하이브리드 공간 판별
# =========================================
def estimate_place_hybrid(image_bytes, objects):
    # 1) YOLO 결과
    yolo_place, yolo_scores = estimate_place(objects)
    best_yolo_score, second_yolo_score, yolo_margin = get_score_info(yolo_scores)
    top_yolo_scores = get_top_scores(yolo_scores, top_k=3)

    # 2) CLIP 결과 (YOLO 확신 여부와 무관하게 항상 계산)
    clip_result = predict_space(image_bytes)
    clip_is_confident = (
        clip_result["confidence"] >= CLIP_CONFIDENCE_THRESHOLD and
        clip_result["margin"] >= CLIP_MARGIN_THRESHOLD
    )

    yolo_is_confident = (
        yolo_place != "장소를 추정할 수 없습니다."
        and best_yolo_score >= YOLO_PLACE_THRESHOLD
        and yolo_margin >= YOLO_MARGIN_THRESHOLD
    )

    # 3) 최종 결정 로직 (앙상블)
    if yolo_is_confident and not clip_is_confident:
        # YOLO가 확신하지만 CLIP이 불확실하면 YOLO를 우선
        return {
            "place": yolo_place,
            "source": "yolo",
            "confidence": round(best_yolo_score, 4),
            "yolo_result": {
                "place": yolo_place,
                "best_score": round(best_yolo_score, 4),
                "second_score": round(second_yolo_score, 4),
                "margin": round(yolo_margin, 4),
                "scores": top_yolo_scores
            },
            "clip_result": clip_result
        }

    if clip_is_confident:
        # CLIP이 확신하면 CLIP 결과를 우선
        return {
            "place": clip_result["space"],
            "source": "clip",
            "confidence": clip_result["confidence"],
            "yolo_result": {
                "place": yolo_place,
                "best_score": round(best_yolo_score, 4),
                "second_score": round(second_yolo_score, 4),
                "margin": round(yolo_margin, 4),
                "scores": top_yolo_scores
            },
            "clip_result": clip_result
        }

    # 둘 다 불확실하면 unknown
    return {
        "place": "unknown",
        "source": "unknown",
        "confidence": 0.0,
        "yolo_result": {
            "place": yolo_place,
            "best_score": round(best_yolo_score, 4),
            "second_score": round(second_yolo_score, 4),
            "margin": round(yolo_margin, 4),
            "scores": top_yolo_scores
        },
        "clip_result": clip_result
    }

# =========================================
# 서버 상태 확인
# =========================================

@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "YOLO + CLIP Space Recognition Server Running",
        "gpu": torch.cuda.is_available(),
        "device": str(device)
    }


# =========================================
# 이미지 분석
# =========================================

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):
    """
    이미지 1장을 YOLO + CLIP 하이브리드 방식으로 분석합니다.
    """

    image_bytes = await file.read()

    image = decode_image(image_bytes)

    if image is None:
        return {
            "success": False,
            "message": "이미지를 읽을 수 없습니다."
        }

    objects, counts = get_detected_objects(image)

    result = estimate_place_hybrid(
        image_bytes=image_bytes,
        objects=objects
    )
    h, w = image.shape[:2]

    return {
        "success": True,
        "image_size": {"width": w, "height": h},

        # 최종 공간 결과
        "place": result["place"],
        "source": result["source"],
        "confidence": result["confidence"],

        # YOLO 객체 검출 결과
        "objects": objects,
        "object_counts": dict(counts),

        # YOLO 가중치 기반 공간 점수 결과
        "yolo_result": result["yolo_result"],

        # YOLO가 확정했으면 None
        # CLIP을 실행했으면 CLIP 상위 3개 점수 포함
        "clip_result": result["clip_result"]
    }



# =========================================
# 이미지 여러 장 분석
# =========================================

@app.post("/analyze-multiple")
async def analyze_multiple(
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
    file4: Optional[UploadFile] = File(None)
):
    """
    최대 4장의 이미지를 각각 하이브리드 방식으로 분석합니다.
    최종 장소는 unknown을 제외한 결과의 다수결로 결정합니다.
    """

    files = [
        file1,
        file2,
        file3,
        file4
    ]

    # 실제 업로드된 파일만 사용
    files = [
        file
        for file in files
        if file is not None
    ]

    image_results = []
    place_history = []

    for index, image_file in enumerate(files):
        image_bytes = await image_file.read()

        image = decode_image(image_bytes)

        if image is None:
            image_results.append({
                "image_index": index + 1,
                "success": False,
                "message": "이미지를 읽을 수 없습니다."
            })
            continue

        objects, counts = get_detected_objects(image)

        result = estimate_place_hybrid(
            image_bytes=image_bytes,
            objects=objects
        )

        place_history.append(result["place"])
        
        h, w = image.shape[:2]

        image_results.append({
            "image_index": index + 1,
            "image_size": {"width": w, "height": h},
            "success": True,
            "place": result["place"],
            "source": result["source"],
            "confidence": result["confidence"],
            "objects": objects,
            "object_counts": dict(counts),
            "yolo_result": result["yolo_result"],
            "clip_result": result["clip_result"]
        })

    # unknown 결과는 최종 다수결에서 제외
    valid_places = [
        place
        for place in place_history
        if place != "unknown"
    ]

    if valid_places:
        final_place = Counter(
            valid_places
        ).most_common(1)[0][0]
    else:
        final_place = "unknown"

    return {
        "success": True,
        "place": final_place,
        "image_count": len(files),
        "history": place_history,
        "images": image_results
    }

