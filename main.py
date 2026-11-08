# main.py  (검토 반영 수정본)
#
# 가정:
#   - place_estimator.estimate_place(objects) -> (place: str, scores: dict[str, float])
#     장소를 못 정하면 place == "장소를 추정할 수 없습니다."
#   - clip_estimator.predict_space(image_bytes) -> {"space": str, "confidence": float,
#     "margin": float, ...(top3 등)}  형태의 dict
#   - YOLO 쪽 장소 이름과 CLIP 쪽 space 이름이 같은 라벨 체계를 쓴다
#     (다르면 "둘 다 같은 장소인지" 비교가 의미가 없으므로 매핑이 필요합니다)
 
import io
import threading
from collections import Counter
from contextlib import asynccontextmanager
from typing import Optional
 
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from ultralytics import YOLO
 
from clip_estimator import load_model, predict_space
from place_estimator import estimate_place
 
 
# =========================================
# 설정
# =========================================
 
# YOLO 객체 탐지 confidence 기준 (2026.09.30: 0.35 -> 0.5)
CONF_THRES = 0.5
 
# 같은 클래스는 objects 목록에 1개만 (counts는 전체 개수)
MAX_OBJECTS_PER_CLASS = 1
 
# YOLO 공간 점수 확정 기준 (weights.py 가중치 * confidence 합산 기준)
YOLO_PLACE_THRESHOLD = 2.5
YOLO_MARGIN_THRESHOLD = 0.5
 
# CLIP 확정 기준
CLIP_CONFIDENCE_THRESHOLD = 0.45
CLIP_MARGIN_THRESHOLD = 0.08
 
# True : YOLO가 확신해도 CLIP을 항상 돌려서 교차 검증 (느리지만 정확)
# False: YOLO가 확신하면 CLIP을 건너뜀 -> clip_result = None (빠름)
ALWAYS_RUN_CLIP = True
 
YOLO_UNKNOWN_PLACE = "장소를 추정할 수 없습니다."
 
 
# =========================================
# 모델 로드
# =========================================
 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8m.pt")
 
# 같은 GPU 모델을 여러 스레드가 동시에 쓰지 않도록 잠금
inference_lock = threading.Lock()
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=================================")
    print("[ YOLO + CLIP Server Started ]")
    print("GPU Available:", torch.cuda.is_available())
    print("Device:", device)
    # GPU 메모리가 부족하면 아래 줄을 주석 처리 (첫 CLIP 요청 때 로드됨)
    print("main: CLIP 모델 사전 로딩 시작")
    load_model()
    print("main: CLIP 모델 사전 로딩 완료")
    print("=================================")
    yield
 
 
app = FastAPI(lifespan=lifespan)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # "*"와 credentials=True는 같이 쓰지 않는 게 안전
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# =========================================
# 객체 검출
# =========================================
 
def get_detected_objects(frame):
    """
    objects: 객체 상세 목록 (같은 클래스는 최대 MAX_OBJECTS_PER_CLASS개)
    counts : 객체명별 개수 (제한 없이 전체 개수)
    """
    results = model.predict(frame, conf=CONF_THRES, device=device, verbose=False)
    result = results[0]
 
    objects = []
    counts = Counter()
 
    if result.boxes is None:
        return objects, counts
 
    for box in result.boxes:
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
 
        # 전체 개수는 제한 전에 센다
        counts[class_name] += 1
        if counts[class_name] > MAX_OBJECTS_PER_CLASS:
            continue
 
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        objects.append({
            "name": class_name,
            "confidence": round(confidence, 4),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })
 
    return objects, counts
 
 
# =========================================
# 이미지 디코딩
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
    ranked = sorted(scores.values(), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    return best, second, best - second
 
 
def get_top_scores(scores, top_k=3):
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k])
 
 
def safe_predict_space(image_bytes):
    """CLIP 결과가 예상 형태가 아니거나 에러가 나면 None."""
    try:
        r = predict_space(image_bytes)
    except Exception as e:
        print("CLIP 오류:", e)
        return None
    if not isinstance(r, dict) or not {"space", "confidence", "margin"} <= r.keys():
        return None
    return r
 
 
# =========================================
# YOLO + CLIP 하이브리드 공간 판별
# =========================================
 
def _result(place, source, confidence, confidence_type, yolo_result, clip_result, conflict=False):
    return {
        "place": place,
        "source": source,
        "confidence": round(float(confidence), 4),
        # yolo_score는 가중치 합(0~1 아님), clip_prob는 확률(0~1)
        "confidence_type": confidence_type,
        "conflict": conflict,
        "yolo_result": yolo_result,
        "clip_result": clip_result,
    }
 
 
def estimate_place_hybrid(image_bytes, objects):
    # 1) YOLO
    yolo_place, yolo_scores = estimate_place(objects)
    best, second, margin = get_score_info(yolo_scores)
    yolo_result = {
        "place": yolo_place,
        "best_score": round(best, 4),
        "second_score": round(second, 4),
        "margin": round(margin, 4),
        "scores": get_top_scores(yolo_scores),
    }
    yolo_ok = (
        yolo_place != YOLO_UNKNOWN_PLACE
        and best >= YOLO_PLACE_THRESHOLD
        and margin >= YOLO_MARGIN_THRESHOLD
    )
 
    # 2) YOLO가 확신하고 CLIP 생략 모드면 여기서 끝
    if yolo_ok and not ALWAYS_RUN_CLIP:
        return _result(yolo_place, "yolo", best, "yolo_score", yolo_result, None)
 
    # 3) CLIP
    clip_result = safe_predict_space(image_bytes)
    clip_ok = (
        clip_result is not None
        and clip_result["confidence"] >= CLIP_CONFIDENCE_THRESHOLD
        and clip_result["margin"] >= CLIP_MARGIN_THRESHOLD
    )
 
    # 4) 결정
    if yolo_ok and clip_ok:
        if clip_result["space"] == yolo_place:
            return _result(yolo_place, "yolo+clip", clip_result["confidence"],
                           "clip_prob", yolo_result, clip_result)
        # 둘 다 확신하는데 다름 -> CLIP 우선 (원래 동작 유지), 충돌 표시
        return _result(clip_result["space"], "clip", clip_result["confidence"],
                       "clip_prob", yolo_result, clip_result, conflict=True)
 
    if yolo_ok:
        return _result(yolo_place, "yolo", best, "yolo_score", yolo_result, clip_result)
 
    if clip_ok:
        return _result(clip_result["space"], "clip", clip_result["confidence"],
                       "clip_prob", yolo_result, clip_result)
 
    return _result("unknown", "unknown", 0.0, "none", yolo_result, clip_result)
 
 
# =========================================
# 한 장 분석 (동기 함수: 스레드풀에서 실행)
# =========================================
 
def log_result(objects, counts, result):
    """콘솔에 객체명과 장소를 출력합니다."""
    object_text = ", ".join(
        f"{name}({counts[name]})" for name in (o["name"] for o in objects)
    ) or "없음"
 
    clip = result["clip_result"]
    clip_text = (
        f"{clip['space']} ({clip['confidence']:.2f})" if clip else "실행 안 함"
    )
 
    print("---------------------------------")
    print(f"[객체] {object_text}")
    print(f"[YOLO] {result['yolo_result']['place']} "
          f"(점수 {result['yolo_result']['best_score']})")
    print(f"[CLIP] {clip_text}")
    print(f"[최종 장소] {result['place']}  <- {result['source']}"
          + ("  (YOLO/CLIP 불일치)" if result["conflict"] else ""))
    print("---------------------------------")
 
 
def analyze_one(image_bytes):
    image = decode_image(image_bytes)
    if image is None:
        return {"success": False, "message": "이미지를 읽을 수 없습니다."}
 
    with inference_lock:
        objects, counts = get_detected_objects(image)
        result = estimate_place_hybrid(image_bytes, objects)
 
    log_result(objects, counts, result)
 
    h, w = image.shape[:2]
    return {
        "success": True,
        "image_size": {"width": w, "height": h},
        **result,
        "objects": objects,
        "object_counts": dict(counts),
    }
 
 
# =========================================
# 엔드포인트
# =========================================
 
@app.get("/")
def health_check():
    return {
        "success": True,
        "message": "YOLO + CLIP Space Recognition Server Running",
        "gpu": torch.cuda.is_available(),
        "device": str(device),
    }
 
 
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    # 무거운 추론은 이벤트 루프를 막지 않도록 스레드풀에서
    return await run_in_threadpool(analyze_one, image_bytes)
 
 
@app.post("/analyze-multiple")
async def analyze_multiple(
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
    file3: Optional[UploadFile] = File(None),
    file4: Optional[UploadFile] = File(None),
):
    files = [f for f in (file1, file2, file3, file4) if f is not None]
 
    image_results = []
    for index, f in enumerate(files, start=1):
        image_bytes = await f.read()
        r = await run_in_threadpool(analyze_one, image_bytes)
        image_results.append({"image_index": index, **r})
 
    ok_results = [r for r in image_results if r["success"]]
    place_history = [r["place"] for r in ok_results]
    valid_places = [p for p in place_history if p != "unknown"]
 
    if valid_places:
        votes = Counter(valid_places)
        top = votes.most_common()
        # 동표면 confidence 합이 큰 쪽
        tied = [p for p, c in top if c == top[0][1]]
        final_place = max(
            tied,
            key=lambda p: sum(r["confidence"] for r in ok_results
                              if r["place"] == p and r["confidence_type"] == "clip_prob"),
        )
    else:
        final_place = "unknown"
 
    print("=================================")
    print(f"[여러 장 결과] {' / '.join(place_history) or '없음'}"
          f"  (성공 {len(ok_results)}/{len(files)}장)")
    print(f"[최종 장소 (다수결)] {final_place}")
    print("=================================")
 
    return {
        "success": True,
        "place": final_place,
        "image_count": len(files),
        "success_count": len(ok_results),
        "history": place_history,
        "images": image_results,
    }
 