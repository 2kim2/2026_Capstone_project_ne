```python
# 전체 백엔드에서 좌표(box) 및 이미지 크기(image_size) 관련 API 발췌
# /detect: YOLO box 좌표 직접 추출
# /detect-space: 비전 서버 분석 결과 전달
# /detect-space-4: 비전 서버 images[i].image_size 및 objects[].box 전달
# 주의: 원본의 app, AI_BASE, YOLO_ANALYZE_URL, SessionLocal 등 공통 정의가 필요합니다.

@app.post("/detect")
async def detect(
    file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        return {
            "success": False,
            "message": "업로드된 이미지가 비어 있습니다.",
        }
    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )
    if img is None:
        return {
            "success": False,
            "message": "이미지를 읽을 수 없습니다.",
        }
    image_height, image_width = img.shape[:2]
    results = yolo_model(img)
    objects = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            # YOLO bounding box 좌표 (원본 이미지 기준)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append(
                {
                    "name": yolo_model.names[cls_id],
                    "confidence": confidence,
                    "box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                    },
                }
            )
    if not objects:
        return {
            "success": False,
            "message": "객체를 감지하지 못했습니다.",
            "objects": [],
            "image_size": {
                "width": image_width,
                "height": image_height,
            },
        }
    objects.sort(key=lambda item: item["confidence"], reverse=True)
    main_object = objects[0]["name"]
    try:
        response = requests.post(
            f"{AI_BASE}/generate",
            json={"object": main_object},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=60,
        )
        print("AI URL:", f"{AI_BASE}/generate")
        print("AI status:", response.status_code)
        print("AI content-type:", response.headers.get("content-type"))
        print("AI response:", response.text[:500])
        if not response.ok:
            return {
                "success": False,
                "object": main_object,
                "objects": objects,
                "message": "AI 서버 요청에 실패했습니다.",
                "ai_status": response.status_code,
                "ai_response": response.text[:500],
            }
        try:
            ai_result = response.json()
            print("AI JSON:", ai_result)
        except requests.exceptions.JSONDecodeError:
            return {
                "success": False,
                "object": main_object,
                "objects": objects,
                "message": "AI 서버가 JSON이 아닌 응답을 반환했습니다.",
                "ai_status": response.status_code,
                "ai_response": response.text[:500],
            }
        return {
            "success": True,
            "object": main_object,
            "objects": objects,
            "image_size": {
                "width": image_width,
                "height": image_height,
            },
            "ai": ai_result,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "object": main_object,
            "objects": objects,
            "message": "AI 서버 응답 시간이 초과되었습니다.",
        }
    except requests.exceptions.ConnectionError as error:
        return {
            "success": False,
            "object": main_object,
            "objects": objects,
            "message": "AI 서버에 연결할 수 없습니다.",
            "error": str(error),
        }
    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "object": main_object,
            "objects": objects,
            "message": "AI 서버 요청 중 오류가 발생했습니다.",
            "error": str(error),
        }

@app.post("/detect-space")
async def detect_space(
    file: UploadFile = File(...),
    user_id: int = Form(...),
):
    image_bytes = await file.read()
    if not image_bytes:
        return {
            "success": False,
            "message": "업로드된 이미지가 비어 있습니다.",
        }
    # 프론트에서 bounding box 좌표를 화면 좌표로 변환할 때 사용
    original_img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )
    if original_img is None:
        return {
            "success": False,
            "message": "이미지를 읽을 수 없습니다.",
        }
    image_height, image_width = original_img.shape[:2]
    try:
        files = {
            "file": (
                file.filename or "image.jpg",
                image_bytes,
                file.content_type or "image/jpeg",
            )
        }
        response = requests.post(
                YOLO_ANALYZE_URL,
            files=files,
            headers={
                "ngrok-skip-browser-warning": "true"
            },
            timeout=60,
        )
        print("YOLO 서버 상태:", response.status_code)
        print("YOLO 서버 응답:", response.text[:1000])
        if not response.ok:
            return {
                "success": False,
                "message": "YOLO 서버 요청에 실패했습니다.",
                "status_code": response.status_code,
                "ai_response": response.text[:500],
            }
        yolo_result = response.json()
        if not yolo_result.get("success"):
            return {
                "success": False,
                "message": "YOLO 분석에 실패했습니다.",
                "yolo": yolo_result,
            }
        study_result = complete_study(
            StudyCompleteRequest(
                user_id=user_id,
                word_count=1,
            )
        )
        if not study_result.get("success"):
            return {
                "success": False,
                "place": yolo_result.get("place"),
                "objects": yolo_result.get("objects", []),
                "scores": yolo_result.get("scores", {}),
                "image_size": {
                    "width": image_width,
                    "height": image_height,
                },
                "message": study_result.get(
                    "message",
                    "학습 진도 저장에 실패했습니다.",
                ),
                "study": study_result,
            }
        place = normalize_place(yolo_result.get("place"))
        if place == "café":
            place = "cafe"
        objects = object_names(yolo_result.get("objects", []))
        context_id = save_detected_context(user_id, place, objects)
        return {
            "success": True,
            "contextId": context_id,
            "topics": topics_for(place),
            "space": place,
            "place": place,
            "objects": objects,
            "scores": yolo_result.get("scores", {}),
            "image_size": {
                "width": image_width,
                "height": image_height,
            },
            "study": study_result,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "YOLO 서버 응답 시간이 초과되었습니다.",
        }
    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "message": "YOLO 서버에 연결할 수 없습니다.",
            "error": str(error),
        }
    except Exception as error:
        print("공간/사물 분석 오류:", error)
        return {
            "success": False,
            "message": f"공간/사물 분석 중 오류가 발생했습니다: {error}",
        }
# =========================================================
# 4방향 이미지 공간 + 사물 인식 API
# front / back / left / right 이미지를 한 번에 받음
# 각 이미지는 기존 YOLO /analyze 서버로 따로 전송
# 학습 진도는 4장이 성공해도 1회만 증가
# =========================================================

@app.post("/detect-space-4")
async def detect_space_4(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    user_id: int = Form(...),
):
    uploads = {
        "front": front,
        "back": back,
        "left": left,
        "right": right,
    }
    view_results = {}
    try:
        for view_name, upload in uploads.items():
            image_bytes = await upload.read()
            if not image_bytes:
                return {
                    "success": False,
                    "message": f"{view_name} 이미지가 비어 있습니다.",
                }
            # 원본 이미지 크기 확인
            original_img = cv2.imdecode(
                np.frombuffer(image_bytes, np.uint8),
                cv2.IMREAD_COLOR,
            )
            if original_img is None:
                return {
                    "success": False,
                    "message": f"{view_name} 이미지를 읽을 수 없습니다.",
                }
            image_height, image_width = original_img.shape[:2]
            # =====================================================
            # 테스트 모드: YOLO 서버 건너뛰기
            # =====================================================
            if SKIP_YOLO:
                print(f"[{view_name}] YOLO 서버 우회 테스트")
                fake_objects = {
                    "front": [
                        {"name": "table", "confidence": 0.95},
                        {"name": "cup", "confidence": 0.91},
                    ],
                    "back": [
                        {"name": "chair", "confidence": 0.93},
                    ],
                    "left": [
                        {"name": "menu", "confidence": 0.90},
                    ],
                    "right": [
                        {"name": "bottle", "confidence": 0.89},
                    ],
                }
                view_results[view_name] = {
                    "space": "cafe",
                    "place": "cafe",
                    "objects": fake_objects[view_name],
                    "scores": {},
                    "image_size": {
                        "width": image_width,
                        "height": image_height,
                    },
                }
                continue
            # =====================================================
            # 실제 YOLO 서버 사용
            # =====================================================
            files = {
                "file": (
                    upload.filename or f"{view_name}.jpg",
                    image_bytes,
                    upload.content_type or "image/jpeg",
                )
            }
            response = requests.post(
                YOLO_ANALYZE_URL,
                files=files,
                headers={
                    "ngrok-skip-browser-warning": "true",
                },
                timeout=60,
            )
            print(
                f"[{view_name}] YOLO 서버 상태:",
                response.status_code,
            )
            print(
                f"[{view_name}] YOLO 서버 응답:",
                response.text[:1000],
            )
            if not response.ok:
                return {
                    "success": False,
                    "failed_view": view_name,
                    "message": f"{view_name} 이미지 YOLO 서버 요청에 실패했습니다.",
                    "status_code": response.status_code,
                    "ai_response": response.text[:500],
                }
            try:
                yolo_result = response.json()
            except requests.exceptions.JSONDecodeError:
                return {
                    "success": False,
                    "failed_view": view_name,
                    "message": f"{view_name} 이미지 YOLO 서버가 JSON이 아닌 응답을 반환했습니다.",
                    "ai_response": response.text[:500],
                }
            if not yolo_result.get("success"):
                return {
                    "success": False,
                    "failed_view": view_name,
                    "message": f"{view_name} 이미지 YOLO 분석에 실패했습니다.",
                    "yolo": yolo_result,
                }
            # 비전 서버가 반환한 이미지별 원본 메타데이터를 그대로 사용합니다.
            images = yolo_result.get("images")
            if isinstance(images, list) and images:
                vision_image = images[0]  # 이 엔드포인트는 방향별로 이미지 1장씩 요청
                if not isinstance(vision_image, dict):
                    return {"success": False, "failed_view": view_name,
                            "message": "비전 서버 images[0] 응답 형식이 올바르지 않습니다."}
            else:
                # 단일 이미지 응답 형식과의 호환성
                vision_image = yolo_result

            vision_size = vision_image.get("image_size")
            if not isinstance(vision_size, dict) or not all(
                key in vision_size for key in ("width", "height")
            ):
                return {"success": False, "failed_view": view_name,
                        "message": "비전 서버 응답에 image_size.width/height가 없습니다."}

            vision_objects = vision_image.get("objects", [])
            if not isinstance(vision_objects, list):
                return {"success": False, "failed_view": view_name,
                        "message": "비전 서버 objects 응답이 배열이 아닙니다."}
            # box 포함 객체 전체를 전달합니다. 없는 box를 임의로 만들지 않습니다.
            view_results[view_name] = {
                "space": vision_image.get("place", yolo_result.get("place")),
                "place": normalize_place(vision_image.get("place", yolo_result.get("place"))),
                "objects": vision_objects,
                "scores": vision_image.get("scores", yolo_result.get("scores", {})),
                "image_size": vision_size,
            }
            if any(not isinstance(obj, dict) or obj.get("box") is None for obj in vision_objects):
                print(f"[VISION] {view_name}: box 없는 객체가 있습니다. 비전 서버 응답을 확인하세요.", flush=True)

        place = next((v.get("place") for v in view_results.values() if v.get("place") in TOPICS_BY_PLACE), None)
        objects = object_names([obj for view in view_results.values() for obj in view.get("objects", [])])
        context_id = save_detected_context(user_id, place, objects)
        return {
            "success": True,
            "contextId": context_id,
            "place": place,
            "objects": objects,
            "topics": topics_for(place),
            "views": view_results,
            "yolo_skipped": SKIP_YOLO,
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "YOLO 서버 응답 시간이 초과되었습니다.",
        }
    except requests.exceptions.RequestException as error:
        return {
            "success": False,
            "message": "YOLO 서버에 연결할 수 없습니다.",
            "error": str(error),
        }
    except Exception as error:
        print("4방향 공간/사물 분석 오류:", error)
        return {
            "success": False,
            "message": f"4방향 공간/사물 분석 중 오류가 발생했습니다: {error}",
        }

```

from datetime import date, timedelta, datetime
import json
import uuid
import threading
import base64
import io
import wave
import time
import cv2
import numpy as np
import requests
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import (
Boolean,
Column,
Date,
DateTime,
ForeignKey,
Integer,
String,
UniqueConstraint,
create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from ultralytics import YOLO

# =========================================================

# 기본 설정

# =========================================================

app = FastAPI()
VOICE_LIMIT_SECONDS = 40
VOICE_SILENCE_SECONDS = 10
CONTEXT_TTL_MINUTES = 30
_start_guard = threading.RLock()
_start_cache = {}
_voice_runtime = {}
_voice_runtime_lock = threading.Lock()
def _runtime(session_id: str):
with _voice_runtime_lock:
return _voice_runtime.setdefault(session_id, {
"user_speaking": False,
"ai_playing": False,
"close_after_user": False,
"one_more_after_ai": False,
"silence_timer": None,
"five_min_timer": None,
"pending_closing_response": None,
"processing_audio": False,
"audio_requests_in_flight": 0,
"closing_lock": threading.RLock(),
"active_stream_turn": None,
"active_stream_response": None,
})
def _cancel_timer(timer):
if timer is not None:
try:
timer.cancel()
except Exception:
pass
def _set_session_closing_wait(session_id: str):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if session and session.status == "ACTIVE" and not session.closing_requested:
session.status = "CLOSING_WAIT"
db.commit()
finally:
db.close()
def _schedule_silence_close(session_id: str):
state = _runtime(session_id)
_cancel_timer(state.get("silence_timer"))
timer = threading.Timer(VOICE_SILENCE_SECONDS, _request_closing_without_audio, args=(session_id,))
timer.daemon = True
state["silence_timer"] = timer
timer.start()
def _on_five_minutes(session_id: str):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if not session or session.status == "ENDED" or session.closing_requested:
return
if session.status == "ACTIVE":
session.status = "CLOSING_WAIT"
db.commit()
finally:
db.close()
state = _runtime(session_id)
print(f"[VOICE TIMER] 5분 도달: {session_id}", flush=True)
if state.get("processing_audio"):
# 음성 업로드/STT 처리 중에는 종료를 예약만 합니다.
state["close_after_user"] = True
return
if state["user_speaking"]:
state["close_after_user"] = True
return
if state["ai_playing"]:
state["one_more_after_ai"] = True
return
_schedule_silence_close(session_id)
def _schedule_five_min_timer(session_id: str):
state = _runtime(session_id)
_cancel_timer(state.get("five_min_timer"))
timer = threading.Timer(VOICE_LIMIT_SECONDS, _on_five_minutes, args=(session_id,))
timer.daemon = True
state["five_min_timer"] = timer
timer.start()
def _request_closing_without_audio(session_id: str):
state = _runtime(session_id)
# 음성 요청과 종료 상태 전환을 동일한 잠금으로 보호합니다.
with state["closing_lock"]:
if (state.get("pending_closing_response") is not None
or state["user_speaking"] or state["ai_playing"]
or state.get("processing_audio")):
return
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if not session or session.status != "CLOSING_WAIT" or session.closing_requested:
return
messages = db.query(VoiceConversationMessage).filter(VoiceConversationMessage.session_id == session_id).order_by(VoiceConversationMessage.created_at.asc()).all()
history = [{"role": m.role, "content": m.content} for m in messages]
session.status = "CLOSING"
session.closing_requested = True
db.commit()
except Exception:
db.close()
raise
# 네트워크 요청 중에는 잠금을 유지하지 않습니다.
_cancel_timer(state.get("silence_timer"))
state["silence_timer"] = None
try:
payload = {"history_json": json.dumps(history, ensure_ascii=False), "is_closing": "true"}
payload.update(ai_fields_for_session(db, session_id))
response = requests.post(VOICE_CONVERSE_URL, data=payload, headers={"ngrok-skip-browser-warning": "true"}, timeout=120)
if not response.ok:
raise RuntimeError(f"AI closing 실패: {response.status_code} {response.text[:300]}")
result = response.json()
ai_text = result.get("ai_text")
if not ai_text:
raise RuntimeError("AI closing 응답에 ai_text가 없습니다.")
db.add(VoiceConversationMessage(session_id=session_id, role="assistant", content=ai_text))
db.commit()
state["pending_closing_response"] = {"success": True, "session_id": session_id, "user_text": "", "ai_text": ai_text, "audio_base64": result.get("audio_base64") or "", "is_closing": True, "status": "CLOSING"}
print(f"[AUTO CLOSING] AI: {ai_text}", flush=True)
except Exception as error:
session.status = "CLOSING_WAIT"
session.closing_requested = False
db.commit()
state["pending_closing_response"] = {"success": False, "session_id": session_id, "message": str(error)}
print(f"[AUTO CLOSING ERROR] {error}", flush=True)
finally:
db.close()

# ngrok을 다시 실행하면 주소가 바뀔 수 있으므로 현재 AI 서버 주소인지 확인하세요.

# 기존 대화 / 예문 AI 서버

AI_BASE = "https://unruly-lego-unarmored.ngrok-free.dev"

VOICE_CONVERSE_URL = f"{AI_BASE}/ai/converse"
VOICE_CONVERSE_STREAM_URL = f"{AI_BASE}/ai/converse/stream"
GENERATE_URL = f"{AI_BASE}/generate"

# 음성 대화는 VOICE_CONVERSE_URL(/ai/converse)만 사용

# 새 YOLO 공간 + 사물 인식 서버

YOLO_BASE = "https://audio-agreed-sadness.ngrok-free.dev"
YOLO_ANALYZE_URL = f"{YOLO_BASE}/yolo/analyze"
YOLO_ANALYZE_MULTIPLE_URL = f"{YOLO_BASE}/yolo/analyze-multiple"
FEEDBACK_URL = f"{YOLO_BASE}/feedback"
EXTRACT_URL = FEEDBACK_URL
SKIP_YOLO = False

# False: 단어를 1개 이상 학습하면 그날의 연속 학습으로 인정

# True: 하루 목표(daily_goal)를 달성해야 연속 학습으로 인정

STREAK_REQUIRES_DAILY_GOAL = True
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],  # 배포 시에는 실제 프론트 주소만 넣는 것을 권장
allow_credentials=False,
allow_methods=["*"],
allow_headers=["*"],
)

# =========================================================

# YOLO 모델 (기존 /detect API용)

# =========================================================

import unicodedata

def normalize_place(place):
if not isinstance(place, str):
return ""
place = unicodedata.normalize("NFKD", place)
place = "".join(c for c in place if not unicodedata.combining(c))
place = place.lower().strip()
place = place.replace(" ", "*").replace("-", "*")
return place

TOPICS_BY_PLACE = {'cafe': [{'topic_id': 'cafe_order', 'title_ko': '주문', 'title_en': 'Ordering'}, {'topic_id': 'cafe_customization', 'title_ko': '옵션 변경', 'title_en': 'Customization'}, {'topic_id': 'cafe_dine_in_or_takeout', 'title_ko': '포장/매장', 'title_en': 'Dine-in or Takeout'}, {'topic_id': 'cafe_payment', 'title_ko': '결제', 'title_en': 'Payment'}], 'restaurant': [{'topic_id': 'restaurant_reservation', 'title_ko': '예약', 'title_en': 'Reservation'}, {'topic_id': 'restaurant_order', 'title_ko': '주문', 'title_en': 'Ordering'}, {'topic_id': 'restaurant_menu_customization', 'title_ko': '메뉴 커스텀', 'title_en': 'Menu Customization'}, {'topic_id': 'restaurant_request_during_meal', 'title_ko': '식사 중 필요한 것 요청', 'title_en': 'Requests During a Meal'}, {'topic_id': 'restaurant_payment', 'title_ko': '계산', 'title_en': 'Payment'}], 'office': [{'topic_id': 'office_small_talk', 'title_ko': '사무실 스몰토크', 'title_en': 'Office Small Talk'}, {'topic_id': 'office_meeting', 'title_ko': '회의', 'title_en': 'Meeting'}, {'topic_id': 'office_progress_update', 'title_ko': '업무 진행 상황 보고', 'title_en': 'Progress Update'}, {'topic_id': 'office_requests_and_scheduling', 'title_ko': '부탁 및 일정 조율', 'title_en': 'Requests and Scheduling'}, {'topic_id': 'office_apology_refusal_problem_solving', 'title_ko': '사과와 거절 그리고 문제해결', 'title_en': 'Apologies, Refusals and Problem Solving'}], 'classroom': [{'topic_id': 'classroom_answer_teacher', 'title_ko': '선생님 질문에 대답하기', 'title_en': 'Answering the Teacher'}, {'topic_id': 'classroom_ask_question', 'title_ko': '질문하기', 'title_en': 'Asking Questions'}, {'topic_id': 'classroom_discussion', 'title_ko': '학급 토론', 'title_en': 'Class Discussion'}, {'topic_id': 'classroom_presentation', 'title_ko': '발표', 'title_en': 'Presentation'}], 'park': [{'topic_id': 'park_walking', 'title_ko': '산책', 'title_en': 'Walking'}, {'topic_id': 'park_weather_and_atmosphere', 'title_ko': '날씨 및 분위기', 'title_en': 'Weather and Atmosphere'}, {'topic_id': 'park_picnic_and_relaxing', 'title_ko': '피크닉 및 휴식', 'title_en': 'Picnics and Relaxing'}, {'topic_id': 'park_with_pets_and_children', 'title_ko': '반려동물 및 아이들과 함께할 때', 'title_en': 'With Pets and Children'}], 'street': [{'topic_id': 'street_ask_directions', 'title_ko': '길 묻기', 'title_en': 'Asking for Directions'}, {'topic_id': 'street_ask_transportation', 'title_ko': '교통수단 묻기', 'title_en': 'Asking About Transportation'}, {'topic_id': 'street_run_into_someone', 'title_ko': '길거리에서 우연히 마주치기', 'title_en': 'Running Into Someone'}], 'kitchen': [{'topic_id': 'kitchen_prepare_ingredients', 'title_ko': '재료 손질', 'title_en': 'Preparing Ingredients'}, {'topic_id': 'kitchen_cooking', 'title_ko': '조리하기', 'title_en': 'Cooking'}, {'topic_id': 'kitchen_find_utensils', 'title_ko': '도구 찾기', 'title_en': 'Finding Utensils'}, {'topic_id': 'kitchen_clean_up', 'title_ko': '정리 정돈', 'title_en': 'Cleaning Up'}, {'topic_id': 'kitchen_set_the_table', 'title_ko': '식탁 차리기', 'title_en': 'Setting the Table'}, {'topic_id': 'kitchen_taste_and_adjust', 'title_ko': '맛 평가', 'title_en': 'Tasting and Adjusting'}, {'topic_id': 'kitchen_offer_and_pass_food', 'title_ko': '음식 권하고 건네기', 'title_en': 'Offering and Passing Food'}, {'topic_id': 'kitchen_finish_meal', 'title_ko': '식사 마무리', 'title_en': 'Finishing the Meal'}], 'living_room': [{'topic_id': 'living_room_watch_tv', 'title_ko': 'TV 보기', 'title_en': 'Watching TV'}, {'topic_id': 'living_room_choose_streaming', 'title_ko': 'OTT 고르기', 'title_en': 'Choosing Streaming Content'}, {'topic_id': 'living_room_play_music', 'title_ko': '음악 틀기', 'title_en': 'Playing Music'}, {'topic_id': 'living_room_use_appliances', 'title_ko': '가전제품 다루기', 'title_en': 'Using Appliances'}, {'topic_id': 'living_room_tidy_up', 'title_ko': '정리 정돈', 'title_en': 'Tidying Up'}], 'bedroom': [{'topic_id': 'bedroom_wake_up', 'title_ko': '기상', 'title_en': 'Waking Up'}, {'topic_id': 'bedroom_choose_clothes', 'title_ko': '옷 고르기', 'title_en': 'Choosing Clothes'}, {'topic_id': 'bedroom_prepare_for_sleep', 'title_ko': '잠자리 환경 조성', 'title_en': 'Preparing for Sleep'}, {'topic_id': 'bedroom_sleep_habits', 'title_ko': '잠버릇', 'title_en': 'Sleep Habits'}], 'beach': [{'topic_id': 'beach_find_a_spot', 'title_ko': '해변 자리 잡기', 'title_en': 'Finding a Spot'}, {'topic_id': 'beach_weather_and_sea_conditions', 'title_ko': '날씨 및 바다 상태', 'title_en': 'Weather and Sea Conditions'}, {'topic_id': 'beach_sunscreen_and_tanning', 'title_ko': '선크림 & 태닝', 'title_en': 'Sunscreen and Tanning'}, {'topic_id': 'beach_activities', 'title_ko': '해변 액티비티', 'title_en': 'Beach Activities'}, {'topic_id': 'beach_swimming', 'title_ko': '물놀이', 'title_en': 'Swimming'}], 'airport': [{'topic_id': 'airport_check_in', 'title_ko': '체크인', 'title_en': 'Check-in'}, {'topic_id': 'airport_immigration', 'title_ko': '출입국심사', 'title_en': 'Immigration'}, {'topic_id': 'airport_find_gate', 'title_ko': '탑승구 찾기', 'title_en': 'Finding the Gate'}], 'zoo': [{'topic_id': 'zoo_ticket_office', 'title_ko': '매표소', 'title_en': 'Ticket Office'}, {'topic_id': 'zoo_view_animals', 'title_ko': '동물 관람', 'title_en': 'Viewing Animals'}, {'topic_id': 'zoo_follow_rules', 'title_ko': '관람규칙', 'title_en': 'Following Rules'}, {'topic_id': 'zoo_gift_shop', 'title_ko': '기념품 샵', 'title_en': 'Gift Shop'}], 'bathroom': [{'topic_id': 'bathroom_shower_and_wash_hair', 'title_ko': '샤워 및 머리 감기', 'title_en': 'Showering and Washing Hair'}, {'topic_id': 'bathroom_brush_teeth', 'title_ko': '양치', 'title_en': 'Brushing Teeth'}, {'topic_id': 'bathroom_clean_up_after_use', 'title_ko': '욕실 사용 후 정리', 'title_en': 'Cleaning Up After Use'}, {'topic_id': 'bathroom_report_condition', 'title_ko': '화장실 상태', 'title_en': 'Reporting Bathroom Conditions'}], 'library': [{'topic_id': 'library_register_membership', 'title_ko': '회원 등록', 'title_en': 'Membership Registration'}, {'topic_id': 'library_find_book', 'title_ko': '도서 위치 찾기', 'title_en': 'Finding a Book'}, {'topic_id': 'library_borrow_book', 'title_ko': '대출', 'title_en': 'Borrowing Books'}, {'topic_id': 'library_return_book', 'title_ko': '반납', 'title_en': 'Returning Books'}]}

# AI 서버의 topic_json용 역할/목표. 프론트 응답의 기존 주제 목록은 변경하지 않습니다.

TOPIC_AI_DETAILS = {'cafe_order': {'user_role': 'customer', 'ai_role': 'barista', 'goal': 'Order a drink.'},
'cafe_customization': {'user_role': 'customer',
'ai_role': 'barista',
'goal': 'Practice customization in a cafe situation.'},
'cafe_dine_in_or_takeout': {'user_role': 'customer',
'ai_role': 'barista',
'goal': 'Practice dine-in or takeout in a cafe situation.'},
'cafe_payment': {'user_role': 'customer', 'ai_role': 'barista', 'goal': 'Practice payment in a cafe situation.'},
'restaurant_reservation': {'user_role': 'customer',
'ai_role': 'server',
'goal': 'Practice reservation in a restaurant situation.'},
'restaurant_order': {'user_role': 'customer',
'ai_role': 'server',
'goal': 'Practice ordering in a restaurant situation.'},
'restaurant_menu_customization': {'user_role': 'customer',
'ai_role': 'server',
'goal': 'Practice menu customization in a restaurant situation.'},
'restaurant_request_during_meal': {'user_role': 'customer',
'ai_role': 'server',
'goal': 'Practice requests during a meal in a restaurant situation.'},
'restaurant_payment': {'user_role': 'customer',
'ai_role': 'server',
'goal': 'Practice payment in a restaurant situation.'},
'office_small_talk': {'user_role': 'coworker',
'ai_role': 'coworker',
'goal': 'Practice office small talk in a office situation.'},
'office_meeting': {'user_role': 'team member',
'ai_role': 'meeting facilitator',
'goal': 'Practice meeting in a office situation.'},
'office_progress_update': {'user_role': 'employee',
'ai_role': 'manager',
'goal': 'Practice progress update in a office situation.'},
'office_requests_and_scheduling': {'user_role': 'employee',
'ai_role': 'coworker',
'goal': 'Practice requests and scheduling in a office situation.'},
'office_apology_refusal_problem_solving': {'user_role': 'employee',
'ai_role': 'coworker',
'goal': 'Practice apologies, refusals and problem solving in a office '
'situation.'},
'classroom_answer_teacher': {'user_role': 'student',
'ai_role': 'teacher',
'goal': 'Practice answering the teacher in a classroom situation.'},
'classroom_ask_question': {'user_role': 'student',
'ai_role': 'teacher',
'goal': 'Practice asking questions in a classroom situation.'},
'classroom_discussion': {'user_role': 'student',
'ai_role': 'classmate',
'goal': 'Practice class discussion in a classroom situation.'},
'classroom_presentation': {'user_role': 'presenter',
'ai_role': 'audience member',
'goal': 'Practice presentation in a classroom situation.'},
'park_walking': {'user_role': 'park visitor',
'ai_role': 'friend',
'goal': 'Practice walking in a park situation.'},
'park_weather_and_atmosphere': {'user_role': 'park visitor',
'ai_role': 'friend',
'goal': 'Practice weather and atmosphere in a park situation.'},
'park_picnic_and_relaxing': {'user_role': 'park visitor',
'ai_role': 'friend',
'goal': 'Practice picnics and relaxing in a park situation.'},
'park_with_pets_and_children': {'user_role': 'park visitor',
'ai_role': 'another park visitor',
'goal': 'Practice with pets and children in a park situation.'},
'street_ask_directions': {'user_role': 'pedestrian',
'ai_role': 'local resident',
'goal': 'Practice asking for directions in a street situation.'},
'street_ask_transportation': {'user_role': 'traveler',
'ai_role': 'local resident',
'goal': 'Practice asking about transportation in a street situation.'},
'street_run_into_someone': {'user_role': 'friend',
'ai_role': 'friend',
'goal': 'Practice running into someone in a street situation.'},
'kitchen_prepare_ingredients': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice preparing ingredients in a kitchen situation.'},
'kitchen_cooking': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice cooking in a kitchen situation.'},
'kitchen_find_utensils': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice finding utensils in a kitchen situation.'},
'kitchen_clean_up': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice cleaning up in a kitchen situation.'},
'kitchen_set_the_table': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice setting the table in a kitchen situation.'},
'kitchen_taste_and_adjust': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice tasting and adjusting in a kitchen situation.'},
'kitchen_offer_and_pass_food': {'user_role': 'host',
'ai_role': 'guest',
'goal': 'Practice offering and passing food in a kitchen situation.'},
'kitchen_finish_meal': {'user_role': 'home cook',
'ai_role': 'cooking partner',
'goal': 'Practice finishing the meal in a kitchen situation.'},
'living_room_watch_tv': {'user_role': 'resident',
'ai_role': 'friend',
'goal': 'Practice watching tv in a living room situation.'},
'living_room_choose_streaming': {'user_role': 'resident',
'ai_role': 'friend',
'goal': 'Practice choosing streaming content in a living room situation.'},
'living_room_play_music': {'user_role': 'resident',
'ai_role': 'friend',
'goal': 'Practice playing music in a living room situation.'},
'living_room_use_appliances': {'user_role': 'resident',
'ai_role': 'friend',
'goal': 'Practice using appliances in a living room situation.'},
'living_room_tidy_up': {'user_role': 'resident',
'ai_role': 'friend',
'goal': 'Practice tidying up in a living room situation.'},
'bedroom_wake_up': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice waking up in a bedroom situation.'},
'bedroom_choose_clothes': {'user_role': 'friend',
'ai_role': 'friend',
'goal': 'Practice choosing clothes in a bedroom situation.'},
'bedroom_prepare_for_sleep': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice preparing for sleep in a bedroom situation.'},
'bedroom_sleep_habits': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice sleep habits in a bedroom situation.'},
'beach_find_a_spot': {'user_role': 'beach visitor',
'ai_role': 'friend',
'goal': 'Practice finding a spot in a beach situation.'},
'beach_weather_and_sea_conditions': {'user_role': 'beach visitor',
'ai_role': 'friend',
'goal': 'Practice weather and sea conditions in a beach situation.'},
'beach_sunscreen_and_tanning': {'user_role': 'beach visitor',
'ai_role': 'friend',
'goal': 'Practice sunscreen and tanning in a beach situation.'},
'beach_activities': {'user_role': 'beach visitor',
'ai_role': 'friend',
'goal': 'Practice beach activities in a beach situation.'},
'beach_swimming': {'user_role': 'beach visitor',
'ai_role': 'friend',
'goal': 'Practice swimming in a beach situation.'},
'airport_check_in': {'user_role': 'traveler',
'ai_role': 'check-in agent',
'goal': 'Practice check-in in a airport situation.'},
'airport_immigration': {'user_role': 'traveler',
'ai_role': 'immigration officer',
'goal': 'Practice immigration in a airport situation.'},
'airport_find_gate': {'user_role': 'traveler',
'ai_role': 'airport staff',
'goal': 'Practice finding the gate in a airport situation.'},
'zoo_ticket_office': {'user_role': 'visitor',
'ai_role': 'ticket clerk',
'goal': 'Practice ticket office in a zoo situation.'},
'zoo_view_animals': {'user_role': 'zoo visitor',
'ai_role': 'zoo staff',
'goal': 'Practice viewing animals in a zoo situation.'},
'zoo_follow_rules': {'user_role': 'zoo visitor',
'ai_role': 'zoo staff',
'goal': 'Practice following rules in a zoo situation.'},
'zoo_gift_shop': {'user_role': 'customer',
'ai_role': 'shop clerk',
'goal': 'Practice gift shop in a zoo situation.'},
'bathroom_shower_and_wash_hair': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice showering and washing hair in a bathroom situation.'},
'bathroom_brush_teeth': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice brushing teeth in a bathroom situation.'},
'bathroom_clean_up_after_use': {'user_role': 'resident',
'ai_role': 'roommate',
'goal': 'Practice cleaning up after use in a bathroom situation.'},
'bathroom_report_condition': {'user_role': 'guest',
'ai_role': 'staff member',
'goal': 'Practice reporting bathroom conditions in a bathroom situation.'},
'library_register_membership': {'user_role': 'visitor',
'ai_role': 'librarian',
'goal': 'Practice membership registration in a library situation.'},
'library_find_book': {'user_role': 'library visitor',
'ai_role': 'librarian',
'goal': 'Practice finding a book in a library situation.'},
'library_borrow_book': {'user_role': 'library member',
'ai_role': 'librarian',
'goal': 'Practice borrowing books in a library situation.'},
'library_return_book': {'user_role': 'library member',
'ai_role': 'librarian',
'goal': 'Practice returning books in a library situation.'}}
def ai_topic_payload(topic):
return {**topic, **TOPIC_AI_DETAILS[topic["topic_id"]]}
def ai_context_fields(context, topic=None):
"""AI /ai/converse multipart/form-data fields; context is stored in SQLite."""
fields = {
"mode": "topic" if topic else "free",
"context_json": json.dumps({
"scene_id": context.id,
"place": context.place,
"objects": json.loads(context.objects_json),
}, ensure_ascii=False),
}
if topic:
fields["topic_id"] = topic["topic_id"]
fields["topic_json"] = json.dumps(ai_topic_payload(topic), ensure_ascii=False)
return fields
def ai_fields_for_session(db, session_id):
"""Rebuild the AI context from persistent session link even after a restart."""
link = db.query(VoiceSessionContext).filter(VoiceSessionContext.session_id == session_id).first()
if link is None:
return {"mode": "free"}
context = db.query(DetectedContext).filter(DetectedContext.id == link.context_id).first()
if context is None:
raise RuntimeError("대화에 연결된 인식 컨텍스트가 없습니다.")
topic = None
if link.mode == "topic":
topic = next((t for t in topics_for(context.place) if t["topic_id"] == link.topic_id), None)
if topic is None:
raise RuntimeError("대화에 연결된 추천 주제가 없습니다.")
return ai_context_fields(context, topic)
def ai_stream_fields_for_session(db, session_id):
"""스트리밍 회화: 추천 주제에는 context_json을 보내지 않습니다."""
fields = ai_fields_for_session(db, session_id)
if fields.get("mode") == "topic":
return {"mode": "topic", "topic_id": fields["topic_id"]}
return {"mode": "free", **({"context_json": fields["context_json"]} if "context_json" in fields else {})}

def topics_for(place):
return TOPICS_BY_PLACE.get(place, [])
@app.get("/topics/{place}")
def get_place_topics(place: str):
return {"success": place in TOPICS_BY_PLACE, "place": place, "topics": topics_for(place)}
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8m.pt")

# =========================================================

# 객체 탐지 API

# =========================================================

@app.post("/detect")
async def detect(
file: UploadFile = File(...)):
image_bytes = await file.read()
if not image_bytes:
return {
"success": False,
"message": "업로드된 이미지가 비어 있습니다.",
}
img = cv2.imdecode(
np.frombuffer(image_bytes, np.uint8),
cv2.IMREAD_COLOR,
)
if img is None:
return {
"success": False,
"message": "이미지를 읽을 수 없습니다.",
}
image_height, image_width = img.shape[:2]
results = yolo_model(img)
objects = []
for result in results:
if result.boxes is None:
continue
for box in result.boxes:
cls_id = int(box.cls[0].item())
confidence = float(box.conf[0].item())
# YOLO bounding box 좌표 (원본 이미지 기준)
x1, y1, x2, y2 = box.xyxy[0].tolist()
objects.append(
{
"name": yolo_model.names[cls_id],
"confidence": confidence,
"box": {
"x1": int(x1),
"y1": int(y1),
"x2": int(x2),
"y2": int(y2),
},
}
)
if not objects:
return {
"success": False,
"message": "객체를 감지하지 못했습니다.",
"objects": [],
"image_size": {
"width": image_width,
"height": image_height,
},
}
objects.sort(key=lambda item: item["confidence"], reverse=True)
main_object = objects[0]["name"]
try:
response = requests.post(
f"{AI_BASE}/generate",
json={"object": main_object},
headers={"ngrok-skip-browser-warning": "true"},
timeout=60,
)
print("AI URL:", f"{AI_BASE}/generate")
print("AI status:", response.status_code)
print("AI content-type:", response.headers.get("content-type"))
print("AI response:", response.text[:500])
if not response.ok:
return {
"success": False,
"object": main_object,
"objects": objects,
"message": "AI 서버 요청에 실패했습니다.",
"ai_status": response.status_code,
"ai_response": response.text[:500],
}
try:
ai_result = response.json()
print("AI JSON:", ai_result)
except requests.exceptions.JSONDecodeError:
return {
"success": False,
"object": main_object,
"objects": objects,
"message": "AI 서버가 JSON이 아닌 응답을 반환했습니다.",
"ai_status": response.status_code,
"ai_response": response.text[:500],
}
return {
"success": True,
"object": main_object,
"objects": objects,
"image_size": {
"width": image_width,
"height": image_height,
},
"ai": ai_result,
}
except requests.exceptions.Timeout:
return {
"success": False,
"object": main_object,
"objects": objects,
"message": "AI 서버 응답 시간이 초과되었습니다.",
}
except requests.exceptions.ConnectionError as error:
return {
"success": False,
"object": main_object,
"objects": objects,
"message": "AI 서버에 연결할 수 없습니다.",
"error": str(error),
}
except requests.exceptions.RequestException as error:
return {
"success": False,
"object": main_object,
"objects": objects,
"message": "AI 서버 요청 중 오류가 발생했습니다.",
"error": str(error),
}
@app.post("/detect-space")
async def detect_space(
file: UploadFile = File(...),
user_id: int = Form(...),
):
image_bytes = await file.read()
if not image_bytes:
return {
"success": False,
"message": "업로드된 이미지가 비어 있습니다.",
}
# 프론트에서 bounding box 좌표를 화면 좌표로 변환할 때 사용
original_img = cv2.imdecode(
np.frombuffer(image_bytes, np.uint8),
cv2.IMREAD_COLOR,
)
if original_img is None:
return {
"success": False,
"message": "이미지를 읽을 수 없습니다.",
}
image_height, image_width = original_img.shape[:2]
try:
files = {
"file": (
file.filename or "image.jpg",
image_bytes,
file.content_type or "image/jpeg",
)
}
response = requests.post(
YOLO_ANALYZE_URL,
files=files,
headers={
"ngrok-skip-browser-warning": "true"
},
timeout=60,
)
print("YOLO 서버 상태:", response.status_code)
print("YOLO 서버 응답:", response.text[:1000])
if not response.ok:
return {
"success": False,
"message": "YOLO 서버 요청에 실패했습니다.",
"status_code": response.status_code,
"ai_response": response.text[:500],
}
yolo_result = response.json()
if not yolo_result.get("success"):
return {
"success": False,
"message": "YOLO 분석에 실패했습니다.",
"yolo": yolo_result,
}
study_result = complete_study(
StudyCompleteRequest(
user_id=user_id,
word_count=1,
)
)
if not study_result.get("success"):
return {
"success": False,
"place": yolo_result.get("place"),
"objects": yolo_result.get("objects", []),
"scores": yolo_result.get("scores", {}),
"image_size": {
"width": image_width,
"height": image_height,
},
"message": study_result.get(
"message",
"학습 진도 저장에 실패했습니다.",
),
"study": study_result,
}
place = normalize_place(yolo_result.get("place"))
if place == "café":
place = "cafe"
objects = object_names(yolo_result.get("objects", []))
context_id = save_detected_context(user_id, place, objects)
return {
"success": True,
"contextId": context_id,
"topics": topics_for(place),
"space": place,
"place": place,
"objects": objects,
"scores": yolo_result.get("scores", {}),
"image_size": {
"width": image_width,
"height": image_height,
},
"study": study_result,
}
except requests.exceptions.Timeout:
return {
"success": False,
"message": "YOLO 서버 응답 시간이 초과되었습니다.",
}
except requests.exceptions.RequestException as error:
return {
"success": False,
"message": "YOLO 서버에 연결할 수 없습니다.",
"error": str(error),
}
except Exception as error:
print("공간/사물 분석 오류:", error)
return {
"success": False,
"message": f"공간/사물 분석 중 오류가 발생했습니다: {error}",
}

# =========================================================

# 4방향 이미지 공간 + 사물 인식 API

# front / back / left / right 이미지를 한 번에 받음

# 각 이미지는 기존 YOLO /analyze 서버로 따로 전송

# 학습 진도는 4장이 성공해도 1회만 증가

# =========================================================

@app.post("/detect-space-4")
async def detect_space_4(
front: UploadFile = File(...),
back: UploadFile = File(...),
left: UploadFile = File(...),
right: UploadFile = File(...),
user_id: int = Form(...),
):
uploads = {
"front": front,
"back": back,
"left": left,
"right": right,
}
view_results = {}
try:
for view_name, upload in uploads.items():
image_bytes = await upload.read()
if not image_bytes:
return {
"success": False,
"message": f"{view_name} 이미지가 비어 있습니다.",
}
# 이미지 크기와 box 좌표는 비전 서버 응답을 그대로 사용합니다.
# 원본 이미지를 디코딩해 크기를 재계산하지 않습니다.
# =====================================================
# 테스트 모드: YOLO 서버 건너뛰기
# =====================================================
if SKIP_YOLO:
print(f"[{view_name}] YOLO 서버 우회 테스트")
fake_objects = {
"front": [
{"name": "table", "confidence": 0.95},
{"name": "cup", "confidence": 0.91},
],
"back": [
{"name": "chair", "confidence": 0.93},
],
"left": [
{"name": "menu", "confidence": 0.90},
],
"right": [
{"name": "bottle", "confidence": 0.89},
],
}
view_results[view_name] = {
"space": "cafe",
"place": "cafe",
"objects": fake_objects[view_name],
"scores": {},
"image_size": None,  # 테스트 모드에는 비전 서버 응답이 없습니다.
}
continue
# =====================================================
# 실제 YOLO 서버 사용
# =====================================================
files = {
"file": (
upload.filename or f"{view_name}.jpg",
image_bytes,
upload.content_type or "image/jpeg",
)
}
response = requests.post(
YOLO_ANALYZE_URL,
files=files,
headers={
"ngrok-skip-browser-warning": "true",
},
timeout=60,
)
print(
f"[{view_name}] YOLO 서버 상태:",
response.status_code,
)
print(
f"[{view_name}] YOLO 서버 응답:",
response.text[:1000],
)
if not response.ok:
return {
"success": False,
"failed_view": view_name,
"message": f"{view_name} 이미지 YOLO 서버 요청에 실패했습니다.",
"status_code": response.status_code,
"ai_response": response.text[:500],
}
try:
yolo_result = response.json()
except requests.exceptions.JSONDecodeError:
return {
"success": False,
"failed_view": view_name,
"message": f"{view_name} 이미지 YOLO 서버가 JSON이 아닌 응답을 반환했습니다.",
"ai_response": response.text[:500],
}
if not yolo_result.get("success"):
return {
"success": False,
"failed_view": view_name,
"message": f"{view_name} 이미지 YOLO 분석에 실패했습니다.",
"yolo": yolo_result,
}
# /yolo/analyze 응답이 images 배열을 포함하는 경우 해당 이미지의
# image_size/objects를 사용합니다. 개별 요청이므로 첫 번째 항목입니다.
images = yolo_result.get("images")
if isinstance(images, list) and images:
vision_image = images[0]
if not isinstance(vision_image, dict):
return {"success": False, "failed_view": view_name,
"message": "비전 서버 images[0] 응답 형식이 올바르지 않습니다."}
else:
# 비전 서버가 단일 이미지 객체를 반환하는 형식도 지원합니다.
vision_image = yolo_result

```
        vision_size = vision_image.get("image_size")
        if not isinstance(vision_size, dict) or not all(
            key in vision_size for key in ("width", "height")
        ):
            return {"success": False, "failed_view": view_name,
                    "message": "비전 서버 응답에 image_size.width/height가 없습니다."}

        vision_objects = vision_image.get("objects", [])
        if not isinstance(vision_objects, list):
            return {"success": False, "failed_view": view_name,
                    "message": "비전 서버 objects 응답이 배열이 아닙니다."}
        # 원본 객체 dict를 그대로 보존하므로 box가 있으면 그대로 포함됩니다.
        # 비전 서버가 box를 보내지 않은 객체는 좌표를 생성하지 않습니다.
        view_results[view_name] = {
            "space": vision_image.get("place", yolo_result.get("place")),
            "place": normalize_place(vision_image.get("place", yolo_result.get("place"))),
            "objects": vision_objects,
            "scores": vision_image.get("scores", yolo_result.get("scores", {})),
            "image_size": vision_size,
        }
        missing_boxes = sum(
            1 for obj in vision_objects
            if not isinstance(obj, dict) or obj.get("box") is None
        )
        if missing_boxes:
            print(f"[VISION] {view_name}: box 없는 객체 {missing_boxes}개 (비전 서버 확인 필요)", flush=True)
    place = next((v.get("place") for v in view_results.values() if v.get("place") in TOPICS_BY_PLACE), None)
    objects = object_names([obj for view in view_results.values() for obj in view.get("objects", [])])
    context_id = save_detected_context(user_id, place, objects)
    return {
        "success": True,
        "contextId": context_id,
        "place": place,
        "objects": objects,
        "topics": topics_for(place),
        "views": view_results,
        "yolo_skipped": SKIP_YOLO,
    }
except requests.exceptions.Timeout:
    return {
        "success": False,
        "message": "YOLO 서버 응답 시간이 초과되었습니다.",
    }
except requests.exceptions.RequestException as error:
    return {
        "success": False,
        "message": "YOLO 서버에 연결할 수 없습니다.",
        "error": str(error),
    }
except Exception as error:
    print("4방향 공간/사물 분석 오류:", error)
    return {
        "success": False,
        "message": f"4방향 공간/사물 분석 중 오류가 발생했습니다: {error}",
    }
```

# =========================================================

# DB 설정

# =========================================================

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
DATABASE_URL,
connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(
bind=engine,
autoflush=False,
autocommit=False,
)
Base = declarative_base()

# =========================================================

# DB 테이블

# =========================================================

class User(Base):
**tablename** = "users"
id = Column(Integer, primary_key=True, index=True)
username = Column(String, unique=True, nullable=False, index=True)
password = Column(String, nullable=False)
class StudyStatus(Base):
**tablename** = "study_status"
id = Column(Integer, primary_key=True, index=True)
user_id = Column(
Integer,
ForeignKey("users.id"),
unique=True,
nullable=False,
)
current_streak = Column(Integer, default=0, nullable=False)
best_streak = Column(Integer, default=0, nullable=False)
# 마지막으로 '연속 학습 조건을 충족한' 날짜
last_study_date = Column(Date, nullable=True)
daily_goal = Column(Integer, default=5, nullable=False)
class DailyStudyRecord(Base):
**tablename** = "daily_study_records"
id = Column(Integer, primary_key=True, index=True)
user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
study_date = Column(Date, nullable=False)
studied_word_count = Column(Integer, default=0, nullable=False)
# 이 값은 "오늘 한 번이라도 학습했는가"를 의미
studied_today = Column(Boolean, default=False, nullable=False)
**table_args** = (
UniqueConstraint(
"user_id",
"study_date",
name="unique_user_study_date",
),
)
class VoiceConversationSession(Base):
**tablename** = "voice_conversation_sessions"
id = Column(String, primary_key=True)
user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
started_at = Column(DateTime, nullable=False)
ended_at = Column(DateTime, nullable=True)
# ACTIVE -> CLOSING_WAIT -> CLOSING -> ENDED
status = Column(String, default="ACTIVE", nullable=False)
closing_requested = Column(Boolean, default=False, nullable=False)
class VoiceConversationMessage(Base):
**tablename** = "voice_conversation_messages"
id = Column(Integer, primary_key=True, index=True)
session_id = Column(
String,
ForeignKey("voice_conversation_sessions.id"),
nullable=False,
)
role = Column(String, nullable=False)
content = Column(String, nullable=False)
created_at = Column(
DateTime,
default=datetime.utcnow,
nullable=False,
)
class DetectedContext(Base):
**tablename** = "detected_contexts"
id = Column(String, primary_key=True)
user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
place = Column(String, nullable=False)
objects_json = Column(String, nullable=False)
created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
class VoiceSessionContext(Base):
**tablename** = "voice_session_contexts"
session_id = Column(String, ForeignKey("voice_conversation_sessions.id"), primary_key=True)
context_id = Column(String, ForeignKey("detected_contexts.id"), nullable=False)
mode = Column(String, nullable=False)
topic_id = Column(String, nullable=True)
Base.metadata.create_all(bind=engine)

# =========================================================

# 요청 데이터 모델

# =========================================================

class AuthRequest(BaseModel):
username: str
password: str
class StudyCompleteRequest(BaseModel):
user_id: int
word_count: int = Field(gt=0)
class ConversationStartRequest(BaseModel):
target_expression: str
class VoiceStateRequest(BaseModel):
event: str

# =========================================================

# 회원가입 / 로그인

# =========================================================

def object_names(objects):
"""Normalize YOLO object dicts or strings to the requested string list."""
result = []
for obj in objects or []:
name = obj.get("name") if isinstance(obj, dict) else obj
if isinstance(name, str) and name and name not in result:
result.append(name)
return result

def get_valid_context(db, user_id, context_id):
if not context_id:
raise HTTPException(status_code=400, detail="contextId가 필요합니다.")
context = db.query(DetectedContext).filter(
DetectedContext.id == context_id, DetectedContext.user_id == user_id
).first()
if context is None:
raise HTTPException(status_code=404, detail="존재하지 않는 contextId입니다.")
if datetime.utcnow() >= context.created_at + timedelta(minutes=CONTEXT_TTL_MINUTES):
raise HTTPException(status_code=410, detail="contextId가 만료되었습니다. 공간을 다시 인식해 주세요.")
return context
def save_detected_context(user_id, place, objects):
place = normalize_place(place)
if place not in TOPICS_BY_PLACE:
return None
db = SessionLocal()
try:
context_id = f"ctx_{uuid.uuid4().hex}"
db.add(DetectedContext(id=context_id, user_id=user_id, place=place, objects_json=json.dumps(objects, ensure_ascii=False)))
db.commit()
return context_id
finally:
db.close()
@app.post("/register")
def register(data: AuthRequest):
db = SessionLocal()
try:
existing_user = (
db.query(User)
.filter(User.username == data.username)
.first()
)
if existing_user:
return {
"success": False,
"message": "이미 존재하는 아이디입니다.",
}
user = User(
username=data.username,
password=data.password,
)
db.add(user)
db.commit()
db.refresh(user)
return {
"success": True,
"message": "회원가입 성공",
"user_id": user.id,
}
except Exception as error:
db.rollback()
return {
"success": False,
"message": str(error),
}
finally:
db.close()
@app.post("/conversation/start")
def start_conversation():
user_id = 1
db = SessionLocal()
print("[VOICE START ALIAS] /conversation/start 호출")
print("[VOICE START ALIAS] user_id:", user_id)
try:
user = (
db.query(User)
.filter(User.id == user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다.",
}
session_id = str(uuid.uuid4())
session = VoiceConversationSession(
id=session_id,
user_id=user_id,
started_at=datetime.utcnow(),
)
db.add(session)
db.commit()
print("[VOICE START ALIAS] session_id 생성:", session_id)
return {
"success": True,
"session_id": session_id,
}
finally:
db.close()
# 이 엔드포인트는 예전 텍스트/객체 대화용입니다.
# 현재 음성 대화 프론트에서는 /voice/conversation/start 를 사용해야 합니다.
return {
"success": False,
"message": (
"이 API는 현재 사용하지 않습니다. "
"음성 대화는 /voice/conversation/start 로 세션을 만들고, "
"/voice/conversation/message 로 녹음 파일을 보내세요."
),
}
@app.post("/voice/conversation/start")
def start_voice_conversation(user_id: int = Form(...), contextId: str | None = Form(None), topic_id: str | None = Form(None), mode: str = Form("free")):
# 동일 사용자/인식 컨텍스트/모드/주제의 중복 시작을 직렬화합니다.
# 단일 uvicorn 프로세스 기준으로 동작합니다.
with _start_guard:
db = SessionLocal()
try:
get_valid_context(db, user_id, contextId)
key = (user_id, contextId, mode, topic_id) if contextId else None
cached = _start_cache.get(key) if key is not None else None
if cached:
session = db.query(VoiceConversationSession).filter(
VoiceConversationSession.id == cached["session_id"]
).first()
if session and session.status == "ACTIVE" and session.ended_at is None:
return cached
_start_cache.pop(key, None)
finally:
db.close()
result = _start_voice_conversation_impl(user_id, contextId, topic_id, mode)
if result.get("success"):
if key is not None:
_start_cache[key] = result
else:
# AI 시작 실패 시 생성된 세션을 활성 상태로 남겨두지 않습니다.
failed_id = result.get("session_id")
if failed_id:
db = SessionLocal()
try:
failed = db.query(VoiceConversationSession).filter(
VoiceConversationSession.id == failed_id
).first()
if failed:
failed.status = "ENDED"
failed.ended_at = datetime.utcnow()
db.commit()
state = _voice_runtime.pop(failed_id, None)
if state:
_cancel_timer(state.get("five_min_timer"))
_cancel_timer(state.get("silence_timer"))
finally:
db.close()
return result
def _start_voice_conversation_impl(user_id, contextId, topic_id, mode):
db = SessionLocal()
print("[VOICE START] user_id:", user_id)
try:
user = (
db.query(User)
.filter(User.id == user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다.",
}
selected_topic = None
selected_context = None
if mode not in ("free", "topic"):
return {"success": False, "message": "mode는 free 또는 topic이어야 합니다."}
if not contextId:
return {"success": False, "message": "자유주제와 추천 주제 대화 모두 contextId가 필요합니다."}
if mode == "topic" and not topic_id:
return {"success": False, "message": "추천 주제 대화에는 topic_id가 필요합니다."}
if mode == "free" and topic_id:
return {"success": False, "message": "자유주제 대화에는 topic_id를 보내지 마세요."}
if contextId:
selected_context = get_valid_context(db, user_id, contextId)
if mode == "topic":
selected_topic = next((t for t in topics_for(selected_context.place) if t["topic_id"] == topic_id), None)
if not selected_topic:
raise HTTPException(status_code=422, detail="해당 장소에 속하지 않거나 존재하지 않는 topic_id입니다.")
# 1. 새 음성 대화 세션 생성
session_id = str(uuid.uuid4())
session = VoiceConversationSession(
id=session_id,
user_id=user_id,
started_at=datetime.utcnow(),
)
db.add(session)
if selected_context:
db.add(VoiceSessionContext(session_id=session_id, context_id=contextId, mode=mode, topic_id=topic_id))
db.commit()
state = _runtime(session_id)
if selected_context:
state["topic_context"] = ai_context_fields(selected_context, selected_topic)
_schedule_five_min_timer(session_id)
print("[VOICE START] session_id 생성:", session_id)
# 2. 자유주제 시작 인사 요청
# 사용자 음성은 보내지 않음
try:
response = requests.post(
VOICE_CONVERSE_URL,
data={
"mode": mode,
"is_opening": "true",
"is_closing": "false",
"history_json": "[]",
**ai_fields_for_session(db, session_id),
},
headers={
"ngrok-skip-browser-warning": "true"
},
timeout=120,
)
print("[VOICE OPENING] AI status:", response.status_code)
print("[VOICE OPENING] AI response:", response.text[:500])
if not response.ok:
return {
"success": False,
"message": "AI 시작 인사 요청에 실패했습니다.",
"session_id": session_id,
"status_code": response.status_code,
"ai_response": response.text[:500],
}
result = response.json()
ai_text = result.get("ai_text")
audio_base64 = result.get("audio_base64") or ""
if not ai_text:
return {
"success": False,
"message": "AI 시작 인사 응답에 ai_text가 없습니다.",
"session_id": session_id,
}
# 시작 인사도 대화 기록에 저장
db.add(
VoiceConversationMessage(
session_id=session_id,
role="assistant",
content=ai_text,
)
)
db.commit()
print(f"[AI 시작 인사] {ai_text}", flush=True)
# 3. 프론트에 시작 인사 텍스트 + 음성 전달
return {
"success": True,
"session_id": session_id,
"contextId": contextId,
"mode": mode,
"ai_text": ai_text,
"audio_base64": audio_base64,
"is_opening": True,
"status": "ACTIVE",
}
except requests.exceptions.Timeout:
return {
"success": False,
"message": "AI 시작 인사 응답 시간이 초과되었습니다.",
"session_id": session_id,
}
except requests.exceptions.RequestException as error:
return {
"success": False,
"message": "AI 시작 인사 서버에 연결할 수 없습니다.",
"session_id": session_id,
"error": str(error),
}
except Exception as error:
return {
"success": False,
"message": f"AI 시작 인사 처리 중 오류가 발생했습니다: {error}",
"session_id": session_id,
}
finally:
db.close()
@app.post("/voice/conversation/message")
async def voice_conversation_message(session_id: str = Form(...), audio: UploadFile = File(...)):
state = _runtime(session_id)
# 요청이 도착한 즉시 진행 중으로 표시해 자동 종료와 충돌하지 않게 합니다.
with state["closing_lock"]:
state["audio_requests_in_flight"] += 1
state["processing_audio"] = True
_cancel_timer(state.get("silence_timer"))
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if not session:
return {"success": False, "message": "대화 세션을 찾을 수 없습니다."}
if session.ended_at is not None or session.status == "ENDED":
return {"success": False, "message": "이미 종료된 대화 세션입니다.", "session_id": session_id}
if session.status == "CLOSING" or session.closing_requested:
return {"success": False, "message": "마무리 답변 처리 중입니다.", "session_id": session_id}
audio_bytes = await audio.read()
if not audio_bytes:
return {"success": False, "message": "음성 파일이 비어 있습니다."}
messages = db.query(VoiceConversationMessage).filter(VoiceConversationMessage.session_id == session_id).order_by(VoiceConversationMessage.created_at.asc()).all()
history = [{"role": m.role, "content": m.content} for m in messages]
history_json = json.dumps(history, ensure_ascii=False)
elapsed = (datetime.utcnow() - session.started_at).total_seconds()
if elapsed >= VOICE_LIMIT_SECONDS and session.status == "ACTIVE":
session.status = "CLOSING_WAIT"
db.commit()
is_closing = False
if session.status == "CLOSING_WAIT" and not session.closing_requested:
if state["close_after_user"] or state["one_more_after_ai"] or (elapsed >= VOICE_LIMIT_SECONDS and not state["ai_playing"]):
is_closing = True
if is_closing:
session.status = "CLOSING"
session.closing_requested = True
state["close_after_user"] = False
state["one_more_after_ai"] = False
db.commit()
files = {"audio": (audio.filename or "audio.wav", audio_bytes, audio.content_type or "audio/wav")}
data = {"history_json": history_json, "is_closing": str(is_closing).lower()}
data.update(ai_fields_for_session(db, session_id))
try:
response = requests.post(VOICE_CONVERSE_URL, files=files, data=data, headers={"ngrok-skip-browser-warning": "true"}, timeout=120)
except Exception:
if is_closing:
session.status = "CLOSING_WAIT"
session.closing_requested = False
db.commit()
raise
if not response.ok:
if is_closing:
session.status = "CLOSING_WAIT"
session.closing_requested = False
db.commit()
return {"success": False, "message": "음성 AI 서버 요청에 실패했습니다.", "status_code": response.status_code, "ai_response": response.text[:500]}
result = response.json()
user_text = result.get("user_text")
ai_text = result.get("ai_text")
print(f"[AI 응답 로그] User: {user_text} | AI: {ai_text}", flush=True)
if is_closing:
print("========== [마무리 멘트] ==========", flush=True)
print(f"AI: {ai_text}", flush=True)
print("==================================", flush=True)
else:
print("[일반 대화]", flush=True)
audio_base64 = result.get("audio_base64") or ""
if user_text is None or ai_text is None:
if is_closing:
session.status = "CLOSING_WAIT"
session.closing_requested = False
db.commit()
return {"success": False, "message": "음성 AI 서버 응답에 user_text 또는 ai_text가 없습니다."}
if user_text:
db.add(VoiceConversationMessage(session_id=session_id, role="user", content=user_text))
if ai_text:
db.add(VoiceConversationMessage(session_id=session_id, role="assistant", content=ai_text))
db.commit()
return {"success": True, "session_id": session_id, "user_id": session.user_id, "user_text": user_text, "ai_text": ai_text, "audio_base64": audio_base64, "is_closing": is_closing, "status": session.status, "elapsed_seconds": int(elapsed)}
except requests.exceptions.Timeout:
return {"success": False, "message": "음성 AI 서버 응답 시간이 초과되었습니다."}
except requests.exceptions.RequestException as error:
return {"success": False, "message": "음성 AI 서버에 연결할 수 없습니다.", "error": str(error)}
except Exception as error:
db.rollback()
return {"success": False, "message": str(error)}
finally:
db.close()
with state["closing_lock"]:
state["audio_requests_in_flight"] = max(0, state["audio_requests_in_flight"] - 1)
state["processing_audio"] = state["audio_requests_in_flight"] > 0
# 마지막 업로드/STT가 끝난 후에만 자동 종료 타이머를 재개합니다.
if not state["processing_audio"]:
check_db = SessionLocal()
try:
current = check_db.query(VoiceConversationSession).filter(
VoiceConversationSession.id == session_id
).first()
if (current and current.status == "CLOSING_WAIT"
and not current.closing_requested
and not state["user_speaking"]
and not state["ai_playing"]
and state.get("pending_closing_response") is None):
_schedule_silence_close(session_id)
finally:
check_db.close()

# =========================================================

# 음성 대화 스트리밍 (기존 /voice/conversation/message는 유지)

# AI 서버는 SSE: user_transcript -> assistant_sentence_text/audio

# > turn_completed 순서로 이벤트를 전송해야 합니다.

# =========================================================

def _stream_sse(event):
return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

def _stream_audio_duration(event):
"""WAV base64이면 정확한 재생 길이(초)를 반환; 다른 형식은 None."""
existing = event.get("duration") or event.get("duration_seconds")
if isinstance(existing, (int, float)) and existing > 0:
return round(float(existing), 3)
encoded = event.get("audio_base64")
if not encoded or not isinstance(encoded, str):
return None
fmt = str(event.get("format") or "").lower()
if fmt and "wav" not in fmt and "wave" not in fmt:
return None
try:
audio_bytes = base64.b64decode(encoded, validate=True)
with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
rate = wav.getframerate()
return round(wav.getnframes() / rate, 3) if rate else None
except (ValueError, EOFError, wave.Error, base64.binascii.Error):
return None

def _stream_parse_sse(lines):
"""requests.iter_lines()에서 SSE data 이벤트를 하나씩 추출."""
data = []
for raw in lines:
if isinstance(raw, bytes):
raw = raw.decode("utf-8", errors="replace")
if raw == "":
if data:
yield "\n".join(data)
data = []
continue
if raw.startswith("data:"):
data.append(raw[5:].lstrip(" "))
if data:
yield "\n".join(data)

@app.post("/voice/conversation/{session_id}/message/stream")
async def stream_voice_message(session_id: str, audio: UploadFile = File(...)):
"""AI SSE를 중계하고 사용자/AI 문장을 DB에 저장합니다."""
state = _runtime(session_id)
# 같은 세션의 요청은 중복 처리하지 않습니다.
with state["closing_lock"]:
if state.get("processing_audio") or state.get("audio_requests_in_flight", 0):
raise HTTPException(status_code=409, detail="이전 음성 요청이 처리 중입니다.")
state["audio_requests_in_flight"] = 1
state["processing_audio"] = True
_cancel_timer(state.get("silence_timer"))
state["silence_timer"] = None

```
is_closing = False
session_reserved = False
try:
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="음성 파일이 비어 있습니다.")
    with SessionLocal() as db:
        with state["closing_lock"]:
            session = db.query(VoiceConversationSession).filter(
                VoiceConversationSession.id == session_id
            ).first()
            if session is None:
                raise HTTPException(status_code=404, detail="대화 세션이 없습니다.")
            if session.status in ("CLOSING", "ENDED") or session.closing_requested:
                raise HTTPException(status_code=409, detail="대화가 종료 중이거나 종료되었습니다.")
            messages = db.query(VoiceConversationMessage).filter(
                VoiceConversationMessage.session_id == session_id
            ).order_by(VoiceConversationMessage.created_at.asc(), VoiceConversationMessage.id.asc()).all()
            history_json = json.dumps([
                {"role": m.role, "content": m.content} for m in messages
            ], ensure_ascii=False)
            elapsed = (datetime.utcnow() - session.started_at).total_seconds()
            if elapsed >= VOICE_LIMIT_SECONDS and session.status == "ACTIVE":
                session.status = "CLOSING_WAIT"
                db.commit()
            is_closing = (
                session.status == "CLOSING_WAIT" and
                (state.get("close_after_user") or state.get("one_more_after_ai") or
                 (elapsed >= VOICE_LIMIT_SECONDS and not state.get("ai_playing")))
            )
            if is_closing:
                session.status = "CLOSING"
                session.closing_requested = True
                db.commit()
                state["close_after_user"] = False
                state["one_more_after_ai"] = False
                session_reserved = True
            fields = ai_stream_fields_for_session(db, session_id)
            fields.update({"history_json": history_json, "is_closing": str(is_closing).lower()})

    turn_id = uuid.uuid4().hex
    with state["closing_lock"]:
        state["active_stream_turn"] = turn_id

    filename = audio.filename or "audio.wav"
    content_type = audio.content_type or "audio/wav"

    def stream_events():
        user_saved = False
        assistant_saved = False
        user_text = ""
        assistant_text = ""
        sentence_texts = []
        finished = False
        try:
            with requests.post(
                VOICE_CONVERSE_STREAM_URL,
                files={"audio": (filename, audio_bytes, content_type)},
                data=fields,
                headers={"Accept": "text/event-stream", "ngrok-skip-browser-warning": "true"},
                stream=True,
                timeout=(15, 120),
            ) as upstream:
                with state["closing_lock"]:
                    if state.get("active_stream_turn") != turn_id:
                        return
                    state["active_stream_response"] = upstream

                print("[AI STREAM] URL:", VOICE_CONVERSE_STREAM_URL)
                print("[AI STREAM] 전송 필드:", fields)
                print("[AI STREAM] 파일명:", filename)
                print("[AI STREAM] MIME:", content_type)
                print("[AI STREAM] 파일 크기:", len(audio_bytes))

                print("[AI STREAM] 응답 상태:", upstream.status_code)

                if upstream.status_code >= 400:
                    error_body = upstream.text
                    print("[AI STREAM] 오류 응답:", error_body[:3000])
                    upstream.raise_for_status()

                ctype = upstream.headers.get("content-type", "").lower()
                if "text/event-stream" not in ctype:
                    raise RuntimeError("AI 서버가 SSE를 반환하지 않았습니다: " + ctype)
                # AI 서버는 텍스트를 먼저 전송할 수 있으므로 문장별 음성까지
                # 수신한 뒤 한 쌍으로 프론트에 전달합니다.
                # sentence_index는 AI 서버의 원래 값을 유지합니다.
                pending = {}
                emitted = set()
                arrival_order = []
                completed_event = None
                stream_error = False

                def sentence_key(event):
                    index = event.get("sentence_index", event.get("sequence", event.get("sentence_id")))
                    if index is None:
                        raise RuntimeError("AI 문장 이벤트에 sentence_index/sequence가 없습니다.")
                    return str(index)

                def emit_pair(key):
                    pair = pending[key]
                    text_event = dict(pair["text"])
                    audio_event = dict(pair["audio"])
                    sentence_id = f"{turn_id}:{key}"
                    for ev in (text_event, audio_event):
                        ev["sentence_id"] = sentence_id
                        ev.setdefault("sentence_index", pair["index"])
                        ev.setdefault("sequence", pair["index"])
                    duration = _stream_audio_duration(audio_event)
                    if duration is not None:
                        audio_event["duration"] = duration
                        text_event["duration"] = duration
                    print(f"[SSE SEND] {datetime.utcnow().isoformat()} sentence={key} text+audio duration={duration}", flush=True)
                    emitted.add(key)
                    return (_stream_sse(text_event), _stream_sse(audio_event))

                for payload in _stream_parse_sse(upstream.iter_lines(chunk_size=1)):
                    if state.get("active_stream_turn") != turn_id:
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        yield _stream_sse({"type": "error", "message": "AI 서버가 JSON이 아닌 SSE 이벤트를 보냈습니다."})
                        stream_error = True
                        break
                    if not isinstance(event, dict):
                        continue
                    kind = event.get("type") or event.get("event")
                    print(f"[SSE RECV] {datetime.utcnow().isoformat()} type={kind} index={event.get('sentence_index')} ", flush=True)
                    if kind == "user_transcript":
                        user_text = event.get("user_text") or event.get("text") or ""
                        if user_text and not user_saved:
                            with SessionLocal() as save_db:
                                save_db.add(VoiceConversationMessage(
                                    session_id=session_id, role="user", content=user_text
                                ))
                                save_db.commit()
                            user_saved = True
                        yield _stream_sse(event)
                    elif kind in ("assistant_sentence_text", "assistant_sentence_audio"):
                        key = sentence_key(event)
                        if key in emitted:
                            # 이미 전송한 문장을 다시 보내지 않습니다.
                            continue
                        if key not in pending:
                            pending[key] = {"index": event.get("sentence_index", event.get("sequence", key))}
                            arrival_order.append(key)
                        slot = "text" if kind == "assistant_sentence_text" else "audio"
                        pending[key][slot] = event
                        if "text" in pending[key] and "audio" in pending[key]:
                            for out in emit_pair(key):
                                yield out
                    elif kind == "turn_completed":
                        completed_event = event
                        # 완료 신호를 먼저 보내지 않고 누락된 문장/음성을 검증합니다.
                        break
                    elif kind == "error":
                        yield _stream_sse(event)
                        stream_error = True
                        break
                    else:
                        yield _stream_sse(event)

                if completed_event is not None and state.get("active_stream_turn") == turn_id:
                    missing = [key for key in arrival_order if key not in emitted]
                    if missing:
                        print(f"[SSE INCOMPLETE] missing sentence pairs={missing}", flush=True)
                        yield _stream_sse({"type": "error", "message": "AI 서버의 문장 텍스트/음성이 모두 도착하지 않았습니다.", "missing_sentence_indices": missing})
                        stream_error = True
                    else:
                        parts = [
                            (pending[key]["text"].get("text") or pending[key]["text"].get("ai_sentence") or "").strip()
                            for key in arrival_order
                        ]
                        joined = " ".join(part for part in parts if part)
                        upstream_text = (completed_event.get("ai_text") or "").strip()
                        # 최종 ai_text가 전송된 문장과 다르면 완료 처리하지 않습니다.
                        if joined and upstream_text and " ".join(joined.split()) != " ".join(upstream_text.split()):
                            print(f"[SSE MISMATCH] emitted={joined!r} final={upstream_text!r}", flush=True)
                            yield _stream_sse({"type": "error", "message": "AI 최종 답변과 문장별 이벤트가 일치하지 않습니다.", "ai_text": upstream_text})
                            stream_error = True
                        else:
                            assistant_text = upstream_text or joined
                            if assistant_text and not assistant_saved:
                                with SessionLocal() as save_db:
                                    save_db.add(VoiceConversationMessage(
                                        session_id=session_id, role="assistant", content=assistant_text
                                    ))
                                    save_db.commit()
                                assistant_saved = True
                            completed_event["ai_text"] = assistant_text
                            print(f"[SSE COMPLETE] {datetime.utcnow().isoformat()} sentences={len(emitted)}", flush=True)
                            yield _stream_sse(completed_event)
                            finished = True
                if stream_error:
                    return
                if not finished and state.get("active_stream_turn") == turn_id:
                    yield _stream_sse({"type": "error", "message": "AI 스트림이 완료 이벤트 없이 종료되었습니다."})
        except (requests.RequestException, RuntimeError) as exc:
            if state.get("active_stream_turn") == turn_id:
                yield _stream_sse({"type": "error", "message": str(exc)})
        except Exception as exc:
            if state.get("active_stream_turn") == turn_id:
                yield _stream_sse({"type": "error", "message": "스트리밍 처리 오류: " + str(exc)})
        finally:
            with state["closing_lock"]:
                if state.get("active_stream_turn") == turn_id:
                    state["active_stream_turn"] = None
                    state["active_stream_response"] = None
                # 오류로 마무리 요청이 실패하면 다시 시도할 수 있게 합니다.
                if is_closing and not finished:
                    with SessionLocal() as db:
                        current = db.query(VoiceConversationSession).filter(
                            VoiceConversationSession.id == session_id
                        ).first()
                        if current and current.status == "CLOSING":
                            current.status = "CLOSING_WAIT"
                            current.closing_requested = False
                            db.commit()
                state["audio_requests_in_flight"] = max(
                    0, state.get("audio_requests_in_flight", 1) - 1
                )
                state["processing_audio"] = state["audio_requests_in_flight"] > 0
                if not state["processing_audio"]:
                    with SessionLocal() as db:
                        current = db.query(VoiceConversationSession).filter(
                            VoiceConversationSession.id == session_id
                        ).first()
                        if (current and current.status == "CLOSING_WAIT" and
                            not current.closing_requested and
                            not state.get("user_speaking") and not state.get("ai_playing") and
                            state.get("pending_closing_response") is None):
                            _schedule_silence_close(session_id)

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
except BaseException:
    with state["closing_lock"]:
        if session_reserved:
            with SessionLocal() as db:
                current = db.query(VoiceConversationSession).filter(
                    VoiceConversationSession.id == session_id
                ).first()
                if current and current.status == "CLOSING":
                    current.status = "CLOSING_WAIT"
                    current.closing_requested = False
                    db.commit()
        state["audio_requests_in_flight"] = max(0, state["audio_requests_in_flight"] - 1)
        state["processing_audio"] = state["audio_requests_in_flight"] > 0
    raise
```

# =========================================================

# 일반 음성 대화: 사용자 STT 먼저 표시하는 단계별 SSE 응답

# 기존 JSON /voice/conversation/message 와 문장별 스트리밍 API는 유지합니다.

# AI 서버의 /ai/converse/stream 이 user_transcript 를 먼저 보내야 합니다.

# =========================================================

@app.post("/voice/conversation/{session_id}/message/progressive")
async def progressive_voice_message(session_id: str, audio: UploadFile = File(...)):
"""프론트가 user_text -> AI 로딩 -> AI 문장/음성 -> 완료 순서로 받는 SSE API."""
# 기존 스트리밍 함수의 세션 검증, 타이머, DB 저장, 오류 처리 및
# 문장/음성 매칭 로직을 그대로 재사용하여 중복 저장을 방지합니다.
upstream_response = await stream_voice_message(session_id=session_id, audio=audio)

```
def progressive_events():
    user_sent = False
    completed = False
    try:
        # 내부 스트리밍 응답은 이미 SSE data: JSON 형식입니다.
        for payload in _stream_parse_sse(
            line
            for chunk in upstream_response.body_iterator
            for line in (chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk).split("\n")
        ):
            try:
                event = json.loads(payload)
            except (TypeError, ValueError):
                yield _stream_sse({"type": "error", "message": "잘못된 스트리밍 이벤트입니다."})
                return
            kind = event.get("type") or event.get("event")
            if kind == "user_transcript":
                user_sent = True
                yield _stream_sse({
                    "type": "user_text", "session_id": session_id,
                    "user_text": event.get("user_text") or event.get("text") or "",
                })
                yield _stream_sse({"type": "ai_loading", "session_id": session_id})
            elif kind == "assistant_sentence_text":
                # 프론트는 문장별 텍스트를 즉시 표시할 수도 있습니다.
                yield _stream_sse(event)
            elif kind == "assistant_sentence_audio":
                # TTS가 문장별로 생성되므로 audio_base64를 문장 단위로 전송합니다.
                yield _stream_sse(event)
            elif kind == "turn_completed":
                completed = True
                yield _stream_sse({
                    "type": "ai_completed", "session_id": session_id,
                    "ai_text": event.get("ai_text") or "",
                    "is_closing": event.get("is_closing", False),
                })
            elif kind == "error":
                yield _stream_sse(event)
                return
            else:
                yield _stream_sse(event)
        if not completed:
            yield _stream_sse({"type": "error", "message": "AI 답변이 완료되지 않았습니다."})
    finally:
        # 클라이언트 연결 종료 시 내부 스트림도 종료하여 정리 로직을 실행합니다.
        close = getattr(upstream_response.body_iterator, "close", None)
        if callable(close):
            close()

return StreamingResponse(
    progressive_events(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
)
```

@app.post("/voice/conversation/{session_id}/state")
def update_voice_state(session_id: str, data: VoiceStateRequest):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if not session:
return {"success": False, "message": "대화 세션을 찾을 수 없습니다."}
state = _runtime(session_id)
if session.status == "ENDED":
return {"success": True, "session_id": session_id, "status": "ENDED", "already_ended": True}
event = data.event
if event == "user_speech_started":
# 끼어들기: 기존 SSE 응답을 닫고 늦게 오는 이벤트를 무효화
state["active_stream_turn"] = None
upstream = state.get("active_stream_response")
if upstream is not None:
upstream.close()
state["user_speaking"] = True
_cancel_timer(state.get("silence_timer"))
if session.status == "CLOSING_WAIT" and not state["ai_playing"]:
state["close_after_user"] = True
elif event == "user_speech_finished":
state["user_speaking"] = False
elif event == "ai_playback_started":
state["ai_playing"] = True
_cancel_timer(state.get("silence_timer"))
elif event == "ai_playback_finished":
state["ai_playing"] = False
if session.status == "CLOSING":
session.status = "ENDED"
session.ended_at = datetime.utcnow()
db.commit()
_cancel_timer(state.get("five_min_timer"))
_cancel_timer(state.get("silence_timer"))
study = complete_study(StudyCompleteRequest(user_id=session.user_id, word_count=1))
return {"success": True, "session_id": session_id, "status": "ENDED", "study": study}
if session.status == "CLOSING_WAIT" and state["one_more_after_ai"]:
_schedule_silence_close(session_id)
else:
return {"success": False, "message": "지원하지 않는 state event입니다.", "event": event}
pending = state.get("pending_closing_response")
return {"success": True, "session_id": session_id, "status": session.status, "event": event, "pending_closing_response": pending}
finally:
db.close()
@app.get("/voice/conversation/{session_id}/pending")
def get_pending_closing_response(session_id: str):
state = _runtime(session_id)
pending = state.get("pending_closing_response")
if pending is None:
return {"success": True, "session_id": session_id, "pending": False}
state["pending_closing_response"] = None
return {"success": True, "session_id": session_id, "pending": True, "response": pending}
@app.post("/voice/conversation/{session_id}/end")
def end_voice_conversation(session_id: str):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(VoiceConversationSession.id == session_id).first()
if not session:
return {"success": False, "message": "대화 세션을 찾을 수 없습니다."}
if session.status != "ENDED":
session.status = "ENDED"
session.ended_at = datetime.utcnow()
db.commit()
state = _runtime(session_id)
state["active_stream_turn"] = None
upstream = state.get("active_stream_response")
if upstream is not None:
upstream.close()
_cancel_timer(state.get("five_min_timer"))
_cancel_timer(state.get("silence_timer"))
return {"success": True, "session_id": session_id, "status": "ENDED"}
finally:
db.close()

# =========================================================

# 대화 피드백 / 핵심 표현 추출

# AI + 사용자 발화 전체를 대화 순서대로 전달합니다.

# AI 서버에서 translation_by_turn 및 keywords_all[].source를 반환합니다.

# =========================================================

def _feedback_conversation(db, session_id):
messages = db.query(VoiceConversationMessage).filter(
VoiceConversationMessage.session_id == session_id
).order_by(
VoiceConversationMessage.created_at.asc(),
VoiceConversationMessage.id.asc()
).all()
return [
{"id": index, "role": message.role, "message": message.content.strip()}
for index, message in enumerate(
(m for m in messages
if m.role in ("user", "assistant")
and isinstance(m.content, str) and m.content.strip()),
start=1
)
]

def _request_feedback_ai(conversation):
response = requests.post(
FEEDBACK_URL,
json={"conversation": conversation},
headers={"ngrok-skip-browser-warning": "true"},
timeout=180
)
response.raise_for_status()
result = response.json()
if not isinstance(result, dict):
raise ValueError("AI 서버가 JSON 객체가 아닌 피드백을 반환했습니다.")
return result

@app.post("/voice/conversation/{session_id}/feedback")
def get_conversation_feedback(session_id: str):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(
VoiceConversationSession.id == session_id
).first()
if not session:
return {"success": False, "message": "대화 세션을 찾을 수 없습니다."}

```
    conversation = _feedback_conversation(db, session_id)
    if not any(turn["role"] == "user" for turn in conversation):
        return {"success": False, "message": "피드백을 생성할 사용자 발화가 없습니다."}

    # 시작 당시 연결된 컨텍스트만 조회합니다(만료 여부는 검사하지 않음).
    linked = db.query(VoiceSessionContext).filter(
        VoiceSessionContext.session_id == session_id
    ).first()
    context = (
        db.query(DetectedContext).filter(DetectedContext.id == linked.context_id).first()
        if linked else None
    )
    context_info = (
        {"contextId": context.id, "place": context.place,
         "objects": json.loads(context.objects_json), "mode": linked.mode,
         "topic_id": linked.topic_id}
        if context else None
    )

    result = _request_feedback_ai(conversation)
    return {"success": True, "session_id": session_id,
            "context": context_info, "feedback": result}
except requests.exceptions.Timeout:
    return {"success": False, "message": "피드백 AI 서버 응답 시간이 초과되었습니다."}
except requests.exceptions.RequestException as error:
    return {"success": False, "message": "피드백 AI 서버 요청에 실패했습니다.", "error": str(error)}
except (ValueError, json.JSONDecodeError) as error:
    return {"success": False, "message": "피드백 AI 서버 응답 형식이 올바르지 않습니다.", "error": str(error)}
finally:
    db.close()
```

@app.post("/voice/conversation/{session_id}/extract")
def extract_conversation_words(session_id: str):
db = SessionLocal()
try:
session = db.query(VoiceConversationSession).filter(
VoiceConversationSession.id == session_id
).first()
if not session:
raise HTTPException(status_code=404, detail="대화 세션을 찾을 수 없습니다.")

```
    conversation = _feedback_conversation(db, session_id)
    if not any(turn["role"] == "user" for turn in conversation):
        raise HTTPException(status_code=400, detail="분석할 사용자 발화가 없습니다.")

    result = _request_feedback_ai(conversation)
    required_fields = (
        "dialog_start_id", "dialog_end_id", "summary",
        "paraphrase_by_user", "keywords_all"
    )
    if any(field not in result for field in required_fields):
        raise HTTPException(
            status_code=502,
            detail="AI 서버의 추출 결과 형식이 올바르지 않습니다."
        )
    return {"success": True, "session_id": session_id, "result": result}
except requests.exceptions.RequestException as error:
    raise HTTPException(status_code=502, detail=f"AI 서버 요청 실패: {error}")
except (ValueError, json.JSONDecodeError) as error:
    raise HTTPException(status_code=502, detail=f"AI 서버 응답 형식 오류: {error}")
finally:
    db.close()
```

@app.get("/voice/conversation/{session_id}")
def get_voice_conversation(session_id: str):
db = SessionLocal()
try:
session = (
db.query(VoiceConversationSession)
.filter(VoiceConversationSession.id == session_id)
.first()
)
if not session:
return {
"success": False,
"message": "대화 세션을 찾을 수 없습니다.",
}
messages = (
db.query(VoiceConversationMessage)
.filter(VoiceConversationMessage.session_id == session_id)
.order_by(VoiceConversationMessage.created_at.asc())
.all()
)
return {
"success": True,
"session_id": session.id,
"user_id": session.user_id,
"started_at": session.started_at.isoformat(),
"ended_at": (
session.ended_at.isoformat()
if session.ended_at is not None
else None
),
"messages": [
{
"role": message.role,
"content": message.content,
"created_at": message.created_at.isoformat(),
}
for message in messages
],
}
finally:
db.close()
@app.get("/voice/conversations/{user_id}")
def get_voice_conversations(user_id: int):
"""
특정 계정(user_id)의 모든 음성 대화 세션과
각 세션의 전체 대화 내용을 최신 세션부터 조회한다.
"""
db = SessionLocal()
try:
user = (
db.query(User)
.filter(User.id == user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다.",
}
sessions = (
db.query(VoiceConversationSession)
.filter(VoiceConversationSession.user_id == user_id)
.order_by(VoiceConversationSession.started_at.desc())
.all()
)
conversations = []
for session in sessions:
messages = (
db.query(VoiceConversationMessage)
.filter(VoiceConversationMessage.session_id == session.id)
.order_by(VoiceConversationMessage.created_at.asc())
.all()
)
conversations.append(
{
"session_id": session.id,
"started_at": session.started_at.isoformat(),
"ended_at": (
session.ended_at.isoformat()
if session.ended_at is not None
else None
),
"message_count": len(messages),
"messages": [
{
"role": message.role,
"content": message.content,
"created_at": message.created_at.isoformat(),
}
for message in messages
],
}
)
return {
"success": True,
"user_id": user_id,
"username": user.username,
"conversation_count": len(conversations),
"conversations": conversations,
}
except Exception as error:
return {
"success": False,
"message": str(error),
}
finally:
db.close()
@app.post("/login")
def login(data: AuthRequest):
db = SessionLocal()
try:
user = (
db.query(User)
.filter(User.username == data.username)
.first()
)
if not user:
return {
"success": False,
"message": "아이디가 존재하지 않습니다.",
}
if user.password != data.password:
return {
"success": False,
"message": "비밀번호가 틀렸습니다.",
}
return {
"success": True,
"message": "로그인 성공",
"user_id": user.id,
"username": user.username,
}
finally:
db.close()

# =========================================================

# 연속 학습 계산 도우미

# =========================================================

def update_streak(status: StudyStatus, today: date) -> None:
"""오늘을 연속 학습일로 한 번만 반영한다."""
if status.last_study_date == today:
return
yesterday = today - timedelta(days=1)
if status.last_study_date == yesterday:
status.current_streak += 1
else:
status.current_streak = 1
status.last_study_date = today
status.best_streak = max(
status.best_streak,
status.current_streak,
)

# =========================================================

# 학습 완료 API

# =========================================================

@app.post("/study/complete")
def complete_study(data: StudyCompleteRequest):
db = SessionLocal()
try:
today = date.today()
user = (
db.query(User)
.filter(User.id == data.user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다.",
}
status = (
db.query(StudyStatus)
.filter(StudyStatus.user_id == data.user_id)
.first()
)
if not status:
status = StudyStatus(
user_id=data.user_id,
current_streak=0,
best_streak=0,
daily_goal=5,
)
db.add(status)
db.flush()
today_record = (
db.query(DailyStudyRecord)
.filter(
DailyStudyRecord.user_id == data.user_id,
DailyStudyRecord.study_date == today,
)
.first()
)
if not today_record:
today_record = DailyStudyRecord(
user_id=data.user_id,
study_date=today,
studied_word_count=0,
studied_today=False,
)
db.add(today_record)
db.flush()
count_before = today_record.studied_word_count
today_record.studied_word_count = min(
today_record.studied_word_count + data.word_count,
status.daily_goal
)
today_record.studied_today = True
goal_achieved_before = count_before >= status.daily_goal
goal_achieved_now = (
today_record.studied_word_count >= status.daily_goal
)
if STREAK_REQUIRES_DAILY_GOAL:
# 목표 미달에서 목표 달성으로 처음 넘어간 순간에만 반영
if not goal_achieved_before and goal_achieved_now:
update_streak(status, today)
else:
# 오늘 첫 학습 요청에서 바로 반영
if count_before == 0:
update_streak(status, today)
db.commit()
db.refresh(today_record)
db.refresh(status)
progress_rate = int(
min(
today_record.studied_word_count
/ status.daily_goal
* 100,
100,
)
)
return {
"success": True,
"message": "학습 기록이 저장되었습니다.",
"current_streak": status.current_streak,
"best_streak": status.best_streak,
"studied_today": today_record.studied_today,
"today_word_count": today_record.studied_word_count,
"daily_goal": status.daily_goal,
"goal_achieved": goal_achieved_now,
"progress_rate": progress_rate,
"streak_requires_daily_goal": STREAK_REQUIRES_DAILY_GOAL,
}
except Exception as error:
db.rollback()
return {
"success": False,
"message": str(error),
}
finally:
db.close()

# =========================================================

# 홈 화면 API

# =========================================================

@app.get("/home/{user_id}")
def get_home(user_id: int):
db = SessionLocal()
try:
today = date.today()
user = (
db.query(User)
.filter(User.id == user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다.",
}
status = (
db.query(StudyStatus)
.filter(StudyStatus.user_id == user_id)
.first()
)
if not status:
status = StudyStatus(
user_id=user_id,
current_streak=0,
best_streak=0,
daily_goal=5,
)
db.add(status)
db.commit()
db.refresh(status)
if status.last_study_date is None:
days_since_last_study = None
else:
days_since_last_study = (
today - status.last_study_date
).days
# 어제까지 조건을 충족하지 않았다면 오늘 조회 시 streak 종료
if days_since_last_study >= 2 and status.current_streak != 0:
status.current_streak = 0
db.commit()
db.refresh(status)
today_record = (
db.query(DailyStudyRecord)
.filter(
DailyStudyRecord.user_id == user_id,
DailyStudyRecord.study_date == today,
)
.first()
)
if today_record:
today_word_count = today_record.studied_word_count
studied_today = today_record.studied_today
else:
today_word_count = 0
studied_today = False
goal_achieved = today_word_count >= status.daily_goal
# =====================================================
# 이번 주 학습 기록
# 월요일 \~ 일요일
# 하루 목표 5개를 달성한 날만 goal_achieved = True
# =====================================================
monday = today - timedelta(days=today.weekday())
weekly_study = []
day_names = [
"mon",
"tue",
"wed",
"thu",
"fri",
"sat",
"sun",
]
for i in range(7):
target_date = monday + timedelta(days=i)
record = (
db.query(DailyStudyRecord)
.filter(
DailyStudyRecord.user_id == user_id,
DailyStudyRecord.study_date == target_date,
)
.first()
)
if record:
word_count = record.studied_word_count
else:
word_count = 0
weekly_study.append(
{
"day": day_names[i],
"date": target_date.isoformat(),
"word_count": word_count,
"goal_achieved": word_count >= status.daily_goal,
}
)
progress_rate = int(
min(
today_word_count
/ status.daily_goal
* 100,
100,
)
)
if status.current_streak >= 30:
flame_level = 4
elif status.current_streak >= 7:
flame_level = 3
elif status.current_streak >= 3:
flame_level = 2
elif status.current_streak >= 1:
flame_level = 1
else:
flame_level = 0
if days_since_last_study is not None:
if days_since_last_study == 1:
flame_level = max(flame_level - 1, 0)
elif days_since_last_study == 2:
flame_level = max(flame_level - 2, 0)
elif days_since_last_study >= 3:
flame_level = 0
return {
"success": True,
"current_streak": status.current_streak,
"best_streak": status.best_streak,
"studied_today": studied_today,
"today_word_count": today_word_count,
"daily_goal": status.daily_goal,
"goal_achieved": goal_achieved,
"progress_rate": progress_rate,
"flame_level": flame_level,
"days_since_last_study": days_since_last_study,
"streak_requires_daily_goal": STREAK_REQUIRES_DAILY_GOAL,
"weekly_study": weekly_study,
}
except Exception as error:
return {
"success": False,
"message": str(error),
}
finally:
db.close()
@app.post("/study/reset-today/{user_id}")
def reset_today_study(user_id: int):
db = SessionLocal()
try:
today = date.today()
record = (
db.query(DailyStudyRecord)
.filter(
DailyStudyRecord.user_id == user_id,
DailyStudyRecord.study_date == today,
)
.first()
)
if not record:
return {
"success": False,
"message": "오늘 학습 기록이 없습니다."
}
record.studied_word_count = 0
record.studied_today = False
db.commit()
return {
"success": True,
"user_id": user_id,
"today_word_count": 0,
"message": "오늘 학습 기록을 0/5로 초기화했습니다."
}
finally:
db.close()
@app.get("/study/calendar/{user_id}")
def get_study_calendar(user_id: int, year: int, month: int):
db = SessionLocal()
try:
user = (
db.query(User)
.filter(User.id == user_id)
.first()
)
if not user:
return {
"success": False,
"message": "사용자를 찾을 수 없습니다."
}
# 해당 월의 시작일
start_date = date(year, month, 1)
# 다음 달 시작일
if month == 12:
end_date = date(year + 1, 1, 1)
else:
end_date = date(year, month + 1, 1)
records = (
db.query(DailyStudyRecord)
.filter(
DailyStudyRecord.user_id == user_id,
DailyStudyRecord.study_date >= start_date,
DailyStudyRecord.study_date < end_date,
)
.order_by(DailyStudyRecord.study_date.asc())
.all()
)
study_days = []
for record in records:
if record.studied_today:
study_days.append({
"date": record.study_date.isoformat(),
"word_count": record.studied_word_count,
"studied": True,
})
return {
"success": True,
"user_id": user_id,
"year": year,
"month": month,
"study_days": study_days,
}
except Exception as error:
return {
"success": False,
"message": str(error),
}
finally:
db.close()
