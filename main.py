from fastapi import FastAPI, UploadFile, File
import requests
from PIL import Image
import io
import torch
import open_clip
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from fastapi import File, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO

app = FastAPI()

AI_BASE = "https://unruly-lego-unarmored.ngrok-free.dev"


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    results = yolo_model(img)

    objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = yolo_model.names[cls_id]
            confidence = float(box.conf[0])

            objects.append({
                "name": name,
                "confidence": confidence
            })

    if not objects:
        return {"success": False}

    objects.sort(key=lambda x: x["confidence"], reverse=True)

    main_object = objects[0]["name"]

    response = requests.post(
        f"{AI_BASE}/generate",
        json={"object": main_object}
    )

    return {
        "object": main_object,
        "ai": response.json()
    }

clip_model=None
clip_preprocess=None
clip_tokenizer=None

yolo_model = YOLO("yolov8m.pt")

def load_model():
    global clip_model, clip_preprocess, clip_tokenizer

    if clip_model is None:
        clip_model, clip_preprocess, clip_tokenizer = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )

        clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        clip_model = clip_model.to(device).eval()

    return clip_model, clip_preprocess, clip_tokenizer

# ---------------------------
# 1. CLIP 모델 로드
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"



# ---------------------------
# 2. 공간 후보 정의
# ---------------------------
SPACE_LABELS = [
    "a cafe",
    "a library",
    "a park",
    "a classroom",
    "a street",
    "a shopping mall",
    "a restaurant",
    "an office",
    "a restroom",
    "a room",
    "a university classroom",
    "a lecture room"
]

def get_text_tokens(tokenizer):
    return tokenizer(SPACE_LABELS).to(device)


# ---------------------------
# 3. 이미지 전처리
# ---------------------------
def image_to_tensor(image_bytes,preprocess):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

# ---------------------------
# 4. 공간 예측 함수
# ---------------------------
def predict_space(image_bytes, model, preprocess, tokenizer):
    image = image_to_tensor(image_bytes, preprocess)

    text_tokens = get_text_tokens(tokenizer)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).softmax(dim=-1)

        best_idx = similarity.argmax().item()
        best_score = similarity[0][best_idx].item()

        return {
            "space": SPACE_LABELS[best_idx],
            "confidence": round(best_score, 3)
        }

# ---------------------------
# 5. API 엔드포인트
# ---------------------------
@app.post("/detect-space")
async def detect_space(file: UploadFile = File(...)):

    model, preprocess, tokenizer = load_model()

    image_bytes = await file.read()

    result = predict_space(
        image_bytes,
        model,
        preprocess,
        tokenizer
    )

    return result




from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# DB 설정
# -------------------------
DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# -------------------------
# 테이블
# -------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    password = Column(String)


Base.metadata.create_all(bind=engine)


# -------------------------
# 입력 모델
# -------------------------
class AuthRequest(BaseModel):
    username: str
    password: str


# -------------------------
# 회원가입 (중복 방지 핵심)
# -------------------------
@app.post("/register")
def register(data: AuthRequest):
    db = SessionLocal()

    # ✔ 이미 있는 유저인지 확인
    existing_user = db.query(User).filter(User.username == data.username).first()

    if existing_user:
        return {
            "success": False,
            "msg": "이미 존재하는 아이디입니다"
        }

    # ✔ 없으면 생성
    user = User(
        username=data.username,
        password=data.password
    )

    db.add(user)
    db.commit()

    return {
        "success": True,
        "msg": "회원가입 성공"
    }


# -------------------------
# 로그인
# -------------------------
@app.post("/login")
def login(data: AuthRequest):
    db = SessionLocal()

    user = db.query(User).filter(User.username == data.username).first()

    if not user:
        return {"success": False, "msg": "아이디 없음"}

    if user.password != data.password:
        return {"success": False, "msg": "비밀번호 틀림"}

    return {"success": True, "msg": "로그인 성공"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = yolo_model(img)

    objects = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])

            name = yolo_model.names[cls_id]

            objects.append(name)

    return {
        "objects": list(set(objects))
    }