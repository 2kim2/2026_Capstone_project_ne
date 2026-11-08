from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import open_clip


app = FastAPI()


# =========================================================
# 1. Device 설정
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=================================")
print("CLIP Server Started")
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("Device:", device)
print("=================================")


# =========================================================
# 2. CLIP 모델 변수
# =========================================================

clip_model = None
clip_preprocess = None
clip_tokenizer = None


# =========================================================
# 3. 공간 후보 정의
# =========================================================

SPACE_LABELS = [
    # 기존 YOLO 공간 인식 클래스
    "cafe",
    "restaurant",
    "office",
    "classroom",
    "park",
    "street",
    "kitchen",
    "living_room",
    "bedroom",
    "beach",
    "airport",
    "zoo",
    "bathroom",
    "library",

    # CLIP으로 추가 인식할 공간
    "university lecture room",
    "exhibition hall",
    "hotel",
    "subway station",
    "hospital"
]



# =========================================================
# 4. CLIP 모델 로드
# =========================================================

def load_model():

    global clip_model
    global clip_preprocess
    global clip_tokenizer

    # 이미 모델이 로드되어 있으면 다시 로드하지 않음
    if clip_model is None:

        print("CLIP 모델 로딩 중...")

        # 현재 open_clip 버전에서는
        # model, preprocess_train, preprocess_val
        # 총 3개를 반환함
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k"
        )

        # 텍스트 tokenizer
        clip_tokenizer = open_clip.get_tokenizer(
            "ViT-B-32"
        )

        # GPU 또는 CPU로 이동
        clip_model = clip_model.to(device)

        # 평가 모드
        clip_model.eval()

        print("CLIP 모델 로딩 완료")
        print("CLIP Device:", device)

    return (
        clip_model,
        clip_preprocess,
        clip_tokenizer
    )


# =========================================================
# 5. 텍스트 토큰 생성
# =========================================================

def get_text_tokens(tokenizer):

    text_tokens = tokenizer(
        SPACE_LABELS
    )

    return text_tokens.to(device)


# =========================================================
# 6. 이미지 전처리
# =========================================================

def image_to_tensor(
    image_bytes,
    preprocess
):

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0)

    return image_tensor.to(device)


# =========================================================
# 7. CLIP 공간 인식
# =========================================================

def predict_space(
    image_bytes,
    model,
    preprocess,
    tokenizer
):

    # -----------------------------------------
    # 이미지 전처리
    # -----------------------------------------

    image = image_to_tensor(
        image_bytes,
        preprocess
    )


    # -----------------------------------------
    # 공간 텍스트 생성
    # -----------------------------------------

    text_tokens = get_text_tokens(
        tokenizer
    )


    # -----------------------------------------
    # CLIP 추론
    # -----------------------------------------

    with torch.no_grad():

        # 이미지 특징
        image_features = model.encode_image(
            image
        )

        # 텍스트 특징
        text_features = model.encode_text(
            text_tokens
        )


        # -------------------------------------
        # 특징 벡터 정규화
        # -------------------------------------

        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )


        # -------------------------------------
        # 이미지와 텍스트 유사도 계산
        # -------------------------------------

        similarity = image_features @ text_features.T


        # -------------------------------------
        # Softmax
        # -------------------------------------

        probabilities = similarity.softmax(
            dim=-1
        )


        # -------------------------------------
        # 가장 높은 공간 선택
        # -------------------------------------

        best_idx = probabilities.argmax().item()

        best_score = probabilities[
            0,
            best_idx
        ].item()


    return {
        "space": SPACE_LABELS[best_idx],
        "confidence": round(
            best_score,
            4
        )
    }


# =========================================================
# 8. 공간 인식 API
# =========================================================

@app.post("/detect-space")
async def detect_space(
    file: UploadFile = File(...)
):

    try:

        # -------------------------------------
        # CLIP 모델 로드
        # -------------------------------------

        model, preprocess, tokenizer = load_model()


        # -------------------------------------
        # 이미지 읽기
        # -------------------------------------

        image_bytes = await file.read()


        # -------------------------------------
        # 빈 파일 확인
        # -------------------------------------

        if not image_bytes:

            return {
                "success": False,
                "message": "이미지가 없습니다."
            }


        # -------------------------------------
        # 공간 예측
        # -------------------------------------

        result = predict_space(
            image_bytes,
            model,
            preprocess,
            tokenizer
        )


        # -------------------------------------
        # 결과 반환
        # -------------------------------------

        return {
            "success": True,
            "space": result["space"],
            "confidence": result["confidence"]
        }


    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# =========================================================
# 9. 서버 상태 확인
# =========================================================

@app.get("/")
def root():

    return {
        "success": True,
        "message": "CLIP Space Recognition Server",
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }