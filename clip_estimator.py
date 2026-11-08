# clip_estimator.py

import io
import os
import threading

import torch
import open_clip
from PIL import Image


# =========================================================
# 1. Device 설정
# =========================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# =========================================================
# 2. CLIP 모델 설정
# =========================================================

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# 현재 clip_estimator.py 파일 기준으로 모델 캐시 폴더 생성
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_CACHE_DIR = os.path.join(
    BASE_DIR,
    "models",
    "clip"
)

os.makedirs(CLIP_CACHE_DIR, exist_ok=True)


# =========================================================
# 3. 공간 라벨
# =========================================================

SPACE_LABELS = [
    "cafe",
    "restaurant",
    "office",
    "classroom",
    "park",
    "street",
    "kitchen",
    "living room",
    "bedroom",
    "beach",
    "airport",
    "zoo",
    "bathroom",
    "library",

    #CLIP 에만 있는 공간 
    "University lecture room",
    "Exhibition hall",
    "Hotel",
    "Subway station",
    "Hospital",
    "Convenience store",
    "Cinema",
    "Banks",
    "Pharmacy",
    "bar",
    "Pharmacy"
]


# =========================================================
# 4. 전역 모델 변수
# =========================================================

clip_model = None
clip_preprocess = None
clip_tokenizer = None
clip_text_features = None

# 동시에 여러 요청이 들어와도 모델을 한 번만 로드하기 위한 Lock
model_lock = threading.Lock()


# =========================================================
# 5. CLIP 모델 로드
# =========================================================

def load_model():
    """
    CLIP 모델과 SPACE_LABELS의 텍스트 특징 벡터를 최초 1회 로드합니다.

    반환:
        model, preprocess, tokenizer
    """

    global clip_model
    global clip_preprocess
    global clip_tokenizer
    global clip_text_features

    # 이미 로드되어 있으면 재사용
    if clip_model is not None:
        return (
            clip_model,
            clip_preprocess,
            clip_tokenizer
        )

    with model_lock:
        # Lock 획득 후 한 번 더 확인
        if clip_model is not None:
            return (
                clip_model,
                clip_preprocess,
                clip_tokenizer
            )

        print("=================================")
        print("[ CLIP 모델 로딩 시작 ]")
        print("Model:", CLIP_MODEL_NAME)
        print("Pretrained:", CLIP_PRETRAINED)
        print("Device:", DEVICE)
        print("Cache directory:", CLIP_CACHE_DIR)
        print("=================================")

        clip_model, _, clip_preprocess = (
            open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME,
                pretrained=CLIP_PRETRAINED,
                cache_dir=CLIP_CACHE_DIR
            )
        )

        clip_tokenizer = open_clip.get_tokenizer(
            CLIP_MODEL_NAME
        )

        clip_model = clip_model.to(DEVICE)
        clip_model.eval()

        # 공간 라벨 텍스트 특징을 미리 계산하여 이후 요청에서 재사용
        text_tokens = clip_tokenizer(
            SPACE_LABELS
        ).to(DEVICE)

        with torch.inference_mode():
            text_features = clip_model.encode_text(
                text_tokens
            )

            clip_text_features = text_features / (
                text_features.norm(
                    dim=-1,
                    keepdim=True
                )
            )

        print("=================================")
        print("clip_estimator: CLIP 모델 로딩 완료")
        print("=================================")

    return (
        clip_model,
        clip_preprocess,
        clip_tokenizer
    )


# =========================================================
# 6. 이미지 bytes → CLIP 입력 Tensor
# =========================================================

def image_to_tensor(image_bytes, preprocess):
    """
    업로드된 원본 이미지 bytes를 CLIP 입력 Tensor로 변환합니다.
    """

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = preprocess(
        image
    ).unsqueeze(0)

    return image_tensor.to(DEVICE)


# =========================================================
# 7. CLIP 공간 인식
# =========================================================

def predict_space(image_bytes):

    global clip_text_features

    if not image_bytes:
        raise ValueError("이미지 데이터가 없습니다.")

    model, preprocess, _ = load_model()

    image_tensor = image_to_tensor(
        image_bytes=image_bytes,
        preprocess=preprocess
    )

    with torch.inference_mode():
        # 이미지 특징 추출
        image_features = model.encode_image(
            image_tensor
        )

        # 이미지 특징 정규화
        image_features = image_features / (
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        # CLIP 권장 방식:
        # cosine similarity에 모델의 logit scale 적용
        logit_scale = model.logit_scale.exp()

        logits = logit_scale * (
            image_features @ clip_text_features.T
        )

        probabilities = logits.softmax(
            dim=-1
        )[0]

        # 높은 점수 순서
        sorted_indices = torch.argsort(
            probabilities,
            descending=True
        )

        best_idx = int(sorted_indices[0].item())
        best_score = float(probabilities[best_idx].item())

        second_score = (
            float(probabilities[int(sorted_indices[1].item())].item())
            if len(SPACE_LABELS) > 1
            else 0.0
        )



    scores = {}
    for idx in sorted_indices[:3]:
        i = int(idx.item())
        scores[SPACE_LABELS[i]] = round(float(probabilities[i].item()), 4)

    return {
        "space": SPACE_LABELS[best_idx],
        "confidence": round(best_score, 4),
        "margin": round(best_score - second_score, 4),
        "scores": scores
    }


# =========================================================
# 8. 간단한 테스트 실행
# =========================================================

if __name__ == "__main__":
    load_model()

    print("CLIP 모델 준비 완료")
    print("Device:", DEVICE)
