# 2026 Capstone Project - AI Object Recognition Backend

## 📌 프로젝트 개요
본 프로젝트는 YOLO 기반 객체 인식과 LLM 기반 예문 생성을 결합한 AI 영어 학습 백엔드 시스템입니다.  
이미지에서 객체를 인식하고, 해당 단어를 AI 서버와 연동하여 영어 예문 3개를 생성합니다.

---

## 🧠 시스템 구조
이미지 입력
↓
YOLOv8 객체 인식
↓
대표 객체 선택
↓
FastAPI Backend
↓
LLM API 서버 호출
↓
영어 예문 3개 생성
↓
프론트로 반환

---

## ⚙️ 사용 기술

- Python 3.10+
- FastAPI
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Requests
- SQLite (추후 확장 예정)
- Git / GitHub (버전 관리)

---

## 📡 API 목록

### 1. 객체 인식 API
POST /detect

#### 기능
이미지를 입력받아 객체를 탐지

#### 응답 예시
```json
{
  "objects": ["cup", "phone"]
}

POST /chat
기능

객체 단어를 AI 서버로 전달하여 예문 생성

응답 예시
{
  "object": "cup",
  "ai": {
    "expressions": [
      "This is a cup.",
      "I drink water from a cup.",
      "The cup is on the table."
    ]
  }
}

🔗 AI 서버 연동 구조
FastAPI 백엔드 → LLM 서버 (ngrok)
객체 단어를 JSON으로 전달
예문 3개 생성 후 반환

📂 프로젝트 구조
2026_Capstone_project_ne/
 ├ main.py
 ├ requirements.txt
 ├ README.md
 ├ .gitignore

🚀 실행 방법
pip install -r requirements.txt
uvicorn main:app --reload

🎯 주요 기능
YOLO 기반 실시간 객체 인식
단일 객체 중심 처리
LLM 기반 영어 예문 자동 생성
FastAPI 기반 REST API 구조

📌 향후 개선 계획
객체 인식 정확도 향상
응답 속도 최적화
모바일 프론트 연동 개선
다중 객체 확장 기능 추가

👨‍💻 개발 역할 (Backend)
YOLO 모델 통합
FastAPI 서버 구축
AI 서버 연동 API 개발
데이터 흐름 pipeline 설계
