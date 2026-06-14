# 2026 Capstone Project - AI Object Recognition Backend

## 📌 프로젝트 개요
본 프로젝트는 YOLO 기반 객체 인식과 LLM 기반 예문 생성을 결합한 AI 영어 학습 백엔드 시스템입니다.  
이미지에서 객체를 인식하고, 해당 단어를 AI 서버와 연동하여 영어 예문 3개를 생성합니다.

---

## 🧠 시스템 구조

```text
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
