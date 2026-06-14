# Everywhere US - AI Object Expression Generation Module
Duksung Women's University _ 2026_Capstone_project_ 네박자

## Project Overview

객체 인식 결과를 기반으로 데이터셋 매핑을 수행한 후, LoRA 기반으로 파인튜닝된 Llama 3.2 3B 모델을 이용하여 해당 객체와 관련된 자연스러운 영어 표현 3개를 생성한다. 생성된 결과는 JSON 형태로 반환되어 모바일 애플리케이션 연동할 수 있다.

## Project Structure
### datasets/

파인튜닝에 사용된 학습 데이터셋을 저장하는 디렉터리입니다.

* 객체명과 관련된 영어 표현 데이터 저장
* JSON 형식의 Instruction Dataset 구성
* LoRA 기반 SFT(Supervised Fine-Tuning) 학습에 활용

---

### output/checkpoint-100/

모델 학습 과정에서 저장된 체크포인트 파일입니다.

* 학습된 LoRA 가중치 저장
* 100 Step 학습 시점 모델 상태 저장
* 추론 및 추가 학습 시 재사용 가능

---

### ai_server.py

FastAPI 기반 AI 추론 서버입니다.

#### 주요 기능

* 객체명 입력 수신
* YOLO 탐지 결과 매핑
* 파인튜닝된 Llama 3.2 3B 모델 로드
* 영어 표현 생성
* JSON 응답 반환

---

### train_sft_1.py

LoRA 기반 SFT 학습 스크립트입니다.

#### 주요 기능

* 데이터셋 로드 및 전처리
* Instruction 형식 학습 데이터 구성
* LoRA 어댑터 설정
* 모델 학습 수행

#### 학습 구성

* Base Model : Llama 3.2 3B Instruct
* Fine-Tuning : LoRA
* Framework : Transformers, TRL, PEFT
* Training Method : Supervised Fine-Tuning (SFT)

---

### yolo_dataset_mapping.json

YOLO 객체 클래스와 학습 데이터셋 객체명을 연결하는 매핑 파일입니다.

---

## System Workflow

```text
YOLO Object Detection
          ↓
Object Name Extraction
          ↓
Object Mapping
(yolo_dataset_mapping.json)
          ↓
Llama 3.2 3B + LoRA
          ↓
Expression Generation
          ↓
JSON Response
```


