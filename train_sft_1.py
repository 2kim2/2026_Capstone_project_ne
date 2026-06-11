import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import Dataset

# ─────────────────────────────────────────
# 1. 기본 설정
# ─────────────────────────────────────────
DATASET_DIR = r"C:\Users\DS\Desktop\ne\datasets"
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./finetuned_model_1"

# ─────────────────────────────────────────
# 2. JSON 데이터셋 로드
# ─────────────────────────────────────────
all_data = []

for filename in os.listdir(DATASET_DIR):
    if filename.endswith(".json"):
        filepath = os.path.join(DATASET_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)

            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

            print(f"로드 성공: {filename}")

        except Exception as e:
            print(f"로드 실패: {filename}")
            print(e)

print(f"\n총 {len(all_data)}개 항목 로드 완료")

# ─────────────────────────────────────────
# 3. 토크나이저 로드
# ─────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────
# 4. 단어 입력 → 데이터셋 답변 형식으로 변환
# ─────────────────────────────────────────
def format_prompt(item):
    word = item.get("input", "") or item.get("instruction", "")
    response = item.get("response", "")

    if isinstance(response, list):
        response = "\n".join(response)

    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{word}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}"
        "<|eot_id|>"
    )

dataset = Dataset.from_list(
    [{"text": format_prompt(item)} for item in all_data]
)

print(f"데이터셋 변환 완료: {dataset}")

# 학습 데이터 확인
print("\n===== 학습 데이터 예시 =====")
print(dataset[0]["text"])
print("==========================\n")

# ─────────────────────────────────────────
# 5. 모델 로드
# ─────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    dtype=torch.float16
)

model.config.use_cache = False

# ─────────────────────────────────────────
# 6. LoRA 설정
# ─────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ─────────────────────────────────────────
# 7. 학습 설정
# ─────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./output",

    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    fp16=True,

    logging_steps=10,
    save_steps=100,

    dataset_text_field="text",
    max_length=512,

    packing=False,
    report_to="none"
)

# ─────────────────────────────────────────
# 8. 학습 실행
# ─────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer,
)

trainer.train()

# ─────────────────────────────────────────
# 9. LoRA 모델 저장
# ─────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"학습 완료 및 LoRA 모델 저장 완료: {OUTPUT_DIR}")
