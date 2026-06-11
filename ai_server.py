import json
import re

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LORA_MODEL = "./finetuned_model_1"


app = FastAPI(title="Object Expression LLM Server")


class GenerateRequest(BaseModel):
    object: str


class GenerateResponse(BaseModel):
    object: str
    original_object: str
    mapping_status: str
    expressions: list[str]
    raw: str


LABEL_MAP = {
    "cellphone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "tvmonitor": "tv",
    "television": "tv",
    "diningtable": "dining table",
}


def load_yolo_mapping_rows(path: str = "yolo_dataset_mapping.json") -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


YOLO_MAPPING_ROWS = load_yolo_mapping_rows()


def build_yolo_dataset_map(rows: list[dict]) -> dict[str, str]:
    mapping = {}
    for row in rows:
        yolo_class = row.get("yolo_class")
        dataset_input = row.get("dataset_input")

        if yolo_class and dataset_input:
            mapping[yolo_class.strip().lower()] = dataset_input.strip().lower()

    return mapping


YOLO_DATASET_MAP = build_yolo_dataset_map(YOLO_MAPPING_ROWS)


def normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = LABEL_MAP.get(normalized, normalized)
    return YOLO_DATASET_MAP.get(normalized, normalized)


def get_mapping_status(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = LABEL_MAP.get(normalized, normalized)

    for row in YOLO_MAPPING_ROWS:
        if row.get("yolo_class") == normalized:
            return row.get("status", "unknown")

    return "unknown"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL)
model.eval()
print("LLM server is ready.")


def build_prompt(object_name: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{object_name}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_answer(full_text: str) -> str:
    answer = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    answer = answer.split("<|eot_id|>")[0]
    return answer.strip()


def split_expressions(answer: str) -> list[str]:
    return [line.strip() for line in answer.splitlines() if line.strip()]


def generate_expressions(object_name: str) -> str:
    prompt = build_prompt(object_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return extract_answer(full_text)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    original_object = req.object.strip()
    object_name = normalize_label(req.object)
    mapping_status = get_mapping_status(req.object)
    answer = generate_expressions(object_name)

    return {
        "object": object_name,
        "original_object": original_object,
        "mapping_status": mapping_status,
        "expressions": split_expressions(answer),
        "raw": answer,
    }

@app.get("/")
def root():
    return {
        "status": "LLM server running",
        "usage": "POST /generate with JSON {'object': 'pizza'}"
    }


@app.get("/generate")
def generate_get(object: str = "pizza"):
    answer = generate_expressions(object)

    return {
        "object": object,
        "original_object": object,
        "mapping_status": get_mapping_status(object),
        "expressions": split_expressions(answer),
        "raw": answer
    }