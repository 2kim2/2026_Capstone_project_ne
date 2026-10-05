# qwen_local_chat.py (품사 기반 추출 + 문장 중요도 반영 추가 버전)

import json
import torch
import spacy
from typing import List, Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM

from fastapi import FastAPI
from pydantic import BaseModel

# 로컬 모델 경로 및 로딩 설정
model_dir = r"C:\Users\DS\Desktop\ne_yolo\feedback_server\qwen2.5_3b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda")

# =========================================
# FastAPI
# =========================================

app = FastAPI()


class ConversationRequest(BaseModel):
    conversation: List[Dict[str, Any]]


# 입력 경로
# input_path = r"C:\Users\DS\Desktop\ne_yolo\feedback_server\sample_talk.json"

# spaCy 영어 모델 로드
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

STOPWORDS = set("""
the a an and or but if in on with for to of is are was were be been being
this that those these and
""".split())

# def load_conversations(path: str) -> List[Dict[str, Any]]:
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     if "conversation" in data and isinstance(data["conversation"], list):
#         return data["conversation"]
#     raise ValueError("sample_talk.json should contain a 'conversation' list with {id, role, message}")

def build_dialogue_text(conversations: List[Dict[str, Any]]) -> str:
    lines = []
    for turn in conversations:
        role = str(turn.get("role", "")).lower()
        speaker = "A" if role in {"user", "customer"} else "B"
        text = str(turn.get("message", ""))
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)

def simple_keywords_from_sent(sent: str) -> List[str]:
    doc = nlp(sent)
    # 명사/동사 중심 키워드
    kws = []
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN", "VERB"} and not tok.is_stop:
            w = tok.lemma_.lower()
            if len(w) > 2:
                kws.append(w)
    # 중복 제거
    unique = []
    seen = set()
    for w in kws:
        if w not in seen:
            unique.append(w)
            seen.add(w)
    return unique[:5]  # 상위 5개로 제한

def generate_word(conversations: List[Dict[str, Any]], id_cutoff: int | None = None) -> List[Dict[str, Any]]:
    # 대화 텍스트를 문장 단위로 분리
    dialog_text = "\n".join([turn.get("message", "") for turn in conversations])
    sentences = [s.strip() for s in dialog_text.split("\n") if s.strip()]
    results = []
    for idx, sen in enumerate(sentences, start=1):
        kws = simple_keywords_from_sent(sen)
        if not kws:
            continue
        # 문장 점수에 기반한 필터링/정렬은 필요 시 apply
        results.append({"id": idx, "keywords": kws})
    # id_cutoff가 있다면 해당 범위 내에서 필터링(확장 가능)
    if id_cutoff is not None:
        results = [r for r in results if r["id"] <= id_cutoff]
    # 핵심 키워드가 없는 문장은 제거하고, 중복 제거
    unique_results = []
    seen_kw = set()
    for r in results:
        kws = [k for k in r["keywords"] if k not in seen_kw]
        if kws:
            unique_results.append({"id": r["id"], "keywords": kws})
            for k in kws:
                seen_kw.add(k)
    return unique_results

def generate_summary(dialog_text: str, model, tokenizer, max_tokens=180):
    prompt = (
        "Summarize the following cafe-style conversation about a project status update in 2-3 sentences. "
        "Keep it concise and neutral. Conversation:\n"
        f"{dialog_text}\n"
        "Summary:"
    )
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
            top_p=0.92,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("Summary:", 1)[-1].strip()

def generate_text(prompt: str, model, tokenizer, max_new_tokens=150):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    # 입력 프롬프트 부분은 제외하고 새로 생성된 부분만 decode
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return text


def paraphrase_with_explanation(sentence_en: str, model, tokenizer):

    # =========================================================
    # 1. 영어 패러프레이징
    # =========================================================

    prompt_para = f"""
Rewrite the following English sentence in natural, fluent English.

Rules:
- Keep exactly the same meaning.
- Do not add any new information.
- Do not remove important information.
- Return ONLY the rewritten English sentence.
- Do not write explanations.
- Do not add labels such as "To:", "Original:", "Rewritten:", "Paraphrase:".
- If the original sentence is already natural, you may keep it unchanged.

Original sentence:
{sentence_en}

Rewritten sentence:
"""

    paraphrase_en = generate_text(
        prompt_para,
        model,
        tokenizer,
        max_new_tokens=80
    )

    paraphrase_en = paraphrase_en.strip()

    # 불필요한 문구 제거
    for prefix in [
        "Rewritten sentence:",
        "Paraphrase:",
        "Original:",
        "To:"
    ]:
        if paraphrase_en.startswith(prefix):
            paraphrase_en = paraphrase_en[len(prefix):].strip()

    # 줄바꿈 이후 이상한 내용이 붙으면 첫 줄만 사용
    paraphrase_en = paraphrase_en.split("\n")[0].strip()

    # =========================================================
    # 2. 이상한 패러프레이징 결과 제거
    # =========================================================

    bad_patterns = [
        "to:",
        "original:",
        "rewritten:",
        "paraphrase:",
        "explanation:"
    ]

    if any(
        pattern in paraphrase_en.lower()
        for pattern in bad_patterns
    ):
        return None, None

    # =========================================================
    # 3. 패러프레이징 결과가 원문과 같으면 생략
    # =========================================================

    original_clean = " ".join(
        sentence_en.strip().lower().split()
    )

    paraphrase_clean = " ".join(
        paraphrase_en.strip().lower().split()
    )

    if original_clean == paraphrase_clean:
        return None, None

    # =========================================================
    # 4. 한국어 설명
    # =========================================================

    prompt_explanation = f"""
다음 영어 원문과 패러프레이징 문장을 비교해서
한국어로만 설명해 주세요.

[원문]
{sentence_en}

[패러프레이징]
{paraphrase_en}

설명 규칙:
- 반드시 한국어로 작성하세요.
- 2~3문장으로 간결하게 작성하세요.
- 원문에서 실제로 변경된 영어 표현을 구체적으로 설명하세요.
- 어떤 표현이 어떤 표현으로 바뀌었는지 설명하세요.
- 왜 변경된 표현이 더 자연스러운지 설명하세요.
- 원문의 의미가 어떻게 유지되었는지 설명하세요.
- 실제로 변경되지 않은 내용은 설명하지 마세요.
- 존재하지 않는 변경 사항을 만들어내지 마세요.
- 영어 표현을 언급할 때만 해당 표현을 " " 안에 표시하세요.
- "To:", "Original:", "Rewritten:" 등의 출력 형식은 설명하지 마세요.
- 영어 문장으로 설명하지 마세요.
- 제목이나 "Explanation:"을 출력하지 마세요.

한국어 설명:
"""

    explanation_ko = generate_text(
        prompt_explanation,
        model,
        tokenizer,
        max_new_tokens=100
    )

    explanation_ko = explanation_ko.strip()

    # 불필요한 제목 제거
    for prefix in [
        "한국어 설명:",
        "Explanation:",
        "설명:"
    ]:
        if explanation_ko.startswith(prefix):
            explanation_ko = explanation_ko[len(prefix):].strip()

    return paraphrase_en, explanation_ko


# def main():
#     conversations = load_conversations(input_path)
#     dialog_text = build_dialogue_text(conversations)
#     summary = generate_summary(dialog_text, model, tokenizer, max_tokens=180)

#     keywords_by_sentence = generate_word(conversations, id_cutoff=None)


#     # 예: 발화 단위로 패러프레이징 및 설명 생성
#     paraphrase_results = []
#     for turn in conversations:
#         if turn.get("role", "").lower() != "user":
#             continue
#         original_en = turn.get("message", "")
#         paraphrase_en, explanation_ko = paraphrase_with_explanation(original_en, model, tokenizer)
            
#         # 패러프레이징할 필요가 없으면 생략
#         if paraphrase_en is None:
#             continue
#         paraphrase_results.append({
#             "id": turn.get("id"),
#             "paraphrase_en": paraphrase_en,
#             "explanation_ko": explanation_ko
#         })

#     # 기존 출력과 결합 예시
#     start_id = conversations[0]["id"]
#     end_id = conversations[-1]["id"]
#     output = {
#         "dialog_start_id": start_id,
#         "dialog_end_id": end_id,
#         "summary": summary,
#         # "dialogue_text": dialog_text,
#         "paraphrase_by_user": paraphrase_results
#     }

#     print("=== Summary Output ===")
#     print(json.dumps([output], ensure_ascii=False, indent=2))

# if __name__ == "__main__":
#     main()

@app.post("/feedback")
def feedback(request: ConversationRequest):

    conversations = request.conversation

    if not conversations:
        return {
            "error": "conversation is empty"
        }

    # ==========================================
    # 1. 대화 전체 텍스트 생성
    # ==========================================
    dialog_text = build_dialogue_text(conversations)

    # ==========================================
    # 2. 대화 요약
    # ==========================================
    summary = generate_summary(
        dialog_text,
        model,
        tokenizer,
        max_tokens=180
    )

    # ==========================================
    # 3. 사용자 발화 패러프레이징
    # ==========================================
    paraphrase_results = []

    for turn in conversations:

        if turn.get("role", "").lower() != "user":
            continue

        original_en = turn.get("message", "").strip()

        if not original_en:
            continue

        paraphrase_en, explanation_ko = paraphrase_with_explanation(
            original_en,
            model,
            tokenizer
        )

        # 패러프레이징할 필요가 없는 경우
        if paraphrase_en is None:
            continue

        paraphrase_results.append({
            "id": turn.get("id"),
            "original_en": original_en,
            "paraphrase_en": paraphrase_en,
            "explanation_ko": explanation_ko
        })

    # ==========================================
    # 4. 결과 반환
    # ==========================================
    return {
        "dialog_start_id": conversations[0].get("id"),
        "dialog_end_id": conversations[-1].get("id"),
        "summary": summary,
        "paraphrase_by_user": paraphrase_results
    }