# qwen_local_chat.py  (v4: 속도 개선판)
#
# v3 대비 바뀐 점 (품질 로직은 그대로)
#   1) 배치 생성: 문장마다 따로 generate 하던 것을, 모든 문장을 한 번에 generate
#   2) 단계 병합: 요약 + 번역 + 패러프레이즈 + 단어 뜻을 같은 배치로 동시에 생성
#   3) 재시도는 검사에 걸린 것만 모아서 한 배치로
#   4) max_new_tokens를 입력 길이에 맞게 자동 조절
#   5) 단계별 소요 시간 로그 ([time] ...)
#
# 필요 패키지: pip install spacy wordfreq && python -m spacy download en_core_web_sm
 
import difflib
import json
import re
import time
import torch
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
 
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from fastapi import FastAPI
from pydantic import BaseModel
 
# =========================================
# 설정
# =========================================
# 7B로 바꿀 때는 경로만 바꾸면 됩니다 (예: ...\qwen2.5_7b). 아래 load_in_4bit 참고.
model_dir = r"C:\Users\DS\Desktop\ne_yolo\feedback_server\qwen2.5_3b"
LOAD_IN_4BIT = False         # 7B를 8~12GB GPU에서 돌릴 때 True (pip install bitsandbytes)
 
MAX_KEYWORDS = 5
KEYWORD_MAX_ZIPF = 4.5       # 이 값보다 흔한 단어는 "쉬운 단어"로 보고 제외 (낮출수록 어려운 단어만)
PHRASE_MAX_ZIPF = 4.8        # 복합명사/구동사(credit card, pick up)용 기준
BATCH_SIZE = 8               # 한 번에 생성할 프롬프트 수. GPU 메모리가 부족하면 4로 낮추세요
RESTRICT_CHARS = True        # 생성 단계에서 한글/영어/기본 문장부호 외 문자(일본어, 중국어, 태국어 등)를 막음
DEBUG_REJECTS = True         # 검사에서 버려진 모델 출력을 콘솔에 출력 (원인 파악용)
 
# 용어집: 3B 모델이 자주 틀리는 메뉴/외래어를 고정 (필요한 만큼 추가하세요)
#   - 번역할 때 문장에 있는 용어를 힌트로 넣고, 번역문에 빠지면 재시도
#   - 단어 뜻은 모델에게 묻지 않고 여기 값을 그대로 씀 (복합어는 부분을 조합: chocolate muffin → 초콜릿 머핀)
GLOSSARY = {
    "oat milk": "오트밀크(귀리 우유)", "almond milk": "아몬드 우유", "soy milk": "두유",
    "whipped cream": "휘핑크림", "espresso shot": "에스프레소 샷", "decaf": "디카페인",
    "latte": "라테", "americano": "아메리카노", "macchiato": "마키아토", "cappuccino": "카푸치노",
    "espresso": "에스프레소", "mocha": "모카", "muffin": "머핀", "bagel": "베이글", "croissant": "크루아상",
    "scone": "스콘", "cheesecake": "치즈케이크", "cheese cake": "치즈케이크", "cookie": "쿠키",
    "chocolate": "초콜릿", "vanilla": "바닐라", "caramel": "카라멜", "hazelnut": "헤이즐넛",
    "receipt": "영수증", "refund": "환불", "credit card": "신용카드",
    "warm up": "데우다", "heat up": "데우다",
}
GLOSSARY_NO_CHECK = {"warm up", "heat up"}   # 번역문에 그대로 안 나와도 되는 항목 (동사 등)
 
# =========================================
# 모델 로딩
# =========================================
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.padding_side = "left"          # 배치 생성은 왼쪽 패딩이어야 함
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
_model_kwargs = dict(trust_remote_code=True, device_map="cuda", attn_implementation="sdpa")
if LOAD_IN_4BIT:
    from transformers import BitsAndBytesConfig
    _model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
else:
    _model_kwargs["torch_dtype"] = torch.float16
model = AutoModelForCausalLM.from_pretrained(model_dir, **_model_kwargs)
model.eval()
 
# =========================================
# 허용 문자: ASCII(영어, 숫자, 기본 기호) + 한글 + 일부 문장부호/화살표
#   이 밖의 문자(テ, อ, 中 등)는 생성 단계에서 막고, 검사 단계에서도 걸러냄
# =========================================
FOREIGN_CHAR_RE = re.compile(
    r"[^\u0000-\u007F"          # ASCII
    r"\uAC00-\uD7A3"            # 한글 음절
    r"\u3131-\u318E"            # 한글 자모 (ㅋ, ㅠ 등)
    r"\u2010-\u2027"            # ‘ ’ “ ” … – — 등
    r"\u2190-\u21FF"            # → 등 화살표
    r"\u00A0-\u00FF"            # 라틴-1 기호와 악센트 글자 (·, ×, é 등)
    r"\s]"
)
 
 
def build_char_blocker():
    """허용 문자 밖의 글자를 포함한 토큰을 미리 찾아 두고, 생성 중에 확률을 0으로 만든다."""
    t = time.perf_counter()
    bad = []
    special = set(tokenizer.all_special_ids)
    for tid in range(len(tokenizer)):
        if tid in special:
            continue
        text = tokenizer.decode([tid]).replace("\ufffd", "")   # 바이트 조각(한글 일부일 수 있음)은 허용
        if text and FOREIGN_CHAR_RE.search(text):
            bad.append(tid)
    bad_ids = torch.tensor(bad, dtype=torch.long, device=model.device)
    print(f"[info] 외국 문자 토큰 {len(bad)}개 차단 준비 ({time.perf_counter() - t:.1f}s)")
 
    class BlockForeignChars(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:, bad_ids] = -float("inf")
            return scores
 
    return LogitsProcessorList([BlockForeignChars()])
 
 
char_blocker = build_char_blocker() if RESTRICT_CHARS else None
 
# =========================================
# spaCy / wordfreq (없어도 돌아가지만 단어 품질이 떨어짐)
# =========================================
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print("[info] spaCy 로드 완료")
except Exception as e:
    nlp = None
    print("=" * 60)
    print(f"[경고] spaCy를 불러오지 못했습니다: {e}")
    print("  단어 추출이 단순 규칙으로 동작합니다. 아래 명령으로 설치하세요.")
    print("  pip install spacy && python -m spacy download en_core_web_sm")
    print("=" * 60)
 
try:
    from wordfreq import zipf_frequency
except Exception:
    zipf_frequency = None
    print("[경고] wordfreq가 없어 단어 난이도 필터가 꺼집니다. pip install wordfreq")
 
# 기능어 + 학습 가치가 낮은 기초 단어 (spaCy 유무와 상관없이 항상 제외)
FUNCTION_WORDS = set("""
a an the and or but if so because as at by for from in into of off on onto out over to up with
about after before than then there here this that these those it its i me my mine you your yours
he him his she her hers we us our they them their what which who whom whose where when why how
is am are was were be been being do does did done have has had having will would shall should
can could may might must not no yes all any some each every much many more most less few
one two three very really just also too only even still please okay ok oh hi hello sorry thank thanks
""".split())
 
BASIC_WORDS = {
    "get", "go", "come", "make", "take", "want", "like", "know", "think", "say", "tell",
    "see", "look", "let", "need", "give", "put", "thing", "something", "anything",
    "everything", "way", "lot", "time", "good", "bad", "nice", "great", "mean",
}
 
# =========================================
# FastAPI
# =========================================
app = FastAPI()
 
 
class ConversationRequest(BaseModel):
    conversation: List[Dict[str, Any]]
 
 
# =========================================
# 프롬프트 (few-shot은 messages로 넣음)
# =========================================
SUMMARY_SYSTEM_PROMPT = (
    "You summarize English conversation practice sessions between a learner (A) and a staff member or tutor (B). "
    "Write a concise, neutral summary in 2-3 English sentences. "
    "Output only the summary, with no title or label."
)
 
TRANSLATE_SYSTEM_PROMPT = (
    "You are a professional English-to-Korean translator. "
    "Translate the English text into natural, everyday spoken Korean, the way a Korean person would really say it.\n"
    "Rules:\n"
    "- Translate every sentence, including short ones like \"Thank you.\" or \"For here.\"\n"
    "- Write the whole answer in Hangul. Do not leave any English letters. "
    "Write menu items and loanwords the way Koreans spell them (vanilla latte -> 바닐라 라테, muffin -> 머핀).\n"
    "- Keep quantities exactly as in the original. \"a\" or \"one\" means one (한 잔, 하나). Never add a number.\n"
    "- Output only the translation. Do not answer, explain, or comment on the text."
)
TRANSLATE_SHOTS = [
    ("Hi, can I get two hot lattes, please?",
     "안녕하세요, 따뜻한 라테 두 잔 주시겠어요?"),
    ("Can I get an iced caramel macchiato and a blueberry bagel?",
     "아이스 카라멜 마키아토 한 잔이랑 블루베리 베이글 하나 주시겠어요?"),
    ("Large, please. Oh, and no whipped cream. Actually, extra whipped cream.",
     "라지로 주세요. 아, 휘핑크림은 빼 주세요. 아니다, 휘핑크림 많이 올려 주세요."),
    ("To go. Thank you. Is there a restroom here?",
     "포장이요. 감사합니다. 여기 화장실 있나요?"),
    ("No, that's it for now. Soy milk, please.",
     "아니요, 지금은 그게 다예요. 두유로 주세요."),
]
 
PARAPHRASE_SYSTEM_PROMPT = (
    "You are an English tutor. Rewrite the learner's English so it sounds natural and correct, "
    "like a native speaker in the same situation.\n"
    "Rules:\n"
    "- Keep EVERY sentence and every piece of information, including short phrases like \"Thank you.\" or \"For here.\"\n"
    "- Fix grammar and unnatural word choices. Change as little as needed.\n"
    "- Do not replace correct expressions with synonyms. Keep set phrases as they are "
    "(that's it, for here, to go, pay by card), because changing them changes the meaning.\n"
    "- Do not add new information.\n"
    "- If the text is already natural, return it exactly as it is.\n"
    "- Output only the rewritten text."
)
PARAPHRASE_SHOTS = [
    ("Hi, I want two hot latte.",
     "Hi, I'd like two hot lattes."),
    ("Large. And can I get it no sugar? Sorry, I mean less sugar.",
     "Large, please. And can I get it without sugar? Sorry, I mean less sugar."),
    ("To go. Thank you. Where is restroom?",
     "To go, please. Thank you. Where is the restroom?"),
    ("Can I have the receipt, please?",
     "Can I have the receipt, please?"),
    ("No, that's all. For here, please.",
     "No, that's all. For here, please."),
    ("Can I pay by card? Thanks a lot!",
     "Can I pay by card? Thanks a lot!"),
    ("Done. Thanks so much!",
     "Done. Thanks so much!"),
]
 
EXPLAIN_SYSTEM_PROMPT = (
    "당신은 한국인 학습자를 가르치는 영어 선생님입니다.\n"
    "원문, 고친 문장, 그리고 실제로 바뀐 부분 목록이 주어집니다.\n"
    "규칙:\n"
    "- 반드시 한국어로만 쓰세요. 영어는 따옴표 안의 표현에만 쓰세요.\n"
    "- '바뀐 부분' 목록에 있는 변경만 설명하세요. 목록에 없는 변경은 절대 만들지 마세요.\n"
    "- 각 변경마다 왜 더 자연스러운지(문법, 단어 선택, 공손함) 짧게 설명하세요.\n"
    "- 전체 2~3문장으로 쓰고, 제목이나 번호는 붙이지 마세요."
)
EXPLAIN_SHOTS = [
    ("원문: Hi, I want two hot latte.\n"
     "고친 문장: Hi, I'd like two hot lattes.\n"
     "바뀐 부분:\n- \"want\" → \"'d like\"\n- \"latte\" → \"lattes\"",
     "\"want\"보다 \"I'd like\"가 주문할 때 더 공손하고 자연스러운 표현입니다. "
     "또 두 잔이므로 \"latte\"를 복수형 \"lattes\"로 써야 합니다."),
    ("원문: Where is restroom?\n"
     "고친 문장: Where is the restroom?\n"
     "바뀐 부분:\n- (추가) \"the\"",
     "\"restroom\"처럼 특정한 장소를 물을 때는 앞에 관사 \"the\"를 붙여야 문법적으로 맞습니다."),
]
 
MEANING_SYSTEM_PROMPT = (
    "You are an English-Korean dictionary. For each English word or phrase, give its short Korean dictionary "
    "meaning as it is used in the given sentence. Write the meaning in Hangul only (1-6 words), never in English. "
    "Verbs end with \"-다\" (e.g. 결제하다). "
    "For loanwords that Koreans also use (menu items, brands, foods), write the Hangul transliteration, "
    "and add a short Korean gloss in parentheses only when the loanword is hard to understand "
    "(e.g. oat milk -> 오트밀크(귀리 우유)). Never put another spelling in the parentheses. "
    "Return ONLY a JSON object mapping each word to its meaning."
)
MEANING_SHOTS = [
    ('Sentence: Can I get a refund for this receipt?\nWords: ["refund", "receipt"]',
     '{"refund": "환불", "receipt": "영수증"}'),
    ('Sentence: I booked a table near the window.\nWords: ["book", "window"]',
     '{"book": "예약하다", "window": "창문"}'),
    ('Sentence: Could you heat up this muffin?\nWords: ["heat up", "muffin"]',
     '{"heat up": "데우다", "muffin": "머핀"}'),
    ('Sentence: Can I get a latte with oat milk and an extra espresso shot?\nWords: ["oat milk", "espresso shot"]',
     '{"oat milk": "오트밀크(귀리 우유)", "espresso shot": "에스프레소 샷"}'),
    ('Sentence: One iced americano and a croissant, please.\nWords: ["iced americano", "croissant"]',
     '{"iced americano": "아이스 아메리카노", "croissant": "크루아상"}'),
    ('Sentence: A vanilla latte and a chocolate cookie, please.\nWords: ["vanilla latte", "chocolate cookie"]',
     '{"vanilla latte": "바닐라 라테", "chocolate cookie": "초콜릿 쿠키"}'),
]
 
 
# =========================================
# 생성 함수 (배치)
#   여러 프롬프트를 한 번의 model.generate로 동시에 생성합니다.
#   GPU는 1개를 만들든 8개를 만들든 걸리는 시간이 거의 같아서, 문장 수만큼 빨라집니다.
# =========================================
def make_job(system_prompt: str, user_content: str,
             shots: Optional[List[Tuple[str, str]]] = None,
             extra: Optional[List[Dict[str, str]]] = None,
             max_new_tokens: int = 128) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in (shots or []):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})
    messages.extend(extra or [])
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt": prompt, "max_new_tokens": max_new_tokens}
 
 
def generate_batch(jobs: List[Dict[str, Any]]) -> List[str]:
    if not jobs:
        return []
    outputs = [""] * len(jobs)
    # 길이가 비슷한 것끼리 묶어야 패딩 낭비가 적음
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["prompt"]))
    for c in range(0, len(order), BATCH_SIZE):
        idx = order[c:c + BATCH_SIZE]
        enc = tokenizer([jobs[i]["prompt"] for i in idx], return_tensors="pt", padding=True).to(model.device)
        start = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max(jobs[i]["max_new_tokens"] for i in idx),
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=char_blocker
            )
        for k, i in enumerate(idx):
            outputs[i] = tokenizer.decode(out[k][start:], skip_special_tokens=True).strip()
    return outputs
 
 
def tokens_for(text: str, per_word: int = 4, base: int = 24, cap: int = 200) -> int:
    """입력 길이에 맞춘 max_new_tokens (짧은 문장에 200토큰씩 잡지 않도록)."""
    return min(cap, base + per_word * len(text.split()))
 
 
@contextmanager
def timer(name: str, log: Dict[str, float]):
    t = time.perf_counter()
    yield
    log[name] = round(time.perf_counter() - t, 2)
    print(f"[time] {name}: {log[name]:.2f}s")
 
 
# =========================================
# 텍스트 유틸
# =========================================
HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z']+")
 
 
def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if WORD_RE.search(p) or HANGUL_RE.search(p)]
 
 
def words(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]
 
 
def hangul_ratio(text: str, ignore_quoted: bool = False) -> float:
    if ignore_quoted:
        text = re.sub(r"\"[^\"]*\"|'[^']*'|“[^”]*”", "", text)
    h = len(HANGUL_RE.findall(text))
    l = len(LATIN_RE.findall(text))
    return h / (h + l) if (h + l) else 0.0
 
 
def clean_output(text: str, prefixes: List[str]) -> str:
    text = text.strip().strip("`").strip()
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()
    text = text.strip()
    # 전체가 따옴표 한 쌍으로 감싸진 경우만 벗김 ('"with"보다 "by"가…' 같은 설명은 건드리지 않음)
    if len(text) >= 2 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1].strip()
    return text
 
 
def normalize(s: str) -> str:
    return " ".join(words(s))
 
 
# =========================================
# 요약
# =========================================
def build_dialogue_text(conversations: List[Dict[str, Any]]) -> str:
    lines = []
    for turn in conversations:
        role = str(turn.get("role", "")).lower()
        speaker = "A" if role in {"user", "customer"} else "B"
        lines.append(f"{speaker}: {turn.get('message', '')}")
    return "\n".join(lines)
 
 
# =========================================
# 1) 번역
# =========================================
# 번역문에 끼어든 메타 문구 ("원래의 문장이 하나예요", "번역:" 등)
META_RE = re.compile(r"원래|원문|문장|번역|translation|sentence", re.IGNORECASE)
KO_COUNT_RE = re.compile(r"(두|세|네|다섯|여섯)\s?(개|잔|조각|명|병|장|그릇|판)")
EN_PLURAL_COUNT_RE = re.compile(r"\b(two|three|four|five|six|couple|[2-9])\b", re.IGNORECASE)
 
 
def translation_problems(src: str, ko: str) -> List[str]:
    """번역문의 문제 목록. 빈 리스트면 통과."""
    if not ko:
        return ["empty"]
    problems = []
    if FOREIGN_CHAR_RE.search(ko):
        problems.append("foreign")
    # 너무 짧음: 한글 글자 수가 원문 영어 단어 수보다 적으면 일부만 번역한 것 ("머핀")
    if len(HANGUL_RE.findall(ko)) < max(1, int(0.8 * len(words(src)))):
        problems.append("too_short")
    if LATIN_RE.search(ko):
        problems.append("english")
    if META_RE.search(ko) and not META_RE.search(src):
        problems.append("meta")
    if KO_COUNT_RE.search(ko) and not EN_PLURAL_COUNT_RE.search(src):
        problems.append("count")
    if len(split_sentences(ko)) < len(split_sentences(src)):
        problems.append("missing")
    for en, k in glossary_hits(src).items():
        if en not in GLOSSARY_NO_CHECK and k.replace(" ", "") not in ko.replace(" ", ""):
            problems.append("term")          # "오트밀크"를 "오트밀"로 쓴 경우 등
            break
    return problems
 
 
def fix_added_counts(src: str, ko: str) -> str:
    """원문에 없는 수량("두 개")은 코드로 지움: "초콜릿 머핀 두 개 주세요" → "초콜릿 머핀 주세요" """
    if ko and not EN_PLURAL_COUNT_RE.search(src):
        ko = re.sub(r"\s{2,}", " ", KO_COUNT_RE.sub("", ko)).replace(" ?", "?").replace(" .", ".").strip()
    return ko
 
 
# =========================================
# 2) 패러프레이즈
# =========================================
def paraphrase_problem(src: str, para: str) -> Optional[str]:
    """문제가 있으면 재시도용 피드백 문장을, 없으면 None을 돌려준다."""
    if not para or hangul_ratio(para) > 0.1:
        return "Write the rewritten text in English only."
    para_words = set(words(para))
    for sent in split_sentences(src):
        sw = words(sent)
        if sw and sum(w in para_words for w in sw) / len(sw) < 0.5:
            return f'You removed "{sent}". Keep every sentence of the original.'
    ratio = difflib.SequenceMatcher(None, words(src), words(para)).ratio()
    if ratio < 0.5:
        return "You changed too much. Keep the original structure and change only what is wrong or unnatural."
    return None
 
 
# =========================================
# 3) 설명 (실제 diff 기반)
# =========================================
def word_diff(src: str, para: str) -> List[str]:
    a, b = src.split(), para.split()
    strip = lambda t: t.strip(".,!?;:\"").lower()
    sm = difflib.SequenceMatcher(None, [strip(t) for t in a], [strip(t) for t in b])
    changes = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        old = " ".join(t.strip(".,!?;:\"") for t in a[i1:i2])
        new = " ".join(t.strip(".,!?;:\"") for t in b[j1:j2])
        if op == "replace":
            changes.append(f'"{old}" → "{new}"')
        elif op == "delete":
            changes.append(f'(삭제) "{old}"')
        elif op == "insert":
            changes.append(f'(추가) "{new}"')
    return changes
 
 
def template_explanation(changes: List[str]) -> str:
    return "다음 부분을 더 자연스럽게 고쳤습니다: " + ", ".join(changes) + "."
 
 
# =========================================
# 4) 핵심 단어
# =========================================
def is_hard_enough(word: str, is_phrase: bool) -> bool:
    if zipf_frequency is None:
        return True
    return zipf_frequency(word, "en") <= (PHRASE_MAX_ZIPF if is_phrase else KEYWORD_MAX_ZIPF)
 
 
def _accept(word: str, seen: set) -> bool:
    return (len(word) > 2 and word not in seen
            and word not in FUNCTION_WORDS and word not in BASIC_WORDS)
 
 
def extract_keywords(sentence_en: str) -> List[str]:
    found: List[str] = []
    seen: set = set()
 
    if nlp is None:
        # 단순 규칙 (spaCy 없음)
        for w in words(sentence_en):
            if _accept(w, seen) and is_hard_enough(w, False):
                found.append(w)
                seen.add(w)
        return found[:MAX_KEYWORDS]
 
    doc = nlp(sentence_en)
    used = set()
 
    # 복합명사 (credit card)
    for tok in doc:
        if tok.dep_ == "compound" and tok.head.pos_ in {"NOUN", "PROPN"} and tok.i < tok.head.i:
            span = doc[tok.i: tok.head.i + 1]
            if all(t.pos_ in {"NOUN", "PROPN"} for t in span):
                phrase = " ".join(t.lemma_.lower() for t in span)
                used.update(t.i for t in span)
                if phrase not in seen and is_hard_enough(phrase, True):
                    found.append(phrase)
                    seen.add(phrase)
 
    # 단일 단어 / 구동사
    for tok in doc:
        if tok.i in used or tok.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}:
            continue
        if tok.is_stop or not tok.is_alpha:
            continue
        lemma = tok.lemma_.lower()
        if tok.dep_ == "amod" and tok.tag_ in {"VBN", "VBG"}:
            lemma = tok.text.lower()
        particles = [c.text.lower() for c in tok.children if c.dep_ == "prt"]
        if tok.pos_ == "VERB" and particles:
            phrase = f"{lemma} {' '.join(particles)}"
            if phrase not in seen and is_hard_enough(phrase, True):
                found.append(phrase)
                seen.add(phrase)
            continue
        if _accept(lemma, seen) and is_hard_enough(lemma, False):
            found.append(lemma)
            seen.add(lemma)
 
    # 어려운 단어 우선 정렬
    if zipf_frequency is not None:
        found.sort(key=lambda w: zipf_frequency(w, "en"))
    return found[:MAX_KEYWORDS]
 
 
def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
 
 
def glossary_meaning(word: str) -> Optional[str]:
    """용어집 그대로, 또는 복합어를 부분 조합 (chocolate muffin → 초콜릿 머핀)."""
    word = word.lower()
    if word in GLOSSARY:
        return GLOSSARY[word]
    parts = word.split()
    if len(parts) > 1 and all(p in GLOSSARY for p in parts):
        return " ".join(main_form(GLOSSARY[p]) for p in parts)
    return None
 
 
def fix_verb_form(m: str) -> str:
    """동사 뜻을 사전형으로: "따뜻하게 해다" → "따뜻하게 하다", "결제해요" → "결제하다" """
    m = re.sub(r"해다$", "하다", m)
    m = re.sub(r"(해요|합니다|해주다|해 주다)$", "하다", m)
    return m
 
 
def valid_meaning(m: Any) -> Optional[str]:
    """한글 뜻만 통과. "오트밀크(귀리 우유)"처럼 괄호 설명이 붙은 형식도 허용."""
    if not isinstance(m, str):
        return None
    m = m.strip().strip(".").strip()
    if not m or len(m) > 30:
        return None
    # 영어 글자가 하나라도 섞이면 거절 ("oat milk", "오트 milk" 등)
    if LATIN_RE.search(m) or FOREIGN_CHAR_RE.search(m) or not HANGUL_RE.search(m):
        return None
    # 괄호 안이 같은 말의 다른 표기면 삭제: "초콜릿 머핀(초콜릿 무핀)" → "초콜릿 머핀"
    paren = re.match(r"^(.*?)\s*\((.*)\)$", m)
    if paren:
        main, gloss = paren.group(1).strip(), paren.group(2).strip()
        if not main:
            return None
        if difflib.SequenceMatcher(None, main, gloss).ratio() >= 0.5:
            m = main
    m = fix_verb_form(m)
    if re.search(r"(요|니다|세요)$", m):         # 사전형이 아닌 문장형 답("데워요")은 버리고 재시도
        return None
    return m
 
 
def _key(word: Any) -> str:
    # "Oat Milk", "oat_milk", "oat-milk" 모두 "oat milk"로 맞춤
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(word).lower()).split())
 
 
# =========================================
# 파이프라인 (단계별로 모든 문장을 한꺼번에 처리)
#   1단계: 요약 + 번역 + 패러프레이즈 + 단어 뜻   (서로 의존하지 않으니 한 배치)
#   2단계: 1단계에서 검사에 걸린 것만 재시도
#   3단계: 설명 (최종 패러프레이즈가 정해진 뒤)
#   4단계: 설명 중 영어로 나온 것만 재시도
# =========================================
TR_PREFIXES = ["번역:", "Translation:", "Korean:"]
PARA_PREFIXES = ["Rewritten:", "Paraphrase:", "Rewritten text:"]
EXP_PREFIXES = ["설명:", "Explanation:"]
MEANING_RETRY_MSG = ("Write every meaning in Hangul only. For loanwords, write how Koreans spell it in Hangul "
                     "(e.g. oat milk -> 오트밀크(귀리 우유)). Return the JSON object.")
EXPLAIN_RETRY_MSG = "영어가 아니라 한국어로 다시 설명해 주세요. 영어 표현은 따옴표 안에만 쓰세요."
 
 
def glossary_hits(text: str) -> Dict[str, str]:
    """문장에 나온 용어집 항목. 긴 항목 우선 (oat milk가 있으면 milk 단독은 무시)."""
    low, hits = text.lower(), {}
    for en in sorted(GLOSSARY, key=len, reverse=True):
        if re.search(rf"\b{re.escape(en)}s?\b", low) and not any(en in h for h in hits):
            hits[en] = main_form(GLOSSARY[en])
    return hits
 
 
def main_form(ko: str) -> str:
    return re.sub(r"\(.*\)$", "", ko).strip()      # "오트밀크(귀리 우유)" → "오트밀크"
 
 
def translate_job(text: str, extra=None, system: str = TRANSLATE_SYSTEM_PROMPT):
    hits = glossary_hits(text)
    if hits:
        system += "\n- Use these Korean words: " + ", ".join(f"{en} = {ko}" for en, ko in hits.items()) + "."
    return make_job(system, text, TRANSLATE_SHOTS, extra, tokens_for(text, 4, 24, 160))
 
 
def paraphrase_job(text: str, extra=None):
    return make_job(PARAPHRASE_SYSTEM_PROMPT, text, PARAPHRASE_SHOTS, extra, tokens_for(text, 2, 16, 120))
 
 
def meaning_job(sentence: str, kws: List[str], extra=None):
    user = f"Sentence: {sentence}\nWords: {json.dumps(kws, ensure_ascii=False)}"
    return make_job(MEANING_SYSTEM_PROMPT, user, MEANING_SHOTS, extra, 20 + 16 * len(kws))
 
 
def explain_user(src: str, para: str, changes: List[str]) -> str:
    return f"원문: {src}\n고친 문장: {para}\n바뀐 부분:\n" + "\n".join(f"- {c}" for c in changes)
 
 
def parse_meanings(raw: str, kws: List[str]) -> Dict[str, Optional[str]]:
    data = {_key(k): v for k, v in (parse_json_object(raw) or {}).items()}
    result = {k: valid_meaning(data.get(_key(k))) for k in kws}
    if DEBUG_REJECTS and any(v is None for v in result.values()):
        print(f"[reject] meaning {kws} <- {raw!r}")
    return result
 
 
def translation_retry_system(ko: str, problems: List[str]) -> str:
    """재시도 힌트는 대화로 덧붙이지 않고 시스템 프롬프트에 넣음.
    (대화로 "Write these in Hangul: muffin"을 보내면 모델이 그 단어만 번역해서 "머핀"만 돌려줌)"""
    hints = []
    if "english" in problems:
        eng = sorted(set(re.findall(r"[A-Za-z]+", ko)))
        hints.append("Also write these words in Hangul: " + ", ".join(eng) + ".")
    if "count" in problems:
        hints.append("Do not add numbers that are not in the original.")
    if "too_short" in problems:
        hints.append("Translate the whole text, not just one word.")
    if "term" in problems:
        hints.append("You must use the Korean words given above exactly.")
    return TRANSLATE_SYSTEM_PROMPT + "\n- " + " ".join(hints) if hints else TRANSLATE_SYSTEM_PROMPT
 
 
# 문제의 심각도. 둘 다 문제가 있을 때 덜 심각한 쪽을 고름
PROBLEM_WEIGHT = {"english": 1, "count": 1, "term": 1, "missing": 3, "foreign": 5, "too_short": 10, "meta": 10, "empty": 10}
DISQUALIFY = {"too_short", "meta", "empty", "foreign"}
 
 
def pick_translation(src: str, candidates: List[str]) -> Optional[str]:
    for c in reversed(candidates):           # 재시도 결과가 통과하면 그걸 우선
        if c and not translation_problems(src, c):
            return c
    usable = [c for c in candidates if c and not DISQUALIFY & set(translation_problems(src, c))]
    if not usable:
        return None
    return min(usable, key=lambda c: (sum(PROBLEM_WEIGHT[p] for p in translation_problems(src, c)),
                                      len(LATIN_RE.findall(c))))
 
 
def run_pipeline(dialog_text: str, sents: List[str]) -> Tuple[str, List[Dict[str, Any]]]:
    log: Dict[str, float] = {}
    n = len(sents)
 
    with timer("0 keywords (spaCy)", log):
        kws = [extract_keywords(s) for s in sents]
 
    # ---------- 1단계 ----------
    with timer("1 summary+translate+paraphrase+meaning", log):
        jobs = [make_job(SUMMARY_SYSTEM_PROMPT, dialog_text, max_new_tokens=120)]
        jobs += [translate_job(s) for s in sents]
        jobs += [paraphrase_job(s) for s in sents]
        # 용어집에 있는 단어는 모델에게 묻지 않음
        meanings: List[Dict[str, Optional[str]]] = [{k: glossary_meaning(k) for k in kws[i]} for i in range(n)]
        ask = [[k for k in kws[i] if not meanings[i][k]] for i in range(n)]
        mean_idx = [i for i in range(n) if ask[i]]
        jobs += [meaning_job(sents[i], ask[i]) for i in mean_idx]
        outs = generate_batch(jobs)
 
    summary = clean_output(outs[0], ["Summary:", "요약:"])
    tr = [fix_added_counts(s, clean_output(o, TR_PREFIXES)) for s, o in zip(sents, outs[1:1 + n])]
    para = [clean_output(o, PARA_PREFIXES) for o in outs[1 + n:1 + 2 * n]]
    for k, i in enumerate(mean_idx):
        for w, v in parse_meanings(outs[1 + 2 * n + k], ask[i]).items():
            meanings[i][w] = v
 
    # ---------- 2단계: 걸린 것만 재시도 ----------
    tags, jobs = [], []
    tr_problems = [translation_problems(s, t) for s, t in zip(sents, tr)]
    for i, probs in enumerate(tr_problems):
        if not probs:
            continue
        parts = split_sentences(sents[i])
        if "missing" in probs and len(parts) > 1:
            for j, p in enumerate(parts):          # 문장별로 따로 번역 후 이어 붙임
                tags.append(("tr_split", i, j)); jobs.append(translate_job(p))
        else:
            tags.append(("tr", i, 0))
            jobs.append(translate_job(sents[i], system=translation_retry_system(tr[i], probs)))
    para_fb = [paraphrase_problem(s, p) for s, p in zip(sents, para)]
    for i, fb in enumerate(para_fb):
        if fb:
            tags.append(("para", i, 0))
            jobs.append(paraphrase_job(sents[i], [{"role": "assistant", "content": para[i]},
                                                  {"role": "user", "content": fb}]))
    for i in range(n):
        missing = [k for k in kws[i] if not meanings[i].get(k)]
        if missing:
            tags.append(("mean", i, missing))
            jobs.append(meaning_job(sents[i], missing, [{"role": "assistant", "content": "{}"},
                                                        {"role": "user", "content": MEANING_RETRY_MSG}]))
 
    with timer(f"2 retries ({len(jobs)})", log):
        outs = generate_batch(jobs)
 
    tr_retry: Dict[int, Any] = {}
    for (kind, i, extra), out in zip(tags, outs):
        if kind == "tr":
            tr_retry[i] = fix_added_counts(sents[i], clean_output(out, TR_PREFIXES))
        elif kind == "tr_split":
            tr_retry.setdefault(i, {})[extra] = fix_added_counts(sents[i], clean_output(out, TR_PREFIXES))
        elif kind == "para":
            retry = clean_output(out, PARA_PREFIXES)
            para[i] = retry if paraphrase_problem(sents[i], retry) is None else ""
        elif kind == "mean":
            for k, v in parse_meanings(out, extra).items():
                if v:
                    meanings[i][k] = v
 
    translations: List[Optional[str]] = []
    for i in range(n):
        cands = [tr[i]]
        if isinstance(tr_retry.get(i), dict):
            cands.append(" ".join(tr_retry[i][j] for j in sorted(tr_retry[i])))
        elif i in tr_retry:
            cands.append(tr_retry[i])
        translations.append(pick_translation(sents[i], cands))
 
    # ---------- 3단계: 설명 ----------
    paraphrases: List[Optional[str]] = [None] * n
    changes: List[List[str]] = [[] for _ in range(n)]
    for i in range(n):
        if para[i]:                               # 재시도까지 실패한 건 ""
            ch = word_diff(sents[i], para[i])
            if ch and normalize(para[i]) != normalize(sents[i]):
                paraphrases[i], changes[i] = para[i], ch
 
    exp_idx = [i for i in range(n) if paraphrases[i]]       # 고칠 게 없으면 설명 생성 안 함
    with timer(f"3 explain ({len(exp_idx)})", log):
        outs = generate_batch([make_job(EXPLAIN_SYSTEM_PROMPT, explain_user(sents[i], paraphrases[i], changes[i]),
                                        EXPLAIN_SHOTS, max_new_tokens=160) for i in exp_idx])
    explanations: List[Optional[str]] = [None] * n
    for i, out in zip(exp_idx, outs):
        explanations[i] = clean_output(out, EXP_PREFIXES)
 
    # ---------- 4단계: 영어로 나온 설명만 재시도 ----------
    def explanation_ok(e: str) -> bool:
        return hangul_ratio(e, ignore_quoted=True) >= 0.8 and not FOREIGN_CHAR_RE.search(e)
 
    bad = [i for i in exp_idx if not explanation_ok(explanations[i])]
    with timer(f"4 explain retries ({len(bad)})", log):
        outs = generate_batch([make_job(EXPLAIN_SYSTEM_PROMPT, explain_user(sents[i], paraphrases[i], changes[i]),
                                        EXPLAIN_SHOTS,
                                        [{"role": "assistant", "content": explanations[i]},
                                         {"role": "user", "content": EXPLAIN_RETRY_MSG}], 160) for i in bad])
    for i, out in zip(bad, outs):
        retry = clean_output(out, EXP_PREFIXES)
        explanations[i] = retry if explanation_ok(retry) else template_explanation(changes[i])
 
    print(f"[time] total: {sum(log.values()):.2f}s  ({n} sentences)")
 
    results = [{
        "translation_ko": translations[i],
        "paraphrase_en": paraphrases[i],
        "explanation_ko": explanations[i],
        "keywords": [{"word": w, "meaning_ko": meanings[i].get(w)} for w in kws[i]]
    } for i in range(n)]
    return summary, results
 
 
# =========================================
# 엔드포인트
# =========================================

def print_keywords(paraphrase_results: List[Dict[str, Any]], vocabulary: Dict[str, Optional[str]]) -> None:
    """추출된 키워드를 콘솔에 출력 (발화별 + 전체 단어장)."""
    # print("[keywords] ----- 발화별 -----")
    # for r in paraphrase_results:
    #     kws = ", ".join(f"{k['word']}({k['meaning_ko'] or '-'})" for k in r["keywords"]) or "(없음)"
    #     print(f"[keywords] {r['id']}: {kws}")
    print(f"[keywords] ----- 전체 {len(vocabulary)}개 -----")
    for w, m in vocabulary.items():
        print(f"[keywords]   {w} : {m or '-'}")
        
        

@app.post("/feedback")
def feedback(request: ConversationRequest):
    
    conversations = request.conversation
    if not conversations:
        
        return {"error": "conversation is empty"}
 
    user_turns = []
    for turn in conversations:
        if str(turn.get("role", "")).lower() != "user":
            continue
        text = str(turn.get("message", "")).strip()
        if text:
            user_turns.append((turn, text))
 
    summary, results = run_pipeline(build_dialogue_text(conversations), [t for _, t in user_turns])
 
    paraphrase_results = []
    vocabulary: Dict[str, Optional[str]] = {}
    for (turn, text), result in zip(user_turns, results):
        for kw in result["keywords"]:
            if not vocabulary.get(kw["word"]):
                vocabulary[kw["word"]] = kw["meaning_ko"]
        paraphrase_results.append({
            "id": turn.get("id"),
            "original_en": text,
            "translation_ko": result["translation_ko"],
            "paraphrase_en": result["paraphrase_en"],      # 고칠 게 없으면 null
            "explanation_ko": result["explanation_ko"],    # 고칠 게 없으면 null
            "keywords": result["keywords"]                 # 쉬운 단어만 있으면 []
        })
        
    print_keywords(paraphrase_results, vocabulary)
 
    return {
        "dialog_start_id": conversations[0].get("id"),
        "dialog_end_id": conversations[-1].get("id"),
        "summary": summary,
        "paraphrase_by_user": paraphrase_results,
        "keywords_all": [{"word": w, "meaning_ko": m} for w, m in vocabulary.items()]
    }
 