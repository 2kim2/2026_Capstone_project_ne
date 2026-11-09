# qwen_local_chat.py  (v5: AI 발화 번역 + 번역 품질 + 속도)
#
# v4(main2.py) 대비 바뀐 점
#   [번역] AI(assistant) 발화도 모두 번역 → 응답에 translation_by_turn 추가 (기존 필드는 그대로)
#   [번역] 화자에 맞는 말투(직원/면접관 = 정중한 존댓말, 학습자 = 손님 말투) + 앞 대화 2턴을 문맥으로 제공
#   [번역] 직원/면접관/의사 말투 few-shot 추가, 문맥까지 번역해 버린 경우(too_long/extra) 문맥 없이 재시도
#   [단어] AI 발화에서도 단어 추출 (KEYWORDS_FROM_AI), 같은 단어는 한 번만 뜻 생성, 용어/제외어 직접 추가 가능
#   [속도] 시작할 때 워밍업, 번역/단어 뜻 캐시, 출력 길이 기준 배치 정렬, OOM 나면 배치 자동 분할,
#          설명을 템플릿으로 바꾸는 빠른 모드(EXPLAIN_WITH_LLM=False), 동시 요청 직렬화(lock)
#   [옵션] ADAPTER_DIR에 번역 LoRA 폴더를 넣으면 번역에만 어댑터 사용 (main4_lora.py 기능 통합)
#   [배포] 음성 AI 서버(main.py)와 한 서버로: 이 파일을 main.py 폴더에 feedback.py로 두고 main.py에서
#            from feedback import router as feedback_router
#            app.include_router(feedback_router, prefix="/ai")
#          LLM_BACKEND="ollama": main.py와 같은 Ollama 모델을 사용 (GPU에 모델을 따로 안 올림)
#          같은 폴더에 vocab_ko.json, vocab_wiktionary.json 필요. pip install spacy wordfreq
#
# 필요 패키지: pip install spacy wordfreq && python -m spacy download en_core_web_sm
#             (LoRA를 쓸 때만) pip install peft
 
import difflib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
 
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
 
# =========================================
# 설정
# =========================================
# ---- LLM 실행 방식 ----
#   "ollama":       음성 AI 서버(main.py)와 같은 Ollama를 HTTP로 호출. 모델을 따로 GPU에 올리지 않음 (권장)
#   "transformers": 이 프로세스가 Qwen을 직접 GPU에 올림 (model_dir 필요, 기존 방식)
LLM_BACKEND = "ollama"
OLLAMA_URL = "http://127.0.0.1:11434"       # main.py와 같은 주소
OLLAMA_MODEL = "liveng-qwen"                # main.py와 같은 모델이면 VRAM을 추가로 쓰지 않음. `ollama list`로 이름 확인
OLLAMA_NUM_CTX = 2048                       # main.py와 반드시 같게! 다르면 Ollama가 모델을 다시 올려서 두 서버 모두 느려짐
OLLAMA_PARALLEL = 4                         # 피드백 작업을 동시에 몇 개 보낼지 (Ollama의 OLLAMA_NUM_PARALLEL과 맞추면 좋음)
 
# transformers 방식일 때만 사용
model_dir = r"C:\Users\DS\Desktop\ne_yolo\feedback_server\qwen2.5_3b"
LOAD_IN_4BIT = True          # 8GB GPU에서는 True 권장 (fp16 3B는 VRAM이 넘쳐 6배 이상 느려짐). pip install bitsandbytes
 
MAX_KEYWORDS = 5
KEYWORD_MAX_ZIPF = 4.5       # 이 값보다 흔한 단어는 "쉬운 단어"로 보고 제외 (낮출수록 어려운 단어만)
PHRASE_MAX_ZIPF = 4.8        # 복합명사/구동사(credit card, pick up)용 기준
BATCH_SIZE = 8               # 한 번에 생성할 프롬프트 수. [batch] 로그의 vram이 GPU 용량에 가까우면 낮추세요 (4bit 기준 8GB면 8 OK)
DEBUG_BATCH = True           # 배치마다 시간/토큰 수/GPU 메모리를 출력 (느린 원인 찾기용)
RESTRICT_CHARS = True        # 생성 단계에서 한글/영어/기본 문장부호 외 문자(일본어, 중국어, 태국어 등)를 막음
DEBUG_REJECTS = True         # 검사에서 버려진 모델 출력을 콘솔에 출력 (원인 파악용)
 
# ---- v5 추가 설정 ----
TRANSLATE_AI = True          # AI(assistant) 발화도 번역 (translation_by_turn)
KEYWORDS_FROM_AI = True      # AI 발화에서도 단어 추출 (keywords_all에 source="ai"로 들어감)
TERM_HINT_RETRY = False      # True면 단어 뜻(명사)이 번역문에 없을 때 그 뜻을 힌트로 다시 번역
                             # 사전 뜻은 문맥을 모르기 때문에 엉뚱한 뜻을 끼워 넣을 수 있어 기본은 끔
                             # ("pet him" → 애완동물을 만질 수 있나요?)
KEYWORDS_REQUIRE_DICT = True # 사전(용어집, vocab_ko.json, vocab_wiktionary.json)에 있는 단어만 단어 목록에 넣음
                             # → 모델이 뜻을 지어내는 일("나이텔리")이 없어지고 빨라짐. 사전이 없으면 자동으로 꺼짐
CONTEXT_TURNS = 2            # 번역할 때 문맥으로 보여 줄 앞 턴 수 (0이면 문맥 없이 문장만 번역)
EXPLAIN_WITH_LLM = True      # False면 설명을 모델 대신 템플릿으로 만듦 (3·4단계 생략, 가장 큰 속도 이득)
WARMUP = True                # 서버 시작할 때 한 번 생성해 둠 (첫 요청이 느린 문제)
CACHE_SIZE = 5000            # 번역/단어 뜻 캐시 크기 (같은 문장, 같은 단어는 다시 생성하지 않음)
# 번역 LoRA 어댑터 폴더 (finetune/train_qlora.py 결과). None이면 기본 모델만 사용
ADAPTER_DIR = None           # 예: r"C:\Users\DS\Desktop\ne_yolo\feedback_server\finetune\out\qwen2.5-3b-translate-lora"
LORA_TASKS = {"translate"}   # 어댑터를 쓸 작업
 
# 단어 목록에서 빼고 싶은 단어 (너무 쉽거나 의미 없는 단어를 여기 추가)
EXTRA_EXCLUDE_WORDS = {"sir", "ma'am", "dollar", "dollars", "cent", "cents", "minute", "minutes", "today", "problem", "sure"}
 
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
    "warm up": "따뜻해지다, 데우다", "heat up": "데우다",
    "tap": "(카드를) 대다", "insert": "(카드를) 넣다",
    "cafe": "카페", "café": "카페", "ice americano": "아이스 아메리카노", "iced americano": "아이스 아메리카노",
    "ice cream": "아이스크림", "iced tea": "아이스티", "ice tea": "아이스티", "counter": "카운터", "ice": "얼음", "no ice": "얼음 없이",
    "iced": "아이스(얼음을 넣은)", "medium": "미디엄", "medium size": "미디엄 사이즈",
    "small size": "스몰 사이즈", "large size": "라지 사이즈", "pick up": "받아 가다",
}
# 번역할 때만 쓰는 표현 힌트 (단어 뜻에는 안 쓰이고, 번역문에 꼭 나와야 하는 것도 아님)
TRANSLATE_HINTS = {
    "absolutely": "물론이죠 / 그럼요", "sure thing": "물론이죠", "got it": "알겠습니다",
    "at all": "전혀", "for here": "손님: 먹고 갈게요 / 직원: 드시고 가세요",
    "to go": "손님: 포장해 주세요 / 직원: 포장해 드릴까요", "okay": "네 / 알겠어요",
}
# 단어 뜻 사전 (main5.py와 같은 폴더). 여기 있는 단어는 모델에게 뜻을 묻지 않음
#   vocab_ko.json         직접 만든 사전 (우선)
#   vocab_wiktionary.json build_vocab.py로 만든 큰 사전 (선택, 직접 만든 사전에 없는 단어만)
#   형식: {"영어 단어(소문자, 사전형)": "한국어 뜻"}. 뜻이 여러 개면 "쓰다듬다, 반려동물"처럼 쉼표로
#   → 뜻이 여러 개면 모델이 문장을 보고 하나를 고름 (sense_job)
_HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB: Dict[str, str] = {}
for _name in ["vocab_wiktionary.json", "vocab_ko.json"]:      # 뒤에 읽은 파일이 우선
    _path = os.path.join(_HERE, _name)
    try:
        with open(_path, encoding="utf-8") as _f:
            _d = {k.lower(): v for k, v in json.load(_f).items()}
        VOCAB.update(_d)
        print(f"[info] 단어 사전 {_name}: {len(_d)}개")
    except FileNotFoundError:
        print(f"[info] 단어 사전 없음: {_name}")
 
GLOSSARY_NO_CHECK = {"warm up", "heat up", "tap", "insert", "pick up", "no ice", "iced"}   # 번역문에 그대로 안 나와도 되는 항목 (동사 등)
 
# =========================================
# 모델 로딩
# =========================================
if LLM_BACKEND == "transformers":
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
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
    if ADAPTER_DIR:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_DIR, adapter_name="lora")
        print(f"[lora] 어댑터 사용: {ADAPTER_DIR} (작업: {sorted(LORA_TASKS)})")
    model.eval()
    print(f"[info] device={model.device}, dtype={next(model.parameters()).dtype}, "
          f"cuda={torch.cuda.is_available()}" + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
else:
    import httpx
    tokenizer = model = None
    ollama_client = httpx.Client(base_url=OLLAMA_URL,
                                 timeout=httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=10.0))
    try:
        _names = [m.get("name", "") for m in ollama_client.get("/api/tags").json().get("models", [])]
        if not any(n == OLLAMA_MODEL or n.split(":")[0] == OLLAMA_MODEL for n in _names):
            print(f"[경고] Ollama에 '{OLLAMA_MODEL}' 모델이 없습니다. 있는 모델: {_names}")
        print(f"[info] LLM: Ollama {OLLAMA_URL} / {OLLAMA_MODEL} (num_ctx={OLLAMA_NUM_CTX}, 동시 {OLLAMA_PARALLEL})")
    except Exception as e:
        print(f"[경고] Ollama에 연결하지 못했습니다 ({OLLAMA_URL}): {e}")
 
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
 
 
# 외국 문자 차단은 transformers 방식에서만 가능 (Ollama는 검사 단계에서 걸러서 재시도)
char_blocker = build_char_blocker() if RESTRICT_CHARS and LLM_BACKEND == "transformers" else None
 
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
# 엔드포인트는 router에 등록 → 음성 AI 서버(main.py)에 붙여서 서버 하나로 돌릴 수 있음
#   main.py에서:  from feedback import router as feedback_router
#                 app.include_router(feedback_router, prefix="/ai")   → /ai/feedback, /ai/translate
#   단독 실행도 가능: uvicorn feedback:app --port 8001  → /feedback, /translate
router = APIRouter()
 
 
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
    "You are a professional English-to-Korean translator for conversation practice. "
    "Translate the English text into natural, everyday spoken Korean, the way a Korean person would really say it "
    "in the same situation. Translate the meaning, not word by word.\n"
    "Rules:\n"
    "- Translate every sentence, including short ones like \"Thank you.\" or \"For here.\"\n"
    "- Write the whole answer in Hangul. Do not leave any English letters. "
    "Write menu items and loanwords the way Koreans spell them (vanilla latte -> 바닐라 라테, muffin -> 머핀).\n"
    "- Keep quantities and prices exactly as in the original. \"a\" or \"one\" means one (한 잔, 하나). Never add a number.\n"
    "- Match the relationship. Staff and customers, doctors, interviewers: polite 존댓말. "
    "Friends or classmates who call each other by first name: friendly 해요체 (~해요, ~네요), "
    "not stiff 합니다체, and never add 님 to their names. Do not use 반말.\n"
    "- Write people's names the standard Korean way (Mark -> 마크, Sarah -> 사라). "
    "Translate ordinary nouns into Korean words (park -> 공원, lake -> 호수); "
    "keep loanwords only for menu items, brands, and words Koreans really use as loanwords.\n"
    "- Leave out pronouns the way Koreans do. Do not write 그는, 그녀는, 그의 for he/she/his "
    "(use the name, or 얘 for a pet, or nothing), and use 저는/제가 instead of 나는/내가 in polite speech.\n"
    "- Never write 당신 or 너. Leave out \"you\" and \"your\" the way Koreans do "
    "(\"your coffee\" -> 커피, \"your number\" -> 번호).\n"
    "- Follow Korean word order: the item comes first, then the count "
    "(\"one iced americano\" -> 아이스 아메리카노 한 잔, never 하나 아이스 아메리카노).\n"
    "- Output only the translation. Do not answer, explain, or comment on the text."
)
# 화자별 말투 힌트 (시스템 프롬프트 끝에 붙음)
ROLE_HINT = {
    "user": "The speaker is the learner (a customer, guest, patient, or employee). "
            "Use the natural polite Korean that person would use (e.g. ~주세요, ~할 수 있을까요?).",
    "ai": "The speaker is the other person in the conversation (staff, receptionist, doctor, interviewer, "
          "manager, or a friend). If they are staff, use polite service Korean (뭐 드릴까요?, ~해 드릴게요); "
          "if they are a friend, use friendly 해요체.",
}
TRANSLATE_SHOTS = [
    ("Hi, can I get two hot lattes, please?",
     "안녕하세요, 따뜻한 라테 두 잔 주시겠어요?"),
    ("Hi there! What can I get for you today?",
     "안녕하세요! 오늘 뭐 드릴까요?"),
    ("Can I get an iced caramel macchiato and a blueberry bagel?",
     "아이스 카라멜 마키아토 한 잔이랑 블루베리 베이글 하나 주시겠어요?"),
    ("Sure. Would you like that hot or iced? Is that for here or to go?",
     "네. 따뜻한 걸로 드릴까요, 아이스로 드릴까요? 드시고 가세요, 아니면 포장해 드릴까요?"),
    ("Large, please. Oh, and no whipped cream. Actually, extra whipped cream.",
     "라지로 주세요. 아, 휘핑크림은 빼 주세요. 아니다, 휘핑크림 많이 올려 주세요."),
    ("That will be 9 dollars and 50 cents. Please tap your card here.",
     "9달러 50센트입니다. 여기에 카드를 대 주세요."),
    ("To go. Thank you. Is there a restroom here?",
     "포장이요. 감사합니다. 여기 화장실 있나요?"),
    ("I want one iced americano and one cheesecake, please.",
     "아이스 아메리카노 한 잔이랑 치즈케이크 하나 주세요."),
    ("Medium size. And can you make it no ice?",
     "미디엄 사이즈로 주세요. 그리고 얼음은 빼 주실 수 있나요?"),
    ("For here. How much is it all? Can I pay by credit card?",
     "먹고 갈게요. 전부 얼마예요? 신용카드로 결제할 수 있나요?"),
    ("You can pick up your drink at the counter when your number is called.",
     "번호가 불리면 카운터에서 음료를 받아 가시면 돼요."),
    ("Could you give us a quick update on the project? Are we still on track?",
     "프로젝트 진행 상황을 간단히 말씀해 주시겠어요? 아직 일정대로 진행되고 있나요?"),
    ("I'll review it today and get back to you.",
     "오늘 검토해 보고 다시 연락드릴게요."),
    ("Hi Mark! It's so nice to see you at the park today.",
     "마크, 안녕하세요! 오늘 공원에서 보니까 정말 반갑네요."),
    ("Oh, wow. Maybe we should finish our walk quickly before it starts to pour.",
     "아, 와. 비가 쏟아지기 전에 빨리 산책을 끝내는 게 좋겠어요."),
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
    "- 대문자/소문자 차이는 설명하지 마세요. tone, nuance 같은 영어 단어를 한국어 문장에 섞지 마세요.\n"
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
    ("원문: I want one ice americano.\n"
     "고친 문장: I'd like one iced americano.\n"
     "바뀐 부분:\n- \"want\" → \"'d like\"\n- \"ice\" → \"iced\"",
     "\"want\"보다 \"I'd like\"가 주문할 때 더 공손합니다. "
     "얼음을 넣은 음료는 명사 \"ice\"가 아니라 형용사 \"iced\"를 써서 \"iced americano\"라고 합니다."),
]
 
MEANING_SYSTEM_PROMPT = (
    "You are an English-Korean dictionary. For each English word or phrase, give its short Korean dictionary "
    "meaning as it is used in the given sentence. Write the meaning in Hangul only (1-6 words), never in English. "
    "Verbs end with \"-다\" (e.g. 결제하다). "
    "Use the native Korean word whenever one exists (fountain -> 분수, exit -> 출구, forecast -> 일기 예보); "
    "never make up a transliteration for an ordinary word. "
    "Only for loanwords that Koreans really use (menu items, brands, foods), write the Hangul transliteration, "
    "and add a short Korean gloss in parentheses only when the loanword is hard to understand "
    "(e.g. oat milk -> 오트밀크(귀리 우유)). Never put another spelling in the parentheses. "
    "Return ONLY a JSON object mapping each word to its meaning."
)
MEANING_SHOTS = [
    ('Sentence: Go straight and turn left at the big fountain.\nWords: ["fountain", "turn left"]',
     '{"fountain": "분수", "turn left": "왼쪽으로 돌다"}'),
    ('Sentence: I checked the forecast. The breeze feels amazing.\nWords: ["forecast", "breeze"]',
     '{"forecast": "일기 예보", "breeze": "산들바람"}'),
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
             max_new_tokens: int = 128, task: str = "") -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in (shots or []):
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_content})
    messages.extend(extra or [])
    if LLM_BACKEND == "transformers":
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = json.dumps(messages, ensure_ascii=False)      # 정렬용 길이 계산에만 씀
    adapter = "lora" if ADAPTER_DIR and task in LORA_TASKS else "__base__"
    return {"prompt": prompt, "messages": messages, "max_new_tokens": max_new_tokens, "adapter": adapter}
 
 
def _ollama_generate(job: Dict[str, Any]) -> str:
    approx_tokens = len(job["prompt"]) // 3
    if approx_tokens + job["max_new_tokens"] > OLLAMA_NUM_CTX:
        print(f"[경고] 프롬프트가 num_ctx({OLLAMA_NUM_CTX})보다 길 수 있어요 (약 {approx_tokens}토큰). "
              f"앞부분이 잘리면 품질이 떨어집니다.")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": job["messages"],
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0.0, "num_predict": job["max_new_tokens"],
                    "num_ctx": OLLAMA_NUM_CTX, "repeat_penalty": 1.05},
    }
    try:
        r = ollama_client.post("/api/chat", json=payload)
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[오류] Ollama 요청 실패: {e}")
        return ""
 
 
_ollama_pool = ThreadPoolExecutor(max_workers=OLLAMA_PARALLEL) if LLM_BACKEND == "ollama" else None
 
 
def _generate_chunk(prompts: List[str], max_new_tokens: int, adapters: List[str]) -> List[str]:
    """프롬프트 묶음 하나를 생성. GPU 메모리가 부족하면 반으로 나눠서 다시 시도."""
    try:
        t0 = time.perf_counter()
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        start = enc["input_ids"].shape[1]
        lora_kw = {"adapter_names": adapters} if ADAPTER_DIR else {}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                **lora_kw,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=char_blocker
            )
        if DEBUG_BATCH:
            vram = (f", vram {torch.cuda.max_memory_allocated() / 2**30:.1f}"
                    f"(예약 {torch.cuda.memory_reserved() / 2**30:.1f})/"
                    f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}GB"
                    if torch.cuda.is_available() else "")
            print(f"[batch] {len(prompts)}개, 입력 {start}토큰, 생성 {out.shape[1] - start}토큰 "
                  f"(최대 {max_new_tokens}), {time.perf_counter() - t0:.1f}s{vram}")
        return [tokenizer.decode(o[start:], skip_special_tokens=True).strip() for o in out]
    except torch.cuda.OutOfMemoryError:
        if len(prompts) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(prompts) // 2
        print(f"[warn] GPU 메모리 부족 → 배치 {len(prompts)}개를 {half}+{len(prompts) - half}로 나눔 "
              f"(BATCH_SIZE를 낮추면 이 경고가 사라집니다)")
        return (_generate_chunk(prompts[:half], max_new_tokens, adapters[:half])
                + _generate_chunk(prompts[half:], max_new_tokens, adapters[half:]))
 
 
def generate_batch(jobs: List[Dict[str, Any]]) -> List[str]:
    if not jobs:
        return []
    if LLM_BACKEND == "ollama":
        t0 = time.perf_counter()
        outs = list(_ollama_pool.map(_ollama_generate, jobs))
        if DEBUG_BATCH:
            print(f"[batch] Ollama {len(jobs)}개 (동시 {OLLAMA_PARALLEL}), {time.perf_counter() - t0:.1f}s")
        return outs
    outputs = [""] * len(jobs)
    # 한 배치는 가장 긴 출력이 끝날 때까지 기다리므로, 출력 길이(max_new_tokens)가 비슷한 것끼리 묶음.
    # 같으면 프롬프트 길이로 묶어서 패딩 낭비를 줄임.
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i]["max_new_tokens"], len(jobs[i]["prompt"])))
    for c in range(0, len(order), BATCH_SIZE):
        idx = order[c:c + BATCH_SIZE]
        res = _generate_chunk([jobs[i]["prompt"] for i in idx],
                              max(jobs[i]["max_new_tokens"] for i in idx),
                              [jobs[i]["adapter"] for i in idx])
        for i, r in zip(idx, res):
            outputs[i] = r
    return outputs
 
 
class LRUCache:
    """간단한 LRU 캐시 (서버가 켜져 있는 동안 유지)."""
    def __init__(self, size: int):
        self.size, self.data = size, OrderedDict()
 
    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None
 
    def put(self, key, value):
        if value is None or self.size <= 0:
            return
        self.data[key] = value
        self.data.move_to_end(key)
        while len(self.data) > self.size:
            self.data.popitem(last=False)
 
 
TRANSLATION_CACHE = LRUCache(CACHE_SIZE)   # (role, 문장) → 번역 (검사를 통과한 것만)
MEANING_CACHE = LRUCache(CACHE_SIZE)       # 단어 → 뜻
 
 
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
SELF_HONORIFIC_RE = re.compile(r"(가|하|드|들으|받으|오|계)(시겠|실 수|시나요|신가요|셨)")
META_RE = re.compile(r"원래|원문|문장|번역|translation|sentence", re.IGNORECASE)
KO_COUNT_RE = re.compile(r"(두|세|네|다섯|여섯)\s?(개|잔|조각|명|병|장|그릇|판)")
EN_PLURAL_COUNT_RE = re.compile(r"\b(two|three|four|five|six|couple|[2-9])\b", re.IGNORECASE)
 
 
def translation_problems(src: str, ko: str, role: str = "ai") -> List[str]:
    """번역문의 문제 목록. 빈 리스트면 통과."""
    if not ko:
        return ["empty"]
    problems = []
    if FOREIGN_CHAR_RE.search(ko):
        problems.append("foreign")
    # 너무 짧음: 한글 글자 수가 원문 영어 단어 수보다 적으면 일부만 번역한 것 ("머핀")
    n_ko, n_en = len(HANGUL_RE.findall(ko)), len(words(src))
    if n_ko < max(1, int(0.8 * n_en)):
        problems.append("too_short")
    # 너무 김 / 문장이 너무 많음: 문맥으로 준 앞 대화까지 번역했거나 답을 덧붙인 경우
    if n_ko > 4 * n_en + 20:
        problems.append("too_long")
    if len(split_sentences(ko)) > len(split_sentences(src)) + 1:
        problems.append("extra")
    if LATIN_RE.search(ko):
        problems.append("english")
    if META_RE.search(ko) and not META_RE.search(src):
        problems.append("meta")
    if re.search(r"당신|너는|너의", ko):
        problems.append("you")              # 한국어는 "당신의 커피"처럼 말하지 않음
    # 학습자(손님)가 자기 행동에 높임말을 씀: "먹고 가시겠어요", "커피 어디서 들으실 수 있어요?"
    if role == "user" and not re.search(r"\byou", src, re.IGNORECASE) and SELF_HONORIFIC_RE.search(ko):
        problems.append("honorific")
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
    if ko:
        ko = re.sub(r"당신(의|은|도|이|을|께서는|께서)\s+", "", ko)   # "당신의 번호가" → "번호가"
        ko = re.sub(r"(^|[.!?,]\s*)그(의|는|녀의|녀는)\s+", r"\1", ko)   # "그의 이름은" → "이름은"
        # 존댓말 문장의 "나는/내가" → "저는/제가" (문장 맨 앞이나 쉼표 뒤만: "화가 나는" 같은 말은 안 건드림)
        for a, b in (("나는", "저는"), ("내가", "제가"), ("나도", "저도"), ("나의", "제")):
            ko = re.sub(rf"(^|[.!?,]\s*){a}\s", rf"\g<1>{b} ", ko)
        ko = re.sub(r"^(okay|ok|yes|yeah)\b[,.!]?\s*", "네, ", ko, flags=re.IGNORECASE)   # "okay, 결제했어요"
        ko = re.sub(r"\bwow\b", "와", ko, flags=re.IGNORECASE)
        ko = re.sub(r"(아|어)?\b(aww+|aw)\b", "아", ko, flags=re.IGNORECASE)    # "아ww" → "아"
        ko = re.sub(r"(?<=[가-힣])w{2,}", "", ko)
        ko = re.sub(r"\boh\b", "아", ko, flags=re.IGNORECASE)
        ko = fix_count_order(src, ko)
    return ko
 
 
COUNT_WORDS = r"(하나|한 잔|한 개|두 잔|두 개|세 잔|세 개|한 조각|두 조각)"
 
 
def fix_count_order(src: str, ko: str) -> str:
    """영어 어순을 그대로 옮긴 수량을 뒤로: "한 잔 아이스 아메리카노와" → "아이스 아메리카노 한 잔과" """
    for item in glossary_hits(src).values():
        def swap(m):
            count, particle = m.group(1), m.group(2) or ""
            final = (ord(count[-1]) - 0xAC00) % 28 != 0          # 마지막 글자 받침 여부 (잔: 있음, 개/하나: 없음)
            pairs = {"와": "과", "과": "와", "랑": "이랑", "이랑": "랑", "를": "을", "을": "를"}
            if particle in pairs:
                want_final = particle in {"과", "이랑", "을"}
                if want_final != final:
                    particle = pairs[particle]
            return f"{item} {count}{particle}"
        ko = re.sub(COUNT_WORDS + r"\s+" + re.escape(item) + r"(와|과|이랑|랑|를|을)?", swap, ko)
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
    # 예시(few-shot) 문장이 섞여 나옴: "Can I pay by card? Thanks a lot! You are right."
    src_words = set(words(src))
    for sent in split_sentences(para):
        pw = words(sent)
        if pw and sum(w in src_words for w in pw) / len(pw) < 0.5:
            return f'You added "{sent}", which is not in the original. Do not add anything.'
    # 원문의 내용어가 빠짐 ("than yesterday"를 지우고 "isn't it"을 붙임 → 뜻이 바뀜)
    dropped = [w for w in dict.fromkeys(words(src))
               if len(w) > 2 and w not in FUNCTION_WORDS and w not in BASIC_WORDS and w not in {"yeah", "yep", "okay"}
               and not any(p[:4] == w[:4] for p in para_words)]
    if dropped:
        return ("Keep the original meaning. You dropped: " + ", ".join(f'"{w}"' for w in dropped)
                + ". Fix only grammar and word choice.")
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
 
 
def drop_case_talk(e: str) -> str:
    """대문자 이야기 문장은 지움 (diff는 대소문자를 무시하므로 실제 변경이 아님)."""
    sents = re.split(r"(?<=[.!?다])\s+", e.strip())
    kept = [x for x in sents if "대문자" not in x and "소문자" not in x]
    return " ".join(kept) if kept else e
 
 
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
            and word not in FUNCTION_WORDS and word not in BASIC_WORDS
            and word not in EXTRA_EXCLUDE_WORDS)
 
 
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
            if all(t.pos_ == "PROPN" for t in span):     # "Star Cafe" 같은 가게/사람 이름은 제외
                used.update(t.i for t in span)
                continue
            if all(t.pos_ in {"NOUN", "PROPN"} for t in span):
                phrase = " ".join(t.lemma_.lower() for t in span)
                used.update(t.i for t in span)
                if phrase not in seen and phrase not in EXTRA_EXCLUDE_WORDS and is_hard_enough(phrase, True):
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
        if tok.pos_ == "PROPN" and lemma not in GLOSSARY:     # 사람/가게 이름 (Sarah, Mark)
            continue
        if _accept(lemma, seen) and is_hard_enough(lemma, False):
            found.append(lemma)
            seen.add(lemma)
 
    # 사전에 있는 단어만 (사전 파일이 있을 때)
    if KEYWORDS_REQUIRE_DICT and VOCAB:
        found = [w for w in found if glossary_meaning(w)]
    # 긴 구에 포함된 짧은 구는 뺌: "dog drinking fountain"이 있으면 "drinking fountain"은 제외
    found = [w for w in found if not any(w != o and re.search(rf"\b{re.escape(w)}\b", o) for o in found)]
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
    if word in VOCAB:
        return VOCAB[word]
    parts = word.split()
    if len(parts) > 1 and all(p in GLOSSARY for p in parts):
        return " ".join(main_form(GLOSSARY[p]) for p in parts)
    return None
 
 
def fix_verb_form(m: str) -> str:
    """동사 뜻을 사전형으로: "따뜻하게 해다" → "따뜻하게 하다", "결제해요" → "결제하다" """
    m = re.sub(r"해다$", "하다", m)
    m = re.sub(r"(해요|합니다|해주다|해 주다)$", "하다", m)
    # "내린다" → "내리다", "간다" → "가다" (받침 ㄴ + 다)
    if len(m) >= 2 and m.endswith("다") and "\uAC00" <= m[-2] <= "\uD7A3" and (ord(m[-2]) - 0xAC00) % 28 == 4:
        m = m[:-2] + chr(ord(m[-2]) - 4) + "다"
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
    # "오트밀크(귀리 우유)" → "오트밀크",  "(카드를) 대다" → "카드를 대다"
    ko = re.sub(r"\(.*\)$", "", ko).strip()
    return re.sub(r"^\((.*?)\)\s*", r"\1 ", ko).strip()
 
 
def translate_job(text: str, role: str = "user", context: str = "", extra=None,
                  system: str = TRANSLATE_SYSTEM_PROMPT, terms: Optional[Dict[str, str]] = None):
    """role: "user"(학습자) / "ai"(상대). context: 앞 대화 (번역하지 않고 상황 파악용으로만 보여 줌)."""
    system += "\n- " + ROLE_HINT.get(role, ROLE_HINT["ai"])
    hits = glossary_hits(text)
    hits = {**(terms or {}), **hits}          # 단어 뜻 단계에서 얻은 뜻도 같이 (fountain = 분수)
    if hits:
        system += "\n- Use these Korean words: " + ", ".join(f"{en} = {ko}" for en, ko in hits.items()) + "."
    low = text.lower()
    phrases = {en: ko for en, ko in TRANSLATE_HINTS.items() if re.search(rf"\b{re.escape(en)}\b", low)}
    if phrases:
        system += "\n- Natural Korean for these phrases: " + ", ".join(f"{en} = {ko}" for en, ko in phrases.items()) + "."
    if context:
        system += ("\n\nThe conversation so far, for context only. Do NOT translate it; "
                   "translate only the new text you are given:\n" + context)
    return make_job(system, text, TRANSLATE_SHOTS, extra, tokens_for(text, 4, 24, 160), task="translate")
 
 
def paraphrase_job(text: str, extra=None):
    return make_job(PARAPHRASE_SYSTEM_PROMPT, text, PARAPHRASE_SHOTS, extra, tokens_for(text, 2, 16, 120),
                    task="paraphrase")
 
 
SENSE_SYSTEM_PROMPT = (
    "You choose the right Korean meaning of an English word in a sentence. "
    "Read the sentence, then answer with only the number of the meaning that fits. No other words."
)
SENSE_SHOTS = [
    ("Sentence: Is it okay if I pet him?\nWord: pet\n1) 쓰다듬다\n2) 반려동물", "1"),
    ("Sentence: Do you have any pets?\nWord: pet\n1) 쓰다듬다\n2) 반려동물", "2"),
    ("Sentence: Luckily, it warmed up nicely today.\nWord: warm up\n1) 따뜻해지다\n2) 데우다", "1"),
]
 
 
def split_senses(m: str) -> List[str]:
    return [x.strip() for x in m.split(",") if x.strip()]
 
 
def sense_job(sentence: str, word: str, senses: List[str]):
    user = f"Sentence: {sentence}\nWord: {word}\n" + "\n".join(f"{k + 1}) {v}" for k, v in enumerate(senses))
    return make_job(SENSE_SYSTEM_PROMPT, user, SENSE_SHOTS, max_new_tokens=4)
 
 
def pick_sense(out: str, senses: List[str]) -> Optional[str]:
    m = re.search(r"\d+", out)
    if m and 1 <= int(m.group()) <= len(senses):
        return senses[int(m.group()) - 1]
    return None
 
 
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
    if "honorific" in problems:
        hints.append("The speaker is talking about their own action, so do not use 시 honorifics "
                     "(먹고 갈게요, not 먹고 가시겠어요).")
    if "you" in problems:
        hints.append("Do not use 당신. Leave out \"you\" and \"your\".")
    if "too_long" in problems or "extra" in problems:
        hints.append("Translate only the given text. Do not add anything.")
    return TRANSLATE_SYSTEM_PROMPT + "\n- " + " ".join(hints) if hints else TRANSLATE_SYSTEM_PROMPT
 
 
# 문제의 심각도. 둘 다 문제가 있을 때 덜 심각한 쪽을 고름
PROBLEM_WEIGHT = {"english": 1, "count": 1, "term": 1, "you": 1, "honorific": 2, "missing": 3, "extra": 3, "foreign": 5, "too_long": 5,
                  "too_short": 10, "meta": 10, "empty": 10}
DISQUALIFY = {"too_short", "meta", "empty", "foreign"}
 
 
def pick_translation(src: str, candidates: List[str], role: str = "ai") -> Optional[str]:
    for c in reversed(candidates):           # 재시도 결과가 통과하면 그걸 우선
        if c and not translation_problems(src, c, role):
            return c
    usable = [c for c in candidates if c and not DISQUALIFY & set(translation_problems(src, c, role))]
    if not usable:
        return None
    return min(usable, key=lambda c: (sum(PROBLEM_WEIGHT[p] for p in translation_problems(src, c, role)),
                                      len(LATIN_RE.findall(c))))
 
 
def run_pipeline(dialog_text: str, turns: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """turns: [{"role": "user"|"ai", "text": 영어, "context": 앞 대화}] (대화 순서)
    돌려주는 results[i]: translation_ko / paraphrase_en / explanation_ko / keywords (turns와 같은 순서)"""
    log: Dict[str, float] = {}
    n = len(turns)
    sents = [t["text"] for t in turns]
    roles = [t["role"] for t in turns]
    user_idx = [i for i in range(n) if roles[i] == "user"]
    tr_idx = [i for i in range(n) if roles[i] == "user" or TRANSLATE_AI]
    kw_idx = [i for i in range(n) if roles[i] == "user" or KEYWORDS_FROM_AI]
 
    with timer("0 keywords (spaCy)", log):
        kws: List[List[str]] = [[] for _ in range(n)]
        for i in kw_idx:
            kws[i] = extract_keywords(sents[i])
 
    # 단어 뜻은 대화 전체에서 단어당 한 번만 구함 (용어집 → 캐시 → 모델 순)
    meaning: Dict[str, Optional[str]] = {}
    ask: Dict[int, List[str]] = {}            # 처음 나온 발화 i → 모델에게 물을 단어들
    choose: List[Tuple[int, str, List[str]]] = []   # (발화 i, 단어, 사전 뜻 후보들)
    for i in kw_idx:
        for w in kws[i]:
            if w in meaning:
                continue
            meaning[w] = glossary_meaning(w) or MEANING_CACHE.get(w)
            if not meaning[w]:
                ask.setdefault(i, []).append(w)
            elif len(split_senses(meaning[w])) > 1:
                choose.append((i, w, split_senses(meaning[w])))   # 사전 뜻이 여러 개 → 문맥으로 고르게 함
 
    # 번역 캐시
    tr: Dict[int, str] = {}
    tr_todo = []
    for i in tr_idx:
        cached = TRANSLATION_CACHE.get((roles[i], sents[i]))
        if cached:
            tr[i] = cached
        else:
            tr_todo.append(i)
 
    # ---------- 1단계: 요약 + 번역 + 패러프레이즈 + 단어 뜻 ----------
    with timer(f"1 summary+translate({len(tr_todo)})+paraphrase({len(user_idx)})+meaning({len(ask)})+sense({len(choose)})", log):
        jobs = [make_job(SUMMARY_SYSTEM_PROMPT, dialog_text, max_new_tokens=120)]
        jobs += [translate_job(sents[i], roles[i], turns[i]["context"]) for i in tr_todo]
        jobs += [paraphrase_job(sents[i]) for i in user_idx]
        ask_idx = list(ask)
        jobs += [meaning_job(sents[i], ask[i]) for i in ask_idx]
        jobs += [sense_job(sents[i], w, senses) for i, w, senses in choose]
        outs = generate_batch(jobs)
 
    summary = clean_output(outs[0], ["Summary:", "요약:"])
    pos = 1
    for i in tr_todo:
        tr[i] = fix_added_counts(sents[i], clean_output(outs[pos], TR_PREFIXES)); pos += 1
    para: Dict[int, str] = {}
    for i in user_idx:
        para[i] = clean_output(outs[pos], PARA_PREFIXES); pos += 1
    for i in ask_idx:
        for w, v in parse_meanings(outs[pos], ask[i]).items():
            meaning[w] = v
        pos += 1
    for i, w, senses in choose:
        picked = pick_sense(outs[pos], senses)    # 못 고르면 후보를 모두 보여 줌 ("쓰다듬다, 반려동물")
        if picked:
            meaning[w] = picked
        elif DEBUG_REJECTS:
            print(f"[reject] sense {w} {senses} <- {outs[pos]!r}")
        pos += 1
 
    # ---------- 2단계: 걸린 것만 재시도 ----------
    tags, jobs = [], []
    tr_problems = {i: translation_problems(sents[i], tr[i], roles[i]) for i in tr_todo}
 
    def missing_terms(i: int) -> Dict[str, str]:
        """단어 뜻(명사)이 번역문에 없으면 그 뜻을 힌트로: 번역 "큰 파인드" vs 뜻 fountain = 분수"""
        out = {}
        for w in kws[i]:
            m = meaning.get(w)
            if not m or w in GLOSSARY or "," in m:
                continue
            m = main_form(m)
            if len(m) >= 2 and not re.search(r"(다|게|히|이)$", m) and m.replace(" ", "") not in tr[i].replace(" ", ""):
                out[w] = m
        return out
 
    for i in tr_todo:
        probs = list(tr_problems[i])
        terms = missing_terms(i) if TERM_HINT_RETRY else {}
        if terms and "term" not in probs:
            probs.append("term")
        if not probs:
            continue
        parts = split_sentences(sents[i])
        if "missing" in probs and len(parts) > 1:
            for j, p in enumerate(parts):          # 문장별로 따로 번역 후 이어 붙임
                tags.append(("tr_split", i, j)); jobs.append(translate_job(p, roles[i]))
        else:
            # 재시도는 문맥 없이 (문맥까지 번역해 버리는 실수를 막기 위해)
            tags.append(("tr", i, 0))
            jobs.append(translate_job(sents[i], roles[i], system=translation_retry_system(tr[i], probs), terms=terms))
    for i in user_idx:
        fb = paraphrase_problem(sents[i], para[i])
        if fb:
            tags.append(("para", i, 0))
            jobs.append(paraphrase_job(sents[i], [{"role": "assistant", "content": para[i]},
                                                  {"role": "user", "content": fb}]))
    for i in ask_idx:
        missing = [w for w in ask[i] if not meaning.get(w)]
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
            for w, v in parse_meanings(out, extra).items():
                if v:
                    meaning[w] = v
 
    translations: List[Optional[str]] = [None] * n
    for i in tr_idx:
        if i not in tr_problems:                  # 캐시에서 온 번역
            translations[i] = tr[i]
            continue
        cands = [tr[i]]
        if isinstance(tr_retry.get(i), dict):
            cands.append(" ".join(tr_retry[i][j] for j in sorted(tr_retry[i])))
        elif i in tr_retry:
            cands.append(tr_retry[i])
        translations[i] = pick_translation(sents[i], cands, roles[i])
        if translations[i] and not translation_problems(sents[i], translations[i], roles[i]):
            TRANSLATION_CACHE.put((roles[i], sents[i]), translations[i])
    for w, v in meaning.items():
        if v and not glossary_meaning(w):
            MEANING_CACHE.put(w, v)
 
    # ---------- 3단계: 설명 ----------
    paraphrases: List[Optional[str]] = [None] * n
    changes: List[List[str]] = [[] for _ in range(n)]
    for i in user_idx:
        if para[i]:                               # 재시도까지 실패한 건 ""
            ch = word_diff(sents[i], para[i])
            if ch and normalize(para[i]) != normalize(sents[i]):
                paraphrases[i], changes[i] = para[i], ch
 
    exp_idx = [i for i in user_idx if paraphrases[i]]       # 고칠 게 없으면 설명 생성 안 함
    explanations: List[Optional[str]] = [None] * n
    if not EXPLAIN_WITH_LLM:
        for i in exp_idx:
            explanations[i] = template_explanation(changes[i])
    else:
        with timer(f"3 explain ({len(exp_idx)})", log):
            outs = generate_batch([make_job(EXPLAIN_SYSTEM_PROMPT, explain_user(sents[i], paraphrases[i], changes[i]),
                                            EXPLAIN_SHOTS, max_new_tokens=160) for i in exp_idx])
        for i, out in zip(exp_idx, outs):
            explanations[i] = drop_case_talk(clean_output(out, EXP_PREFIXES))
 
        # ---------- 4단계: 영어로 나온 설명만 재시도 ----------
        def explanation_ok(e: str) -> bool:
            # 따옴표 밖에 영어 단어가 하나라도 있으면 다시 ("더 존중받는 tone을" 같은 경우)
            outside = re.sub(r"\"[^\"]*\"|'[^']*'|“[^”]*”", "", e)
            return (hangul_ratio(e, ignore_quoted=True) >= 0.8 and not FOREIGN_CHAR_RE.search(e)
                    and not LATIN_RE.search(outside))
 
        bad = [i for i in exp_idx if not explanation_ok(explanations[i])]
        with timer(f"4 explain retries ({len(bad)})", log):
            outs = generate_batch([make_job(EXPLAIN_SYSTEM_PROMPT, explain_user(sents[i], paraphrases[i], changes[i]),
                                            EXPLAIN_SHOTS,
                                            [{"role": "assistant", "content": explanations[i]},
                                             {"role": "user", "content": EXPLAIN_RETRY_MSG}], 160) for i in bad])
        for i, out in zip(bad, outs):
            retry = drop_case_talk(clean_output(out, EXP_PREFIXES))
            explanations[i] = retry if explanation_ok(retry) else template_explanation(changes[i])
 
    print(f"[time] total: {sum(log.values()):.2f}s  ({n} turns, user {len(user_idx)}, "
          f"translation cache hit {len(tr_idx) - len(tr_todo)})")
 
    results = [{
        "translation_ko": translations[i],
        "paraphrase_en": paraphrases[i],
        "explanation_ko": explanations[i],
        "keywords": [{"word": w, "meaning_ko": meaning.get(w)} for w in kws[i]]
    } for i in range(n)]
    return summary, results
 
 
# =========================================
# 시작할 때 워밍업 (CUDA 커널 준비 때문에 첫 생성이 유독 느림)
# =========================================
if WARMUP:
    _t = time.perf_counter()
    generate_batch([translate_job("Hello, can I get a latte, please?"), paraphrase_job("I want coffee.")])
    print(f"[info] 워밍업 완료 ({time.perf_counter() - _t:.1f}s)")
 
 
# =========================================
# 엔드포인트
# =========================================
def print_keywords(paraphrase_results: List[Dict[str, Any]], vocabulary: Dict[str, Dict[str, Any]]) -> None:
    """추출된 키워드를 콘솔에 출력 (전체 단어장)."""
    # print("[keywords] ----- 발화별 -----")
    # for r in paraphrase_results:
    #     kws = ", ".join(f"{k['word']}({k['meaning_ko'] or '-'})" for k in r["keywords"]) or "(없음)"
    #     print(f"[keywords] {r['id']}: {kws}")
    print(f"[keywords] ----- 전체 {len(vocabulary)}개 -----")
    for w, v in vocabulary.items():
        print(f"[keywords]   {w} : {v['meaning_ko'] or '-'}  ({v['source']})")
 
 
_lock = threading.Lock()   # GPU 1개라 요청은 한 번에 하나씩 처리 (동시에 오면 줄 서서 기다림)
 
 
@router.post("/feedback")
def feedback(request: ConversationRequest):
    conversations = request.conversation
    if not conversations:
        return {"error": "conversation is empty"}
 
    # 비어 있지 않은 모든 턴 (user + AI). 문맥은 바로 앞 CONTEXT_TURNS개 턴
    turns, history = [], []
    for turn in conversations:
        text = str(turn.get("message", "")).strip()
        if not text:
            continue
        role = "user" if str(turn.get("role", "")).lower() == "user" else "ai"
        context = "\n".join(history[-CONTEXT_TURNS:]) if CONTEXT_TURNS > 0 else ""
        turns.append({"turn": turn, "role": role, "text": text, "context": context})
        history.append(f"{'A' if role == 'user' else 'B'}: {text}")
 
    with _lock:
        summary, results = run_pipeline(build_dialogue_text(conversations), turns)
 
    paraphrase_results, translation_results = [], []
    vocabulary: Dict[str, Dict[str, Any]] = {}
    same: Dict[str, str] = {}               # "cheese cake"와 "cheesecake"는 같은 단어로 침
    for t, result in zip(turns, results):
        for kw in result["keywords"]:
            word = same.setdefault(kw["word"].replace(" ", "").replace("-", ""), kw["word"])
            v = vocabulary.setdefault(word, {"meaning_ko": kw["meaning_ko"], "source": t["role"]})
            if not v["meaning_ko"]:
                v["meaning_ko"] = kw["meaning_ko"]
            if t["role"] == "user":
                v["source"] = "user"           # 사용자도 쓴 단어면 user 우선
        translation_results.append({
            "id": t["turn"].get("id"),
            "role": t["turn"].get("role"),               # 요청에 온 role 그대로
            "original_en": t["text"],
            "translation_ko": result["translation_ko"],
            "keywords": result["keywords"]
        })
        if t["role"] != "user":
            continue
        paraphrase_results.append({
            "id": t["turn"].get("id"),
            "original_en": t["text"],
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
        "translation_by_turn": translation_results,       # v5 추가: AI + user 모든 턴 번역 (대화 순서)
        "paraphrase_by_user": paraphrase_results,
        "keywords_all": [{"word": w, "meaning_ko": v["meaning_ko"], "source": v["source"]}
                         for w, v in vocabulary.items()]
    }
 
 
# =========================================
# 턴 하나만 번역 (대화 중에 실시간으로 부르는 용도)
#   대화하는 동안 백엔드가 턴마다 이걸 불러 두면, 번역이 캐시에 남아서
#   마지막 /feedback 때는 번역을 다시 만들지 않음 → /feedback이 그만큼 빨라짐
# =========================================
class TranslateRequest(BaseModel):
    role: str = "assistant"
    message: str
    context: List[Dict[str, Any]] = []      # 선택: 앞 턴들 [{"role": ..., "message": ...}]
 
 
def translate_one(role: str, text: str, context: str) -> Optional[str]:
    cached = TRANSLATION_CACHE.get((role, text))
    if cached:
        return cached
    out = fix_added_counts(text, clean_output(generate_batch([translate_job(text, role, context)])[0], TR_PREFIXES))
    cands = [out]
    probs = translation_problems(text, out, role)
    if probs:
        retry = generate_batch([translate_job(text, role, system=translation_retry_system(out, probs))])[0]
        cands.append(fix_added_counts(text, clean_output(retry, TR_PREFIXES)))
    best = pick_translation(text, cands, role)
    if best and not translation_problems(text, best, role):
        TRANSLATION_CACHE.put((role, text), best)
    return best
 
 
@router.post("/translate")
def translate(request: TranslateRequest):
    text = request.message.strip()
    if not text:
        return {"translation_ko": None}
    role = "user" if request.role.lower() == "user" else "ai"
    history = [f"{'A' if str(t.get('role', '')).lower() == 'user' else 'B'}: {str(t.get('message', '')).strip()}"
               for t in request.context if str(t.get("message", "")).strip()]
    context = "\n".join(history[-CONTEXT_TURNS:]) if CONTEXT_TURNS > 0 else ""
    t0 = time.perf_counter()
    with _lock:
        ko = translate_one(role, text, context)
    print(f"[time] /translate: {time.perf_counter() - t0:.2f}s")
    return {"role": request.role, "original_en": text, "translation_ko": ko}
 
 
# 단독 실행용 앱 (main.py에 붙일 때는 쓰이지 않음)
app = FastAPI(title="Feedback Server")
app.include_router(router)
 