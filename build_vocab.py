# build_vocab.py — 위키낱말사전(Wiktionary)으로 영어→한국어 단어 사전 만들기
#
# 결과: vocab_wiktionary.json  (main5.py와 같은 폴더에 두면 서버가 자동으로 읽음)
#   - vocab_ko.json(직접 만든 사전)이 우선이고, 거기 없는 단어만 이 파일에서 찾음
#   - 뜻이 여러 개면 "분수, 원천"처럼 쉼표로 저장 → 서버가 문맥을 보고 하나를 고름
#
# 사용법 (feedback_server 폴더에서):
#   pip install requests wordfreq
#   python build_vocab.py                 # 데이터를 내려받아서 만듦 (약 2~3GB, 시간이 걸림)
#   python build_vocab.py --input 파일경로  # 이미 받아 둔 파일로 만듦 (.jsonl 또는 .jsonl.gz)
#
# 데이터: kaikki.org의 English Wiktionary 추출본 (CC BY-SA, https://kaikki.org/dictionary/English/)
#   자동 다운로드가 안 되면 위 페이지에서 "all word senses" JSONL 파일을 받아 --input으로 넣으세요.
 
import argparse
import gzip
import json
import os
import re
import sys
from collections import OrderedDict
 
DEFAULT_URL = "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl"
HERE = os.path.dirname(os.path.abspath(__file__))
 
HANGUL_RE = re.compile(r"[가-힣]")
BAD_CHAR_RE = re.compile(r"[A-Za-z0-9一-鿿぀-ヿ]")   # 영어, 숫자, 한자, 일본어
KEEP_POS = {"noun", "verb", "adj", "adv", "phrase", "prep_phrase", "intj"}
 
 
def download(url: str, path: str) -> None:
    import requests
    print(f"[다운로드] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="")
    print("\n[다운로드] 완료")
 
 
def open_any(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")
 
 
def clean_ko(word: str):
    """"분수 (噴水)" → "분수". 한글이 아닌 글자가 남으면 버림."""
    w = re.sub(r"\(.*?\)|\[.*?\]", "", word or "").strip()
    w = re.sub(r"\s+", " ", w)
    if not w or not HANGUL_RE.search(w) or BAD_CHAR_RE.search(w) or len(w) > 15:
        return None
    return w
 
 
def korean_translations(entry: dict):
    """(뜻 설명, 한국어 단어) 목록. 항목 전체와 sense별 translations 둘 다 봄."""
    items = list(entry.get("translations") or [])
    for s in entry.get("senses") or []:
        items += s.get("translations") or []
    for t in items:
        if t.get("code") == "ko" or t.get("lang") == "Korean":
            ko = clean_ko(t.get("word", ""))
            if ko:
                yield (t.get("sense") or ""), ko
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="kaikki English JSONL(.gz) 파일 경로")
    ap.add_argument("--out", default=os.path.join(HERE, "vocab_wiktionary.json"))
    ap.add_argument("--max-senses", type=int, default=3, help="단어당 저장할 뜻 개수")
    ap.add_argument("--min-zipf", type=float, default=2.0,
                    help="이보다 드문 단어는 뺌 (wordfreq 기준, 0이면 모두 포함)")
    args = ap.parse_args()
 
    path = args.input
    if not path:
        path = os.path.join(HERE, "kaikki-English.jsonl")
        if not os.path.exists(path):
            try:
                download(DEFAULT_URL, path)
            except Exception as e:
                sys.exit(f"[오류] 다운로드 실패: {e}\n"
                         "https://kaikki.org/dictionary/English/ 에서 JSONL 파일을 받아 --input 으로 넣어 주세요.")
 
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        zipf_frequency = None
        print("[경고] wordfreq가 없어 드문 단어 필터를 끕니다 (pip install wordfreq)")
 
    vocab: "OrderedDict[str, list]" = OrderedDict()
    n_lines = 0
    with open_any(path) as f:
        for line in f:
            n_lines += 1
            if n_lines % 200000 == 0:
                print(f"  {n_lines:,}줄 처리, 단어 {len(vocab):,}개")
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("lang_code", "en") != "en" or e.get("pos") not in KEEP_POS:
                continue
            word = (e.get("word") or "").strip()
            if not word or not re.fullmatch(r"[A-Za-z][A-Za-z' -]*", word) or word[0].isupper():
                continue                      # 고유명사(대문자 시작), 기호가 들어간 표제어는 제외
            word = word.lower()
            if zipf_frequency and args.min_zipf > 0 and zipf_frequency(word, "en") < args.min_zipf:
                continue
            senses = vocab.setdefault(word, [])
            seen_glosses = set()
            for gloss, ko in korean_translations(e):
                if len(senses) >= args.max_senses:
                    break
                # 같은 뜻 설명(sense)에서는 첫 번째 번역만 → 서로 다른 뜻이 후보가 되도록
                if gloss and gloss in seen_glosses:
                    continue
                seen_glosses.add(gloss)
                if ko not in senses:
                    senses.append(ko)
 
    result = OrderedDict((w, ", ".join(s)) for w, s in vocab.items() if s)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=0)
    print(f"[완료] {len(result):,}개 단어 → {args.out}")
    for w in ["fountain", "puppy", "thirsty", "pet", "freeze", "luckily", "breeze", "wooden"]:
        print(f"  {w}: {result.get(w, '(없음)')}")
 
 
if __name__ == "__main__":
    main()
 