import os
import re
import time
import asyncio
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

# -----------------------------
# Config
# -----------------------------
token = os.getenv("HF_TOKEN")
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")

LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "180"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# (권장) 입력 캡 – 너무 길면 LLM이 느려지고 비용 커짐
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "40"))
MAX_REVIEW_CHARS = int(os.getenv("MAX_REVIEW_CHARS", "400"))
MAX_TOTAL_CHARS = int(os.getenv("MAX_TOTAL_CHARS", "12000"))

# LLM 동시성 제한 (필수)
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "1"))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# -----------------------------
# App + Globals
# -----------------------------
app = FastAPI(title="Embedding + Venue Eval API")

embedder: Optional[SentenceTransformer] = None
tokenizer: Optional[AutoTokenizer] = None
llm: Optional[AutoModelForCausalLM] = None

llm_sem = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

# -----------------------------
# DTOs
# -----------------------------
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]

class EvalEmbedRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1)

class EvalEmbedResponse(BaseModel):
    evaluation: str
    embedding: List[float]

# -----------------------------
# Helpers
# -----------------------------
_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = s.strip()
    s = _ws_re.sub(" ", s)
    return s

def preprocess_reviews(reviews: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    total_chars = 0

    for r in reviews:
        if not isinstance(r, str):
            continue
        r = normalize_text(r)
        if not r:
            continue

        if len(r) > MAX_REVIEW_CHARS:
            r = r[:MAX_REVIEW_CHARS]

        if r in seen:
            continue
        seen.add(r)

        if total_chars + len(r) > MAX_TOTAL_CHARS:
            break

        cleaned.append(r)
        total_chars += len(r)

        if len(cleaned) >= MAX_REVIEWS:
            break

    return cleaned

def build_prompt_ko(reviews: List[str]) -> str:
    bullets = "\n".join([f"- {r}" for r in reviews])
    return (
        "아래는 한 고사장에 대한 여러 리뷰이다. 리뷰 내용만 근거로 고사장에 대한 평가를 작성하라.\n"
        "규칙:\n"
        "1) 첫번째 문장은 한줄짜리 핵심 평가를 작성하고 그 다음에 참고할만한 중요 사항들을 작성한다.\n"
        "2) 리뷰 문장을 그대로 반복하거나 나열하지 말고 의미만 요약한다.\n"
        "3) 장점과 단점을 균형 있게 포함한다.\n"
        "4) 불필요한 서론, 결론, 형식어(예: 총평, 요약, 평가:)를 절대 쓰지 않고 바로 작성한다.\n"
        "5) 문단 구분이나 줄바꿈 없이 한 단락으로 작성한다.\n\n"
        "6) 친근한 말투와 표현을 사용한다."
        f"[리뷰]\n{bullets}\n\n"
        "위 리뷰를 기반으로 압축된 평가 문장을 작성하라."
    )
def strip_model_fluff(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(평가|요약|총평)\s*[:\-]\s*", "", text)
    return text.strip()

def generate_evaluation_sync(reviews: List[str]) -> str:
    """동기 LLM 생성 (async 엔드포인트에서 to_thread로 호출)"""
    assert tokenizer is not None and llm is not None

    user_prompt = build_prompt_ko(reviews)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "너는 고사장에 대한 리뷰를 근거로 공정하고 간결한 평가를 작성하는 어시스턴트다."},
            {"role": "user", "content": user_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(llm.device)
    else:
        input_ids = tokenizer(user_prompt, return_tensors="pt").input_ids.to(llm.device)

    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_ids.shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return strip_model_fluff(text)

# -----------------------------
# Startup
# -----------------------------
@app.on_event("startup")
def startup():
    global embedder, tokenizer, llm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # Embedder
    print(f"[startup] loading embedder: {EMB_MODEL}")
    t0 = time.time()
    embedder = SentenceTransformer(EMB_MODEL, device="cuda")
    embedder.encode(["query: warmup"], normalize_embeddings=True)
    torch.cuda.synchronize()
    print(f"[startup] embedder ready ({(time.time()-t0)*1000:.1f} ms)")

    # LLM (optional but we load here for simplicity)
    print(f"[startup] loading LLM (4bit): {LLM_MODEL}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    llm.eval()

    # LLM warmup
    _ = generate_evaluation_sync(["웜업 리뷰입니다. 조용하고 쾌적했어요."])
    torch.cuda.synchronize()
    print(f"[startup] LLM ready ({(time.time()-t0)*1000:.1f} ms)")

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ✅ 기존 기능: 텍스트 1개 → 임베딩
@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    if embedder is None:
        raise HTTPException(status_code=500, detail="embedder not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    emb = embedder.encode([f"query: {text}"], normalize_embeddings=True)[0]
    torch.cuda.synchronize()
    return EmbedResponse(embedding=emb.astype(np.float32).tolist())

# ✅ 신규 기능: 리뷰들 → 평가 + 평가 임베딩
@app.post("/venue/eval-embed", response_model=EvalEmbedResponse)
async def eval_embed(req: EvalEmbedRequest):
    if embedder is None or tokenizer is None or llm is None:
        raise HTTPException(status_code=500, detail="models not loaded")

    reviews = preprocess_reviews(req.reviews)
    if not reviews:
        raise HTTPException(status_code=400, detail="no valid reviews")

    async with llm_sem:
        t0 = time.time()
        evaluation = await asyncio.to_thread(generate_evaluation_sync, reviews)
        llm_ms = (time.time() - t0) * 1000

    t1 = time.time()
    emb = embedder.encode([f"query: {evaluation}"], normalize_embeddings=True)[0]
    torch.cuda.synchronize()
    emb_ms = (time.time() - t1) * 1000

    print(f"[eval-embed] reviews={len(reviews)} llm={llm_ms:.1f}ms emb={emb_ms:.1f}ms")

    return EvalEmbedResponse(
        evaluation=evaluation,
        embedding=emb.astype(np.float32).tolist(),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=1, reload=False)
