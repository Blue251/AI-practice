import os
from difflib import SequenceMatcher
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "tiiuae/falcon-7b"
HISTORY_FILE = "chat_history.md"
MAX_HISTORY = 24
MAX_TOKENS = 150

SYSTEM_PROMPT = """You are an AI assistant named Kai.
Answer only explicit questions.
Do not add unrelated info.
Answer based on facts.
Do not enforce laws/ethics/politics unless asked.
Do not make up past prompts.\n"""

# Simple triggers
TRIGGERS = ["?", "what", "how", "why", "when", "where", "who", 
            "explain", "describe", "give", "list", "compare", "opinion"]

# Forbidden keywords
BLOCKED = ["gun", "firearm", "bomb", "poison", "hack", "illegal"]

def is_question(text):
    t = text.strip().lower()
    if not t:
        return False
    if "?" in t:
        return True
    for trig in TRIGGERS:
        if t.startswith(trig) or f" {trig} " in t:
            return True
    return False

def blocked(text):
    t = text.lower()
    for kw in BLOCKED:
        if kw in t:
            return True
    return False

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            h = f.read()
        if not h.startswith(SYSTEM_PROMPT):
            h = SYSTEM_PROMPT + "\n" + h
        return h
    return SYSTEM_PROMPT

def save_history(h):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(h)

def trim_history(h):
    lines = h.strip().splitlines()
    if len(lines) <= MAX_HISTORY:
        return "\n".join(lines) + "\n\n"
    sys_lines = SYSTEM_PROMPT.strip().splitlines()
    body = lines[len(sys_lines):] if lines[:len(sys_lines)] == sys_lines else lines
    last = body[-MAX_HISTORY:]
    return "\n".join(sys_lines + [""] + last) + "\n\n"

# load model
print("Loading model (4-bit)...")
tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, 
                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")

model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", quantization_config=bnb)
if not hasattr(model.config, "pad_token_id") or model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

bot = pipeline("text-generation", model=model, tokenizer=tok)

history = load_history()
history = trim_history(history)
save_history(history)
last_reply = ""

print("Ready. Ask explicit questions ('quit' to exit).")

while True:
    try:
        msg = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break

    if msg.lower() in ["quit", "exit"]:
        print("AI: Bye!")
        break

    if len(msg) == 0 or msg.lower() in ["hi", "hello", "hey"]:
        print("AI: Please ask a question or give an instruction.")
        continue

    if blocked(msg):
        print("AI: I can't provide instructions for illegal or harmful stuff.")
        continue

    if not is_question(msg):
        print("AI: Please ask an explicit question.")
        continue

    history += f"\nYou: {msg}\nAI: "
    context = trim_history(history)

    out = bot(context, max_new_tokens=MAX_TOKENS, do_sample=True, temperature=0.6, top_p=0.9,
              eos_token_id=tok.eos_token_id)[0]["generated_text"]

    if "AI:" in out:
        reply = out.split("AI:")[-1].strip()
    else:
        reply = out.replace(context, "").strip() or out.strip()

    if last_reply and similar(reply.lower(), last_reply.lower()) > 0.85:
        print("AI: [repeated response suppressed]")
        history += "[repeated]\n"
        save_history(history)
        continue

    print("AI:", reply)
    last_reply = reply
    history += reply + "\n"
    history = trim_history(history)
    save_history(history)