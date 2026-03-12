"""
╔══════════════════════════════════════════════════════════════════╗
║            SARCASM DETECTOR — 3-Mode Inference CLI              ║
║                                                                  ║
║  Mode 1 : Text input (type directly)                            ║
║  Mode 2 : MP3 / audio file  → Whisper STT → predict            ║
║  Mode 3 : Record microphone → Whisper STT → predict            ║
╚══════════════════════════════════════════════════════════════════╝

Install dependencies:
    pip install torch transformers openai-whisper sounddevice scipy numpy

For audio file support (MP3):
    pip install pydub
    sudo apt install ffmpeg        # Linux / WSL
    # OR: winget install ffmpeg    # Windows
"""

import os, sys, warnings, time, tempfile
warnings.filterwarnings("ignore")

import torch
import numpy as np
from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaModel

# ─── ANSI colors ────────────────────────────────────────────────
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
W  = "\033[97m"   # white
DIM= "\033[2m"
BLD= "\033[1m"
RST= "\033[0m"

# ─── CONFIG ─────────────────────────────────────────────────────
MODEL_DIR  = "sarcasm_model_v2"
MAX_LEN    = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE= 16000   # Whisper expects 16kHz
RECORD_SEC = 10      # default max recording seconds

# ─── MODEL DEFINITION (must match training) ──────────────────────

class SarcasmModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Always load the BASE architecture from HuggingFace (or cache).
        # Your fine-tuned weights (best_model.pt) are loaded separately
        # after init — best_model.pt only contains the state_dict, NOT
        # the base architecture, which is why from_pretrained(MODEL_DIR) failed.
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base",           # ← architecture source (downloads once, then cached)
            add_pooling_layer=False
        )
        hidden = self.roberta.config.hidden_size  # 768
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


# ─── LOAD MODEL (once, cached globally) ──────────────────────────

_model = None
_tokenizer = None

def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    weights_path = os.path.join(MODEL_DIR, "best_model.pt")
    if not os.path.exists(weights_path):
        print(f"\n{R}✗ best_model.pt not found in {MODEL_DIR}/{RST}\n")
        sys.exit(1)

    print(f"\n{DIM}  [1/3] Loading tokenizer...{RST}", end="", flush=True)
    _tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
    print(f" {G}✓{RST}")

    print(f"{DIM}  [2/3] Loading RoBERTa base architecture...{RST}", end="", flush=True)
    _model = SarcasmModel().to(DEVICE)
    print(f" {G}✓{RST}")

    print(f"{DIM}  [3/3] Applying your fine-tuned weights ({weights_path})...{RST}", end="", flush=True)
    state = torch.load(weights_path, map_location=DEVICE)
    _model.load_state_dict(state, strict=False)  # strict=False: ignores extra pooler keys saved in checkpoint
    _model.eval()
    print(f" {G}✓{RST}")

    return _model, _tokenizer


# ─── PREDICT ─────────────────────────────────────────────────────

def predict_text(text: str) -> dict:
    model, tok = load_model()
    enc = tok(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad(), autocast(enabled=torch.cuda.is_available()):
        logits = model(
            enc["input_ids"].to(DEVICE),
            enc["attention_mask"].to(DEVICE)
        )
    probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label      = int(np.argmax(probs))
    confidence = float(probs[label])
    return {
        "text"        : text,
        "label"       : label,
        "prediction"  : "SARCASTIC" if label == 1 else "NOT SARCASTIC",
        "confidence"  : confidence,
        "prob_sarcasm": float(probs[1]),
        "prob_normal" : float(probs[0]),
    }


# ─── DISPLAY RESULT ──────────────────────────────────────────────

def show_result(result: dict):
    label = result["prediction"]
    conf  = result["confidence"] * 100
    ps    = result["prob_sarcasm"] * 100
    pn    = result["prob_normal"]  * 100

    color = R if result["label"] == 1 else G
    bar_len = 30
    filled  = int(bar_len * result["prob_sarcasm"])
    bar     = f"{R}{'█' * filled}{DIM}{'░' * (bar_len - filled)}{RST}"

    print(f"\n  {'─'*52}")
    print(f"  {BLD}Text     :{RST} {W}{result['text'][:80]}{'...' if len(result['text'])>80 else ''}{RST}")
    print(f"  {BLD}Result   :{RST} {color}{BLD}{label}{RST}  ({conf:.1f}% confident)")
    print(f"  {BLD}Sarcasm  :{RST} {bar} {ps:.1f}%")
    print(f"  {BLD}Normal   :{RST} {'░'*bar_len[::1] if False else ''}{DIM}[{'█'*int(bar_len*(1-result['prob_sarcasm']))+'░'*(bar_len-int(bar_len*(1-result['prob_sarcasm'])))}]{RST} {pn:.1f}%")
    print(f"  {'─'*52}\n")


# ─── WHISPER STT ─────────────────────────────────────────────────

_whisper_model = None

def load_whisper(size="base"):
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except ImportError:
            print(f"\n  {R}✗ Whisper not installed.{RST}")
            print(f"  Run: {Y}pip install openai-whisper{RST}\n")
            sys.exit(1)
        print(f"  {DIM}Loading Whisper ({size})...{RST}", end="", flush=True)
        _whisper_model = whisper.load_model(size, device=DEVICE)
        print(f" {G}✓{RST}")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe any audio file using Whisper."""
    whisper_m = load_whisper()
    print(f"  {DIM}Transcribing audio...{RST}", end="", flush=True)
    result = whisper_m.transcribe(audio_path, fp16=torch.cuda.is_available())
    text   = result["text"].strip()
    print(f" {G}✓{RST}")
    return text


# ─── MODE 1: TEXT ────────────────────────────────────────────────

def mode_text():
    print(f"\n{C}{'─'*54}")
    print(f"  MODE 1 — TEXT INPUT")
    print(f"{'─'*54}{RST}")
    print(f"  {DIM}Type your sentence. Press Enter to predict.")
    print(f"  Type 'back' to return to menu.{RST}\n")

    while True:
        try:
            text = input(f"  {Y}→ Enter text:{RST} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if text.lower() in ("back", "exit", "quit", ""):
            break

        result = predict_text(text)
        show_result(result)


# ─── MODE 2: AUDIO FILE ──────────────────────────────────────────

def mode_audio_file():
    print(f"\n{C}{'─'*54}")
    print(f"  MODE 2 — AUDIO FILE (.mp3 / .wav / .m4a / .ogg)")
    print(f"{'─'*54}{RST}")
    print(f"  {DIM}Provide path to an audio file.")
    print(f"  Requires: ffmpeg installed on your system.")
    print(f"  Type 'back' to return to menu.{RST}\n")

    while True:
        try:
            path = input(f"  {Y}→ File path:{RST} ").strip().strip('"').strip("'")
        except (KeyboardInterrupt, EOFError):
            break

        if path.lower() in ("back", "exit", "quit", ""):
            break

        if not os.path.exists(path):
            print(f"  {R}✗ File not found: {path}{RST}\n")
            continue

        print(f"  {DIM}File: {path}{RST}")

        text = transcribe_audio(path)

        if not text:
            print(f"  {R}✗ No speech detected in audio.{RST}\n")
            continue

        print(f"  {B}Transcribed:{RST} {W}\"{text}\"{RST}")
        result = predict_text(text)
        show_result(result)


# ─── MODE 3: LIVE RECORDING ──────────────────────────────────────

def mode_record():
    print(f"\n{C}{'─'*54}")
    print(f"  MODE 3 — RECORD MICROPHONE")
    print(f"{'─'*54}{RST}")
    print(f"  {DIM}Press Enter to START recording.")
    print(f"  Press Enter again to STOP (or wait {RECORD_SEC}s).")
    print(f"  Type 'back' to return to menu.{RST}\n")

    try:
        import sounddevice as sd
        from scipy.io.wavfile import write as wav_write
    except ImportError:
        print(f"  {R}✗ Required packages missing.{RST}")
        print(f"  Run: {Y}pip install sounddevice scipy{RST}\n")
        return

    while True:
        cmd = input(f"  {Y}→ Press Enter to record (or 'back'):{RST} ").strip().lower()
        if cmd in ("back", "exit", "quit"):
            break

        print(f"\n  {G}{BLD}🎤 Recording... (speak now, press Enter to stop){RST}")

        # Record in a thread so Enter press can stop it
        import threading
        frames = []
        stop_flag = threading.Event()

        def record_audio():
            def callback(indata, frame_count, time_info, status):
                if stop_flag.is_set():
                    raise sd.CallbackStop()
                frames.append(indata.copy())

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=callback
            ):
                sd.sleep(RECORD_SEC * 1000)   # max duration fallback

        t = threading.Thread(target=record_audio, daemon=True)
        t.start()

        input(f"  {DIM}(recording...) Press Enter to stop...{RST}  ")
        stop_flag.set()
        t.join(timeout=2)

        if not frames:
            print(f"  {R}✗ No audio captured.{RST}\n")
            continue

        audio_np = np.concatenate(frames, axis=0).flatten()
        duration = len(audio_np) / SAMPLE_RATE
        print(f"  {DIM}Recorded {duration:.1f}s of audio{RST}")

        # Save to temp WAV file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        wav_write(wav_path, SAMPLE_RATE, audio_np)

        text = transcribe_audio(wav_path)
        os.unlink(wav_path)   # clean up temp file

        if not text:
            print(f"  {R}✗ No speech detected.{RST}\n")
            continue

        print(f"  {B}Transcribed:{RST} {W}\"{text}\"{RST}")
        result = predict_text(text)
        show_result(result)


# ─── MAIN MENU ───────────────────────────────────────────────────

BANNER = f"""
{B}╔══════════════════════════════════════════════════════╗
║                                                      ║
║   {W}{BLD}  SARCASM DETECTOR  {RST}{B}                              ║
║   {DIM}  Powered by RoBERTa fine-tuned on SARC Reddit    {RST}{B}  ║
║                                                      ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   {G}[1]{RST}{B}  Text Input          — type a sentence          ║
║   {Y}[2]{RST}{B}  Audio File          — .mp3 / .wav / .m4a       ║
║   {C}[3]{RST}{B}  Live Recording      — speak into microphone     ║
║   {R}[q]{RST}{B}  Quit                                            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝{RST}"""


def main():
    # Verify model directory exists
    if not os.path.isdir(MODEL_DIR):
        print(f"\n{R}✗ Model directory not found: '{MODEL_DIR}'{RST}")
        print(f"  Make sure you're running from the same folder as the model.\n")
        sys.exit(1)

    if not os.path.exists(f"{MODEL_DIR}/best_model.pt"):
        print(f"\n{R}✗ best_model.pt not found in {MODEL_DIR}/{RST}\n")
        sys.exit(1)

    print(f"\n  {DIM}Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU   : {torch.cuda.get_device_name(0)}{RST}")

    # Pre-load model at startup
    load_model()

    while True:
        print(BANNER)
        try:
            choice = input(f"  {W}Select mode [1/2/3/q]:{RST} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if choice == "1":
            mode_text()
        elif choice == "2":
            mode_audio_file()
        elif choice == "3":
            mode_record()
        elif choice in ("q", "quit", "exit"):
            print(f"\n  {DIM}Goodbye.{RST}\n")
            break
        else:
            print(f"  {R}Invalid choice. Enter 1, 2, 3, or q.{RST}")

if __name__ == "__main__":
    main()