import os
import wave
import glob
import re
import time
import cv2
import openai
import logging
import threading
from typing import Type
from PIL import Image
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
import assemblyai as aai
import pyaudio
import pytesseract, shutil


from assemblyai.streaming.v3 import (
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    BeginEvent,
    TurnEvent,
    TerminationEvent,
    StreamingError,
)
import pytesseract, shutil

if shutil.which("tesseract") is None:
    for candidate in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

latest_frame = None
_frame_grabber_stop = False
def frame_grabber():
    global latest_frame, _frame_grabber_stop
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while not _frame_grabber_stop:
            ret, frame = cap.read()
            if ret:
                latest_frame = frame
            time.sleep(0.01)  # yield CPU
    finally:
        cap.release()

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
aai.settings.api_key = API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load BLIP model
try:
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        local_files_only=True
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        local_files_only=True
    )
    print("✅ Loaded BLIP model from cache.")
except Exception:
    processor = None
    model = None
    print("⚠️  BLIP model not found locally. Visual descriptions will be skipped.")

# Comprehensive list of visual triggers
VISUAL_TRIGGERS = [
    "as you can see", "as we can see", "look at this", "if you look here", "shown here", "depicted here",
    "on the board", "on this board", "on the screen", "on this screen", "on the slide", "on this slide",
    "on the whiteboard", "on this chart", "up on the board", "what's shown here",
    "this figure shows", "this diagram", "this image shows", "pictured here", "take a look at",
    "check this out", "observe this", "illustrated here", "refer to this image",
    "notice how", "you'll notice", "focus on this", "here's a diagram", "this chart demonstrates",
    "this example shows", "i'm holding up a", "visualize this", "you can see the"
]

capturing = threading.Event()
transcript_lines = []

# OpenAI wrapper updated for openai>=1.0.0
def ask_openai(prompt: str, model_name="gpt-4") -> str:
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You describe images for visually impaired users."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
            timeout=15,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(OpenAI error: {e})"
def read_text_from_frame(frame, debug=False):
    import cv2, numpy as np, pytesseract

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Heuristic ROI: whiteboard is on the LEFT
    h, w = gray.shape
    roi = gray[:, :int(w * 0.65)]

    # Prep
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi = clahe.apply(roi)

    th_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th_adap = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )

    def ensure_white_bg(img):
        # Tesseract prefers black text on white; invert if necessary
        return cv2.bitwise_not(img) if img.mean() < 128 else img

    th_otsu = ensure_white_bg(th_otsu)
    th_adap = ensure_white_bg(th_adap)

    cfgs = [
        "--oem 3 --psm 7 -c user_defined_dpi=300",
        "--oem 3 --psm 6 -c user_defined_dpi=300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    ]

    for name, img in (("otsu", th_otsu), ("adap", th_adap), ("roi", roi)):
        for cfg in cfgs:
            # ---------- DEBUG LINE GOES HERE ----------
            if debug:
                data = pytesseract.image_to_data(
                    img, config=cfg, output_type=pytesseract.Output.DICT
                )
                words = [(t, float(c)) for t, c in zip(data["text"], data["conf"])
                         if t.strip() and c != "-1"]
                print(f"[OCR DEBUG] {name} | {cfg}")
                print("  words:", words)
            # -----------------------------------------

            txt = pytesseract.image_to_string(img, config=cfg).strip()
            if txt:
                return txt

    if debug:
        print("[OCR DEBUG] No text found")
    return ""


def capture_and_describe(idx: int):
    # wait up to 1 s for a frame to appear
    deadline = time.time() + 1
    frame = None
    while frame is None and time.time() < deadline:
        frame = latest_frame

    if frame is None:
        transcript_lines.append("[Visual Description: no frame available]")
        capturing.clear()
        return

    img_filename = f"session{idx}_image.jpg"
    cv2.imwrite(img_filename, frame)
    print(f"📷 Captured image → {img_filename}")

    # then continue with your existing image-to-text logic (BLIP + GPT)…


    # BLIP caption
    caption = ""
    if processor and model:
        try:
            raw = Image.fromarray(frame[..., ::-1]).convert("RGB")
            inputs = processor(raw, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            caption = ""
    else:
        caption = ""


    ocr_text = read_text_from_frame(frame)

    # GPT elaboration
    prompt = (
    "You are an assistant describing classroom visuals for a visually impaired student. "
    "Report exact text verbatim when present, and avoid speculating. "
    f"BLIP caption: “{caption}”. "
    f"OCR text detected: {ocr_text or '[none]'}.\n"
    "If OCR text exists, say it plainly (e.g., “Whiteboard reads: 'TEST'.”)."
)


    detail = ask_openai(prompt)
    line = f"[Visual Description: {detail}]"
    transcript_lines.append(line)
    print("🔊", line)
    capturing.clear()

class AudioGenerator:
    def __init__(self, rate=16000, chunk_size=1024):
        self.rate = rate
        self.chunk = chunk_size
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        self.frames = []

    def __iter__(self):
        return self

    def __next__(self):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        self.frames.append(data)
        return data

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    global capturing

    if event.end_of_turn and event.turn_is_formatted:
        text = event.transcript.strip()
        transcript_lines.append(text)

        lower = text.lower()
        # fuzzy trigger matching with word boundaries
        if any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in VISUAL_TRIGGERS) and not capturing.is_set():
            print(f"📌 Visual trigger detected: '{text}'")
            capturing.set()
            idx = get_next_index()
            threading.Thread(target=capture_and_describe, args=(idx,), daemon=True).start()

    conf = getattr(event, "end_of_turn_confidence", None)
    print(f"Turn: '{event.transcript}'" + (f" (Confidence: {conf:.2f})" if conf is not None else ""))

    if event.end_of_turn and not event.turn_is_formatted:
        self.set_params(StreamingSessionParameters(format_turns=True))

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"Session terminated: {event.audio_duration_seconds:.2f}s of audio")

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error: {error}")

def get_next_index():
    files = glob.glob("session*.wav")
    nums = []
    pattern = re.compile(r"session(\d+)\.wav$")
    for f in files:
        m = pattern.match(os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1

def main():
    threading.Thread(target=frame_grabber, daemon=True).start()

    client = StreamingClient(
        StreamingClientOptions(api_key=API_KEY, api_host="streaming.assemblyai.com")
    )
    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(StreamingParameters(sample_rate=16000, format_turns=True))

    print("Speak into your mic. Press Ctrl+C to stop.")

    mic = AudioGenerator(rate=16000, chunk_size=1024)
    try:
        client.stream(mic)
    except KeyboardInterrupt:
        print("\nInterrupted by user—shutting down.")
    except Exception as e:
        print(f"Streaming error: {e}")
    finally:
        global _frame_grabber_stop
        _frame_grabber_stop = True

        # Wait up to 5 seconds for any visual capture to finish
        wait_deadline = time.time() + 5
        while capturing.is_set() and time.time() < wait_deadline:
            time.sleep(0.1)

        idx = get_next_index()
        wav_filename = f"session{idx}.wav"
        txt_filename = f"transcript{idx}.txt"

        mic.close()
        with wave.open(wav_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(mic.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(mic.rate)
            wf.writeframes(b"".join(mic.frames))

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

        from assemblyai import Transcriber, TranscriptionConfig, SummarizationModel, SummarizationType
        summary_filename = f"summary{idx}.txt"
        config = TranscriptionConfig(
            summarization=True,
            summary_model=SummarizationModel.informative,
            summary_type=SummarizationType.bullets,
        )

        print("Generating AI summary… press Ctrl+C to cancel if it hangs.")
        try:
            summary_job = Transcriber().transcribe(wav_filename, config)
            summary = summary_job.summary or ""
            with open(summary_filename, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Saved summary to {summary_filename}")
        except KeyboardInterrupt:
            print("Summary generation interrupted by user.")
        except Exception as e:
            print(f"Summary generation failed: {e}")

        client.disconnect(terminate=True)
        print(f"Done — saved {wav_filename}, {txt_filename}", end="")
        if os.path.exists(summary_filename):
            print(f", and {summary_filename}")
        else:
            print("")

if __name__ == "__main__":
    main()
