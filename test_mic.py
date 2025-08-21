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
from podcast_mode_router import PodcastRouter


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
latest_frame = None
_frame_grabber_stop = False
def frame_grabber():
    global latest_frame, _frame_grabber_stop
    cap = cv2.VideoCapture(0)
    # optionally set a lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not _frame_grabber_stop:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame
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

router = PodcastRouter(root=".", model="gpt-4", tts=False, turns=4)

def my_transcribe_once():
    # Return ONE line from your STT pipeline, e.g. latest user command
    return "can you help me understand lecture14"

router.route_and_start_by_audio(my_transcribe_once)
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
    raw = Image.fromarray(frame[..., ::-1]).convert("RGB")
    inputs = processor(raw, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # GPT elaboration
    prompt = (
    "You are an assistant describing classroom visuals for a visually impaired student. "
    f"The student cannot see the board or screen, so your job is to clearly describe only the key visual content that is relevant to the lecture. "
    "Ignore the teacher’s appearance or classroom environment. Focus entirely on what's being displayed or held up, such as images, graphs, text, or diagrams. "
    f"Here is a short caption from the image: “{caption}”. Expand it into a clear and concise educational description the student can recall later."
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
