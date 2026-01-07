# podcast_voice_session.py
"""
Voice-first flow to study past lectures:
 1) VOICE COMMAND picks transcript (AssemblyAI): e.g., “lecture 30”, “open transcript 30”, or “latest”.
 2) Single-voice COACH runs a podcast-like chat:
    - Coach gives brief context AND asks one good question each turn.
    - Student replies by voice; system transcribes and continues.
    - Fully voiced AND printed for testing.
    - Say a STOP phrase to end (e.g., "stop", "quit", "that's all", "end session").

Improvements:
 - One COACH voice (merged Host + Mentor).
 - Faster pacing (set in .env): TTS_RATE_FACTOR, CLAUSE_PAUSE, HOST_SILENCE_SECS.
 - Robust OpenAI TTS (WAV/MP3) with fallback to pyttsx3.
 - No leftover WAVs in your project (temp files auto-removed).

Requirements (once in your venv):
  pip install openai python-dotenv assemblyai pyttsx3 pyaudio numpy requests playsound==1.2.2

.env:
  OPENAI_API_KEY=sk-...
  API_KEY=<your AssemblyAI key>

  # Optional, for more natural TTS (OpenAI)
  VOICE_PROVIDER=openai          # or omit/pyttsx3
  OPENAI_TTS_MODEL=gpt-4o-mini-tts
  OPENAI_TTS_VOICE=alloy         # try aria, verse, etc.

  # Pacing
  TTS_RATE_FACTOR=1.25           # ~25% faster
  CLAUSE_PAUSE=0.20              # seconds between clauses
  HOST_SILENCE_SECS=2.0          # wait for this much room quiet after Coach speaks

  # Optional stop words customization (comma-separated)
  STOP_WORDS=stop,quit,exit,cancel,that's all,thats all,that is all,end session,goodbye,bye,finish,done
"""

import os
import re
import io
import time
import json
import wave
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pyaudio
import pyttsx3
import requests
from dotenv import load_dotenv

from podcast_mode_router import PodcastRouter  # used for resolving the transcript path

import assemblyai as aai
import openai

# ---------------------------
# Config / env
# ---------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIA_KEY = os.getenv("API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set (put it in .env or set in shell)")
if not AIA_KEY:
    raise RuntimeError("API_KEY (AssemblyAI) not set (put it in .env or set in shell)")

openai.api_key = OPENAI_API_KEY
aai.settings.api_key = AIA_KEY

VOICE_PROVIDER     = os.getenv("VOICE_PROVIDER", "pyttsx3").lower()  # "pyttsx3" or "openai"
OPENAI_TTS_MODEL   = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE   = os.getenv("OPENAI_TTS_VOICE", "alloy")

# pacing knobs
TTS_RATE_FACTOR    = float(os.getenv("TTS_RATE_FACTOR", "1.25"))   # ~25% faster
CLAUSE_PAUSE       = float(os.getenv("CLAUSE_PAUSE", "0.20"))      # seconds between clauses
HOST_SILENCE_SECS  = float(os.getenv("HOST_SILENCE_SECS", "2.0"))  # post-coach quiet wait

STOP_WORDS = [w.strip().lower() for w in os.getenv(
    "STOP_WORDS",
    "stop,quit,exit,cancel,that's all,thats all,that is all,end session,goodbye,bye,finish,done"
).split(",") if w.strip()]

def is_stop_phrase(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(w in t for w in STOP_WORDS)

# ---------------------------
# Audio I/O helpers
# ---------------------------

class MicRecorder:
    """Simple energy-based VAD recorder to capture one utterance."""
    def __init__(self, rate=16000, chunk=1024, device_index=None):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk,
        )

    def record_until_silence(
        self,
        min_talk_seconds: float = 0.3,
        trailing_silence: float = 0.8,
        overall_timeout: float = 12.0,
        start_threshold: int = 800,
        stop_threshold: int = 500,
    ) -> bytes:
        frames: List[bytes] = []
        spoke = False
        min_frames = int(self.rate / self.chunk * min_talk_seconds)
        silence_frames_needed = int(self.rate / self.chunk * trailing_silence)
        silent_count = 0
        t0 = time.time()

        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)

            samples = np.frombuffer(data, dtype=np.int16)
            rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

            if not spoke and rms > start_threshold:
                spoke = True

            if spoke:
                if rms < stop_threshold:
                    silent_count += 1
                else:
                    silent_count = 0

                if len(frames) > min_frames and silent_count >= silence_frames_needed:
                    break

            if time.time() - t0 > overall_timeout:
                break

        return b"".join(frames)

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()


def transcribe_one_utterance_bytes(pcm_bytes: bytes, rate: int = 16000) -> str:
    """Writes to a temp WAV, transcribes, removes the temp file."""
    # create WAV in a temp file (auto-cleaned)
    with tempfile.NamedTemporaryFile(prefix="voice_", suffix=".wav", delete=False) as tf:
        temp_wav = tf.name
    try:
        p = pyaudio.PyAudio()
        sampwidth = p.get_sample_size(pyaudio.paInt16)
        with wave.open(temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sampwidth)
            wf.setframerate(rate)
            wf.writeframes(pcm_bytes)
        p.terminate()

        # transcribe with AssemblyAI
        from assemblyai import Transcriber as AAITranscriber
        job = AAITranscriber().transcribe(temp_wav)
        return (job.text or "").strip()
    except Exception as e:
        return f"(transcription error: {e})"
    finally:
        try:
            os.remove(temp_wav)
        except OSError:
            pass


# ---------------------------
# TTS: pyttsx3 provider (offline)
# ---------------------------

class Pyttsx3TTS:
    def __init__(self, voice_substr: Optional[str] = None, rate_factor: float = TTS_RATE_FACTOR):
        self.eng = pyttsx3.init()
        if voice_substr:
            for v in self.eng.getProperty("voices"):
                if voice_substr.lower() in (v.name or "").lower():
                    self.eng.setProperty("voice", v.id)
                    break
        rate = self.eng.getProperty("rate")
        self.eng.setProperty("rate", max(120, int(rate * rate_factor)))

    def speak(self, text: str):
        self.eng.say(text)
        self.eng.runAndWait()


# ---------------------------
# TTS: OpenAI provider (WAV/MP3 with fallback)
# ---------------------------

class OpenAITTS:
    def __init__(self, model: str = OPENAI_TTS_MODEL, voice: str = OPENAI_TTS_VOICE):
        self.model = model
        self.voice = voice
        self._session = requests.Session()
        self._url = "https://api.openai.com/v1/audio/speech"
        self._headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        self._fallback = Pyttsx3TTS(voice_substr=os.getenv("PYTTSX3_VOICE", None))

    def speak(self, text: str):
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "format": "wav",  # server may still return mp3
        }
        try:
            r = self._session.post(self._url, headers=self._headers, json=payload, timeout=120)
            r.raise_for_status()
            audio_bytes = r.content
            ctype = r.headers.get("Content-Type", "")
            # WAV?
            if "audio/wav" in ctype or (len(audio_bytes) >= 4 and audio_bytes[:4] == b"RIFF"):
                return self._play_wav_bytes(audio_bytes)
            # MP3?
            if "audio/mpeg" in ctype or audio_bytes[:3] == b"\xFF\xFB" or audio_bytes[:2] == b"\xFF\xF3":
                return self._play_mp3_bytes(audio_bytes)
            print(f"[TTS] Unexpected content-type ({ctype}). Falling back to pyttsx3.")
            self._fallback.speak(text)
        except Exception as e:
            print(f"[TTS] OpenAI TTS error: {e}. Falling back to pyttsx3.")
            self._fallback.speak(text)

    @staticmethod
    def _play_wav_bytes(b: bytes):
        with wave.open(io.BytesIO(b), "rb") as wf:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()

    @staticmethod
    def _play_mp3_bytes(b: bytes):
        import tempfile
        from playsound import playsound
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(b)
                path = tf.name
            playsound(path)  # blocking playback
        finally:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------
# Speaker: clause pacing + silence gate + Coach wrappers
# ---------------------------

def _split_into_clauses(text: str) -> List[str]:
    parts = re.split(r'([.!?;:])', text)
    chunks, buf = [], ""
    for part in parts:
        if part in ".!?;:":
            buf += part
            chunks.append(buf.strip()); buf = ""
        else:
            buf += part
    if buf.strip():
        chunks.append(buf.strip())
    final = []
    for c in chunks:
        if len(c) > 160:
            sub = re.split(r'(,)', c)
            agg = ""
            for s in sub:
                agg += s
                if len(agg) > 80:
                    final.append(agg.strip()); agg = ""
            if agg.strip(): final.append(agg.strip())
        else:
            final.append(c)
    return [c for c in final if c]

class Speaker:
    """Text + Voice with natural pauses and room-silence gating."""
    def __init__(self, provider: str = VOICE_PROVIDER, silence_secs: float = HOST_SILENCE_SECS):
        self.silence_secs = silence_secs
        if provider == "openai":
            self.tts = OpenAITTS()
        else:
            self.tts = Pyttsx3TTS(voice_substr=os.getenv("PYTTSX3_VOICE", None))

    def say_block(self, text: str, between_clause_pause: Optional[float] = None):
        print(text)
        pause = CLAUSE_PAUSE if between_clause_pause is None else between_clause_pause
        for clause in _split_into_clauses(text):
            self.tts.speak(clause)
            time.sleep(pause)

    def say_coach(self, line: str):
        self.say_block(f"Host: {line}")
        self.wait_for_room_silence(self.silence_secs, timeout=max(5.0, self.silence_secs))

    @staticmethod
    def wait_for_room_silence(required_secs: float, timeout: float, threshold: int = 450, chunk_ms: int = 100):
        if required_secs <= 0: return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                        frames_per_buffer=int(16000 * chunk_ms / 1000))
        quiet_ms = 0; waited = 0.0
        try:
            while waited < timeout and quiet_ms < required_secs * 1000:
                data = stream.read(int(16000 * chunk_ms / 1000), exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                quiet_ms = quiet_ms + chunk_ms if rms < threshold else 0
                time.sleep(chunk_ms / 1000.0); waited += chunk_ms / 1000.0
        finally:
            stream.stop_stream(); stream.close(); p.terminate()


# ---------------------------
# LLM: Single COACH (context + one question per turn)
# ---------------------------

class CoachLLM:
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        openai.api_key = OPENAI_API_KEY

    def _chat(self, messages: List[Dict], temperature: float = 0.5, max_tokens: int = 600) -> str:
        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def kickoff(self, transcript: str, max_words: int = 90) -> str:
        sys = {
            "role": "system",
            "content": (
                "You are a single-voice 'Host' for a podcast-style study session. "
                "Each turn, you:\n"
                " - give a tiny bit of context (1–2 sentences), and\n"
                " - ask ONE simple, engaging question to move the conversation.\n"
                "Keep language plain, friendly, and <= 120 words total per turn."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Using these notes (skim as needed), start the session with 1–2 sentences of context, "
                "then ask ONE concise question that invites the student to pick a focus area. "
                "Return ONLY the final line the Host should say.\n\nTranscript:\n{}"
            ).format(transcript[:12000])
        }
        out = self._chat([sys, user], temperature=0.4, max_tokens=300)
        return out

    def next_turn(self, transcript: str, history: List[Dict[str, str]], student_reply: str) -> str:
        sys = {
            "role": "system",
            "content": (
                "You are the 'Host' (one voice). "
                "Respond to the student's latest message with: (a) brief helpful context, "
                "then (b) ONE question that advances the discussion. "
                "Avoid repeating the same question. Keep <= 120 words total. "
                "If the student sounds done (e.g., 'that's all'), politely confirm ending."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Transcript (skim as needed):\n{}\n\n"
                "Chat so far (list of dicts with 'role' and 'text'):\n{}\n\n"
                "Student just said: {}\n\n"
                "Return ONLY the final line the Coach should say."
            ).format(transcript[:12000], json.dumps(history[-8:], ensure_ascii=False), student_reply)
        }
        out = self._chat([sys, user], temperature=0.6, max_tokens=350)
        return out


# ---------------------------
# Session Orchestrator
# ---------------------------

@dataclass
class VoicePodcastSession:
    root: str = "."
    model: str = "gpt-4"
    turns: int = 4

    speaker: Speaker = field(init=False)
    coach: CoachLLM = field(init=False)
    history: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.speaker = Speaker()
        self.coach = CoachLLM(model=self.model)

    def _listen_for_command(self) -> str:
        self.speaker.say_block("Say the lecture you want, like 'lecture 30' or 'open transcript 30' or 'latest'.")
        mic = MicRecorder()
        try:
            audio = mic.record_until_silence(overall_timeout=10.0)
        finally:
            mic.close()
        text = transcribe_one_utterance_bytes(audio)
        print(f"🎧 Heard: {text}")
        return text

    def _listen_reply(self) -> str:
        mic = MicRecorder()
        try:
            audio = mic.record_until_silence(
                min_talk_seconds=1.0,   # more generous to avoid cutoffs
                trailing_silence=1.2,
                overall_timeout=25.0,
                start_threshold=700,
                stop_threshold=500,
            )
        finally:
            mic.close()
        text = transcribe_one_utterance_bytes(audio)
        print(f"👤 You said: {text}")
        return text

    def run(self):
        # 1) Command phase: pick the file
        router = PodcastRouter(root=self.root, model=self.model, tts=False, turns=self.turns)
        cmd_text = self._listen_for_command()
        path = router.resolve_only(cmd_text)
        if not path:
            latest = router.catalog.latest_path()
            if latest:
                path = latest
                self.speaker.say_block(f"I couldn't find that. I'll load the latest: {os.path.basename(path)}.")
            else:
                self.speaker.say_block("I couldn't find any transcripts. Please create one first.")
                return

        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        self.speaker.say_block(f"Opening {os.path.basename(path)}.")

        # 2) Kickoff (single Coach line)
        coach_line = self.coach.kickoff(transcript)
        self.history.append({"role": "coach", "text": coach_line})
        self.speaker.say_coach(coach_line)

        # 3) Conversation loop (voice in/out)
        for _ in range(self.turns):
            self.speaker.say_block("Your turn.")
            student_text = self._listen_reply()
            if is_stop_phrase(student_text):
                self.speaker.say_block("Got it. Ending session. Thanks for studying together!")
                break
            if not student_text:
                self.speaker.say_block("I didn't catch that. Let's pause here.")
                break

            self.history.append({"role": "student", "text": student_text})
            coach_line = self.coach.next_turn(transcript, self.history, student_text)
            self.history.append({"role": "coach", "text": coach_line})

            self.speaker.say_coach(coach_line)



if __name__ == "__main__":
    try:
        VoicePodcastSession(root=".", model="gpt-4", turns=4).run()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
