# podcast_voice_session.py
"""
Voice-first flow to study past lectures:
 1) Listen for a VOICE COMMAND like “help me understand lecture 30” (AssemblyAI).
 2) Resolve that to a transcript file using PodcastRouter.resolve_only().
 3) Run a VOICE podcast:
    - Primer + host question are SPOKEN (and printed)
    - Student replies by voice (AssemblyAI)
    - Mentor answers are SPOKEN (and printed); loop for N turns

Improvements:
 - No leftover WAVs in your project (temporary files auto-removed).
 - Better pacing: clause-level speaking + configurable pauses +
   optional wait for ~3s of room silence before “Your turn.”
 - More realistic voice option: set VOICE_PROVIDER=openai for OpenAI TTS.

Requirements (once in your venv):
  pip install openai python-dotenv assemblyai pyttsx3 pyaudio numpy requests

.env:
  OPENAI_API_KEY=sk-...
  API_KEY=<your AssemblyAI key>
  # Optional (for better TTS):
  VOICE_PROVIDER=openai         # or omit/pyttsx3
  OPENAI_TTS_MODEL=gpt-4o-mini-tts
  OPENAI_TTS_VOICE=alloy        # alloy | verse | aria | etc.
  HOST_SILENCE_SECS=3.0         # seconds of room silence to wait after host speaks
"""

import os
import io
import re
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

from podcast_mode_router import PodcastRouter  # only used to RESOLVE the transcript path

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

VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "pyttsx3").lower()  # "pyttsx3" or "openai"
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")
HOST_SILENCE_SECS = float(os.getenv("HOST_SILENCE_SECS", "3.0"))

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
        min_talk_seconds: float = 0.6,
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
        # write wave header + frames
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
# TTS: pyttsx3 provider
# ---------------------------

class Pyttsx3TTS:
    def __init__(self, voice_substr: Optional[str] = None, rate_factor: float = 0.9):
        self.eng = pyttsx3.init()
        # voice selection (best-effort)
        if voice_substr:
            for v in self.eng.getProperty("voices"):
                if voice_substr.lower() in (v.name or "").lower():
                    self.eng.setProperty("voice", v.id)
                    break
        # pacing
        rate = self.eng.getProperty("rate")
        self.eng.setProperty("rate", max(120, int(rate * rate_factor)))

    def speak(self, text: str):
        self.eng.say(text)
        self.eng.runAndWait()


# ---------------------------
# TTS: OpenAI provider (more natural)
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
        }

    def speak(self, text: str):
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "format": "wav",
        }
        r = self._session.post(self._url, headers=self._headers, json=payload, timeout=120)
        r.raise_for_status()
        audio_bytes = r.content
        self._play_wav_bytes(audio_bytes)

    @staticmethod
    def _play_wav_bytes(b: bytes):
        import io
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


# ---------------------------
# Speaker facade (clause pacing + silence wait)
# ---------------------------

def _split_into_clauses(text: str) -> List[str]:
    # split on sentence-ish punctuation, keep simple
    parts = re.split(r'([.!?;:])', text)
    chunks = []
    buf = ""
    for part in parts:
        if part in ".!?;:":
            buf += part
            chunks.append(buf.strip())
            buf = ""
        else:
            buf += part
    if buf.strip():
        chunks.append(buf.strip())
    # further split long clauses for breath
    final = []
    for c in chunks:
        if len(c) > 160:
            # split on commas if very long
            sub = re.split(r'(,)', c)
            agg = ""
            for s in sub:
                agg += s
                if len(agg) > 80:
                    final.append(agg.strip())
                    agg = ""
            if agg.strip():
                final.append(agg.strip())
        else:
            final.append(c)
    return [c for c in final if c]


class Speaker:
    """Text + Voice with natural pauses and optional room-silence gating."""
    def __init__(self, provider: str = VOICE_PROVIDER, silence_secs: float = HOST_SILENCE_SECS):
        self.silence_secs = silence_secs
        if provider == "openai":
            self.tts = OpenAITTS()
        else:
            # Pick a Windows voice if you want (e.g., "Zira", "David", etc.)
            self.tts = Pyttsx3TTS(voice_substr=os.getenv("PYTTSX3_VOICE", None))

    def say_block(self, text: str, between_clause_pause: float = 0.25):
        print(text)
        for clause in _split_into_clauses(text):
            self.tts.speak(clause)
            time.sleep(between_clause_pause)

    def duet_host(self, host_line: str):
        self.say_block(f"Host: {host_line}")
        self.wait_for_room_silence(self.silence_secs, timeout=max(5.0, self.silence_secs + 2.0))

    def duet_mentor(self, mentor_line: str):
        self.say_block(f"Mentor: {mentor_line}")
        # short breath after mentor
        time.sleep(0.6)

    @staticmethod
    def wait_for_room_silence(required_secs: float, timeout: float, threshold: int = 450, chunk_ms: int = 100):
        """
        Wait until ambient mic input is 'quiet' (RMS below threshold) for required_secs.
        Fallback stop at 'timeout' seconds so we never hang if the room is noisy.
        """
        if required_secs <= 0:
            return
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=int(16000 * chunk_ms / 1000))
        quiet_ms = 0
        waited = 0.0
        try:
            while waited < timeout and quiet_ms < required_secs * 1000:
                data = stream.read(int(16000 * chunk_ms / 1000), exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                rms = int(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                if rms < threshold:
                    quiet_ms += chunk_ms
                else:
                    quiet_ms = 0
                time.sleep(chunk_ms / 1000.0)
                waited += chunk_ms / 1000.0
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


# ---------------------------
# LLM (host/mentor brains)
# ---------------------------

class MentorLLM:
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

    def kickoff(self, transcript: str, max_words: int = 110) -> Tuple[str, str]:
        sys = {
            "role": "system",
            "content": (
                "You are a friendly, concise 'Note Coach' helping a student learn from a lecture transcript. "
                "Podcast vibe—curious but focused. Keep language plain. No fluff. Keep each field <= 120 words."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Summarize the key idea in <= {} words, then craft ONE curious starter question to check understanding. "
                "Return JSON with keys 'primer' and 'host_question'.\n\nTranscript:\n{}"
            ).format(max_words, transcript[:12000])
        }
        out = self._chat([sys, user], temperature=0.4, max_tokens=400)
        try:
            j = json.loads(out)
            return j.get("primer", ""), j.get("host_question", "What part should we unpack first?")
        except Exception:
            return out, "What part should we unpack first?"

    def next_turn(self, transcript: str, history: List[Dict[str, str]], student_reply: str) -> Tuple[str, str]:
        sys = {
            "role": "system",
            "content": (
                "You are the Note Coach. Keep answers crisp and intuitive. "
                "Use the transcript as needed; avoid long quotes. If confused, give a tiny example."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Transcript (skim as needed):\n{}\n\n"
                "Chat so far (list of dicts with 'role' and 'text'):\n{}\n\n"
                "Student just said: {}\n\n"
                "1) Write a natural host follow-up question to deepen understanding.\n"
                "2) Then write a clear mentor answer.\n"
                "Return JSON with keys 'host_question' and 'mentor_answer'. Keep each <= ~120 words."
            ).format(transcript[:12000], json.dumps(history[-8:], ensure_ascii=False), student_reply)
        }
        out = self._chat([sys, user], temperature=0.5, max_tokens=600)
        try:
            j = json.loads(out)
            return j.get("host_question", "What should we tackle next?"), j.get("mentor_answer", out)
        except Exception:
            return "What should we tackle next?", out


# ---------------------------
# Session Orchestrator
# ---------------------------

@dataclass
class VoicePodcastSession:
    root: str = "."
    model: str = "gpt-4"
    turns: int = 4

    speaker: Speaker = field(init=False)
    mentor: MentorLLM = field(init=False)
    history: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.speaker = Speaker()
        self.mentor = MentorLLM(model=self.model)

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
            audio = mic.record_until_silence(overall_timeout=22.0)
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

        # 2) Kickoff
        primer, host_q = self.mentor.kickoff(transcript)
        if primer:
            self.speaker.say_block(primer, between_clause_pause=0.3)
            # small breath after primer
            time.sleep(0.8)
        self.speaker.duet_host(host_q)

        # 3) Conversation loop (voice in/out)
        for _ in range(self.turns):
            self.speaker.say_block("Your turn.")
            student_text = self._listen_reply()
            if not student_text or student_text.lower().strip() in {"stop", "exit", "quit", "cancel"}:
                self.speaker.say_block("Ending session.")
                break

            self.history.append({"role": "student", "text": student_text})
            host, mentor = self.mentor.next_turn(transcript, self.history, student_text)
            self.history.append({"role": "host", "text": host})
            self.history.append({"role": "mentor", "text": mentor})

            # Speak + print both
            self.speaker.duet_host(host)
            self.speaker.duet_mentor(mentor)

        self.speaker.say_block("That's a wrap. Say 'latest' next time to jump to your most recent lecture.")


if __name__ == "__main__":
    try:
        VoicePodcastSession(root=".", model="gpt-4", turns=4).run()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
