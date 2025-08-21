# podcast_voice_session.py
"""
Voice-first flow to study past lectures:
1) Listen for a VOICE COMMAND like “help me understand lecture 29” (AssemblyAI).
2) Resolve that to a transcript file using PodcastRouter.resolve_only().
3) Run a VOICE podcast session:
   - Primer + host question are SPOKEN (pyttsx3)
   - Student replies by voice (mic), transcribed (AssemblyAI)
   - Mentor answers are SPOKEN; loop for N turns

Requirements:
  pip install openai python-dotenv assemblyai pyttsx3 pyaudio numpy
.env:
  OPENAI_API_KEY=sk-...
  API_KEY=<your AssemblyAI key>
"""

import os
import time
import json
import wave
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import pyaudio
import pyttsx3
from dotenv import load_dotenv

# router (for resolving the file to open)
from podcast_mode_router import PodcastRouter

# AssemblyAI
import assemblyai as aai

# OpenAI
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIA_KEY = os.getenv("API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set (put it in .env or set in shell)")
if not AIA_KEY:
    raise RuntimeError("API_KEY (AssemblyAI) not set (put it in .env or set in shell)")

openai.api_key = OPENAI_API_KEY
aai.settings.api_key = AIA_KEY


# ---------------------------
# Audio I/O
# ---------------------------

class MicRecorder:
    """Simple energy-based VAD recorder."""
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

    def save_wav(self, pcm_bytes: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(pcm_bytes)

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()


# ---------------------------
# TTS
# ---------------------------

class Speaker:
    def __init__(self, rate_factor: float = 0.9):
        self.eng = pyttsx3.init()
        rate = self.eng.getProperty("rate")
        self.eng.setProperty("rate", max(120, int(rate * rate_factor)))

    def say(self, text: str):
        print(text)
        self.eng.say(text)
        self.eng.runAndWait()

    def duet(self, host_line: str, mentor_line: str):
        self.say(f"Host: {host_line}")
        time.sleep(0.12)
        self.say(f"Mentor: {mentor_line}")


# ---------------------------
# STT via AssemblyAI (batch)
# ---------------------------

class Transcriber:
    def __init__(self):
        aai.settings.api_key = AIA_KEY

    def transcribe_wav(self, path: str) -> str:
        try:
            from assemblyai import Transcriber as AAITranscriber
            job = AAITranscriber().transcribe(path)
            return (job.text or "").strip()
        except Exception as e:
            return f"(transcription error: {e})"


# ---------------------------
# LLM
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

    def kickoff(self, transcript: str, max_words: int = 100) -> Tuple[str, str]:
        sys = {
            "role": "system",
            "content": (
                "You are a friendly, concise 'Note Coach' helping a student learn from a lecture transcript. "
                "Vibe: educational podcast co-host—curious but focused. Keep language plain. No fluff."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Summarize the key idea of these notes in <= {} words, then craft ONE curious starter question "
                "to check understanding of the central concept. "
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
                "You are the Note Coach. Keep answers crisp and build intuition. "
                "Use the transcript as needed; don't quote long blocks. "
                "If student is confused, give a tiny example. Avoid math markup unless critical."
            )
        }
        user = {
            "role": "user",
            "content": (
                "Transcript (skim as needed):\n{}\n\n"
                "Chat so far (list of small dicts with 'role' and 'text'):\n{}\n\n"
                "Student just said: {}\n\n"
                "1) Write a natural host follow-up question to deepen understanding.\n"
                "2) Then write a clear mentor answer.\n"
                "Return JSON with keys 'host_question' and 'mentor_answer'. Keep each under ~120 words."
            ).format(transcript[:12000], json.dumps(history[-8:], ensure_ascii=False), student_reply)
        }
        out = self._chat([sys, user], temperature=0.5, max_tokens=600)
        try:
            j = json.loads(out)
            return j.get("host_question", "What should we tackle next?"), j.get("mentor_answer", out)
        except Exception:
            return "What should we tackle next?", out


# ---------------------------
# Voice Session Orchestrator
# ---------------------------

@dataclass
class VoicePodcastSession:
    root: str = "."
    model: str = "gpt-4"
    turns: int = 4

    speaker: Speaker = field(init=False)
    transcriber: Transcriber = field(init=False)
    mentor: MentorLLM = field(init=False)
    history: List[Dict[str, str]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.speaker = Speaker()
        self.transcriber = Transcriber()
        self.mentor = MentorLLM(model=self.model)

    def _listen_for_command(self) -> str:
        self.speaker.say("Say the lecture you want, for example: 'lecture 29' or 'open transcript 29' or 'latest'.")
        mic = MicRecorder()
        try:
            audio = mic.record_until_silence(overall_timeout=8.0)
            wav = f"voice_command_{int(time.time())}.wav"
            mic.save_wav(audio, wav)
        finally:
            mic.close()
        text = self.transcriber.transcribe_wav(wav)
        print(f"🎧 Heard: {text}")
        return text

    def _listen_reply(self) -> str:
        mic = MicRecorder()
        try:
            audio = mic.record_until_silence(overall_timeout=20.0)
            wav = f"student_reply_{int(time.time())}.wav"
            mic.save_wav(audio, wav)
        finally:
            mic.close()
        text = self.transcriber.transcribe_wav(wav)
        print(f"👤 You said: {text}")
        return text

    def run(self):
        # 1) Command phase: pick the file
        router = PodcastRouter(root=self.root, model=self.model, tts=False, turns=self.turns)
        cmd_text = self._listen_for_command()
        path = router.resolve_only(cmd_text)
        if not path:
            # fallback: try latest; else list
            latest = router.catalog.latest_path()
            if latest:
                path = latest
                self.speaker.say(f"I couldn't find that. I'll load the latest: {os.path.basename(path)}")
            else:
                self.speaker.say("I couldn't find any transcripts. Please create one first.")
                return

        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        self.speaker.say(f"Opening {os.path.basename(path)}.")

        # 2) Kickoff
        primer, host_q = self.mentor.kickoff(transcript)
        if primer:
            self.speaker.say(primer)
        self.speaker.say(f"Host: {host_q}")
        self.history.append({"role": "host", "text": host_q})

        # 3) Conversation loop (voice in/out)
        for _ in range(self.turns):
            self.speaker.say("Your turn.")
            student_text = self._listen_reply()
            if not student_text or student_text.lower().strip() in {"stop", "exit", "quit", "cancel"}:
                self.speaker.say("Ending session.")
                break

            self.history.append({"role": "student", "text": student_text})
            host, mentor = self.mentor.next_turn(transcript, self.history, student_text)
            self.history.append({"role": "host", "text": host})
            self.history.append({"role": "mentor", "text": mentor})

            self.speaker.duet(host, mentor)

        self.speaker.say("That's a wrap. Say 'latest' next time to jump to your most recent lecture.")


# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    try:
        VoicePodcastSession(root=".", model="gpt-4", turns=4).run()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
