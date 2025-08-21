# podcast_mode_soft.py
"""
Software-only 'Talk to My Notes' coach.
- No GPIO, no mic/VAD required.
- Default input: student types a reply (works everywhere).
- Optional: text-to-speech via pyttsx3 (toggleable).
- LLM: OpenAI Chat API (set OPENAI_API_KEY).

Usage
-----
$ python podcast_mode_soft.py --transcript notes.txt --turns 4 --tts off
$ python podcast_mode_soft.py --transcript notes.txt --turns 4 --tts on

Integrating in your app
-----------------------
from podcast_mode_soft import PodcastCoach

coach = PodcastCoach(model="gpt-4", tts=False)
coach.start_session(transcript_text, max_turns=4)

If you have your own STT already and want to preserve the "chat with voice" vibe,
call `coach.next_turn(transcript_text, student_reply)` in your pipeline whenever
you have a new student utterance (string). The coach returns (host_question, mentor_answer).
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
# Optional: OpenAI + pyttsx3 are soft dependencies.
try:
    import openai  # legacy API for broader compatibility
except Exception:
    openai = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# ---------------------------
# Small utilities
# ---------------------------

def _safe_openai():
    if openai is None:
        raise RuntimeError("openai package not installed. `pip install openai`")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key


def _truncate(s: str, limit: int = 12000) -> str:
    return s if len(s) <= limit else s[:limit]


# ---------------------------
# Speakers
# ---------------------------

class BaseSpeaker:
    def say(self, text: str) -> None:
        print(text)

    def duet(self, host_line: str, mentor_line: str) -> None:
        self.say(f"Host: {host_line}")
        self.say(f"Mentor: {mentor_line}")


class TTSSpeaker(Basepeaker := BaseSpeaker):  # alias for clarity
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not installed. `pip install pyttsx3`")
        self.eng = pyttsx3.init()
        # slightly slower for clarity
        rate = self.eng.getProperty("rate")
        self.eng.setProperty("rate", max(120, int(rate * 0.9)))

    def say(self, text: str) -> None:
        super().say(text)  # also print to console
        self.eng.say(text)
        self.eng.runAndWait()


# ---------------------------
# LLM Brain
# ---------------------------

@dataclass
class NoteCoachBrain:
    model: str = "gpt-4"
    system_preamble: str = (
        "You are a friendly, concise 'Note Coach' who helps a student learn from a lecture transcript. "
        "Vibe: educational podcast co-host—curious but focused. Keep language plain. No fluff."
    )

    def _chat(self, messages: List[Dict], temperature: float = 0.5, max_tokens: int = 600) -> str:
        _safe_openai()
        resp = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30
        )
        return resp.choices[0].message.content.strip()

    def kickoff(self, transcript: str, max_words: int = 100) -> Tuple[str, str]:
        """Return (primer, host_question)."""
        sys = {"role": "system", "content": self.system_preamble}
        user = {
            "role": "user",
            "content": (
                "Summarize the key idea of these notes in <= {} words, then craft ONE curious starter question "
                "to check understanding of the central concept. "
                "Return JSON with keys 'primer' and 'host_question'.\n\nTranscript:\n{}"
            ).format(max_words, _truncate(transcript))
        }
        out = self._chat([sys, user], temperature=0.4, max_tokens=400)
        try:
            j = json.loads(out)
            primer = j.get("primer") or ""
            host_q = j.get("host_question") or "What part should we unpack first?"
            return primer, host_q
        except Exception:
            # fallback: treat entire output as primer
            return out, "What part should we unpack first?"

    def respond(self, transcript: str, history: List[Dict[str, str]], student_reply: str) -> Tuple[str, str]:
        """Return (host_question, mentor_answer)."""
        sys = {"role": "system", "content": self.system_preamble}
        user = {
            "role": "user",
            "content": (
                "Transcript (skim as needed):\n{}\n\n"
                "Chat so far (list of small dicts with 'role' and 'text'):\n{}\n\n"
                "Student just said: {}\n\n"
                "1) Write a natural host follow-up question to deepen understanding.\n"
                "2) Then write a clear mentor answer.\n"
                "Return JSON with keys 'host_question' and 'mentor_answer'. Keep each under ~120 words."
            ).format(_truncate(transcript), json.dumps(history[-8:], ensure_ascii=False), student_reply)
        }
        out = self._chat([sys, user], temperature=0.5, max_tokens=600)
        try:
            j = json.loads(out)
            return j.get("host_question", "What should we tackle next?"), j.get("mentor_answer", "")
        except Exception:
            return "What should we tackle next?", out


# ---------------------------
# Orchestrator
# ---------------------------

@dataclass
class PodcastCoach:
    model: str = "gpt-4"
    tts: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.brain = NoteCoachBrain(model=self.model)
        self.speaker = TTSSpeaker() if self.tts else BaseSpeaker()

    def start_session(self, transcript_text: str, max_turns: int = 4) -> None:
        """Interactive session: student types replies; coach speaks/prints."""
        primer, host_q = self.brain.kickoff(transcript_text)
        if primer:
            self.speaker.say(primer)
        self.speaker.say(f"Host: {host_q}")
        self.history.append({"role": "host", "text": host_q})

        for _ in range(max_turns):
            student = input("\n👤 You: ").strip()
            if not student:
                break
            self.history.append({"role": "student", "text": student})

            host, mentor = self.brain.respond(transcript_text, self.history, student)
            self.history.append({"role": "host", "text": host})
            self.history.append({"role": "mentor", "text": mentor})
            self.speaker.duet(host, mentor)

        self.speaker.say("That's a wrap. Want to continue later, just run this again.")

    def next_turn(self, transcript_text: str, student_reply: str) -> Tuple[str, str]:
        """
        Non-interactive: call from your own pipeline.
        Returns (host_question, mentor_answer).
        """
        if not self.history:  # lazily kickoff if needed
            primer, host_q = self.brain.kickoff(transcript_text)
            if primer:
                self.history.append({"role": "primer", "text": primer})
            self.history.append({"role": "host", "text": host_q})

        self.history.append({"role": "student", "text": student_reply})
        host, mentor = self.brain.respond(transcript_text, self.history, student_reply)
        self.history.append({"role": "host", "text": host})
        self.history.append({"role": "mentor", "text": mentor})
        return host, mentor


# ---------------------------
# CLI
# ---------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Software-only Podcast Coach")
    ap.add_argument("--transcript", type=str, required=False, help="path to a text file with the lecture transcript")
    ap.add_argument("--turns", type=int, default=4, help="max conversational turns")
    ap.add_argument("--tts", choices=["on", "off"], default="off", help="use text-to-speech (pyttsx3)")
    ap.add_argument("--model", type=str, default="gpt-4", help="OpenAI model name")
    args = ap.parse_args()

    text = ""
    if args.transcript and os.path.exists(args.transcript):
        with open(args.transcript, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = "Today we covered binary search trees, in-order traversal, and their O(log n) average search time."

    coach = PodcastCoach(model=args.model, tts=(args.tts == "on"))
    coach.start_session(text, max_turns=args.turns)


if __name__ == "__main__":
    _cli()
