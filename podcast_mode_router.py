# podcast_mode_router.py
"""
Route-by-voice (or text) to the right transcript, then start a 'podcast' coaching session.
- Software-only. No GPIO/Arduino/Raspberry Pi required.
- Works with typed commands OR your own STT function (plug-in).
- Scans a project folder for transcripts like transcript14.txt, lecture27.txt, notes03.txt, etc.

CLI examples
------------
$ python podcast_mode_router.py --root . --say "help me understand lecture27" --tts off
$ python podcast_mode_router.py --root . --interactive --tts on
$ python podcast_mode_router.py --root /path/to/project --turns 4

Integrating with your app
-------------------------
from podcast_mode_router import PodcastRouter

router = PodcastRouter(root=".", model="gpt-4", tts=False)
router.route_and_start_by_text("can you help me understand lecture27?")

# If you already have STT:
def my_transcribe_once() -> str:
    # return the latest ASR text from your pipeline
    ...

router.route_and_start_by_audio(my_transcribe_once)
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
load_dotenv()
# Soft deps
try:
    import openai
except Exception:
    openai = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# -----------------------------
# Common utils
# -----------------------------

def _safe_openai():
    if openai is None:
        raise RuntimeError("openai package not installed. `pip install openai`")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key


def _truncate(s: str, limit: int = 12000) -> str:
    return s if len(s) <= limit else s[:limit]


# -----------------------------
# Speakers
# -----------------------------

class BaseSpeaker:
    def say(self, text: str) -> None:
        print(text)

    def duet(self, host_line: str, mentor_line: str) -> None:
        self.say(f"Host: {host_line}")
        self.say(f"Mentor: {mentor_line}")


class TTSSpeaker(Basepeaker := BaseSpeaker):
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not installed. `pip install pyttsx3`")
        self.eng = pyttsx3.init()
        rate = self.eng.getProperty("rate")
        self.eng.setProperty("rate", max(120, int(rate * 0.9)))

    def say(self, text: str) -> None:
        super().say(text)
        self.eng.say(text)
        self.eng.runAndWait()


# -----------------------------
# Brain
# -----------------------------

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
            return out, "What part should we unpack first?"

    def respond(self, transcript: str, history: List[Dict[str, str]], student_reply: str) -> Tuple[str, str]:
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


# -----------------------------
# Coach (interaction)
# -----------------------------

@dataclass
class PodcastCoach:
    model: str = "gpt-4"
    tts: bool = False
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        self.brain = NoteCoachBrain(model=self.model)
        self.speaker = TTSSpeaker() if self.tts else BaseSpeaker()

    def start_session(self, transcript_text: str, max_turns: int = 4) -> None:
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
        if not self.history:
            primer, host_q = self.brain.kickoff(transcript_text)
            if primer:
                self.history.append({"role": "primer", "text": primer})
            self.history.append({"role": "host", "text": host_q})

        self.history.append({"role": "student", "text": student_reply})
        host, mentor = self.brain.respond(transcript_text, self.history, student_reply)
        self.history.append({"role": "host", "text": host})
        self.history.append({"role": "mentor", "text": mentor})
        return host, mentor


# -----------------------------
# File catalog + routing
# -----------------------------

@dataclass
class FileCatalog:
    root: str = "."
    index: Dict[str, str] = field(default_factory=dict)

    def scan(self) -> None:
        self.index.clear()
        for fname in os.listdir(self.root):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(self.root, fname)
            base = os.path.splitext(fname)[0].lower()

            # Create aliases
            aliases = set()
            aliases.add(fname.lower())
            aliases.add(base)
            m = re.search(r"(lecture|transcript|notes?)[-_ ]?(\d+)", base)
            if m:
                n = m.group(2).lstrip("0") or "0"
                aliases.add(f"{m.group(1)}{n}")
                aliases.add(f"lecture{n}")
                aliases.add(f"transcript{n}")
                aliases.add(f"notes{n}")

            # Register
            for a in aliases:
                self.index[a] = path

    def resolve(self, utterance: str) -> Optional[str]:
        """Map a free-form utterance to a transcript path."""
        u = utterance.lower()

        # 1) lecture / transcript / notes + number
        m = re.search(r"(lecture|transcript|notes?)[-_ ]?(\d+)", u)
        if m:
            key = f"{m.group(1)}{int(m.group(2))}"
            if key in self.index:
                return self.index[key]

        # 2) explicit filename present
        m = re.search(r"([a-z0-9_\-]+\.txt)", u)
        if m:
            key = m.group(1).lower()
            if key in self.index:
                return self.index[key]

        # 3) substring fuzzy (very light)
        for k in sorted(self.index.keys(), key=len, reverse=True):
            if k in u:
                return self.index[k]

        return None

    def list_human(self) -> str:
        items = sorted(set(self.index.values()))
        lines = ["Found transcripts:"]
        for p in items:
            lines.append(f" - {os.path.basename(p)}")
        return "\n".join(lines)


# -----------------------------
# Router facade
# -----------------------------

@dataclass
class PodcastRouter:
    root: str = "."
    model: str = "gpt-4"
    tts: bool = False
    turns: int = 4
    catalog: FileCatalog = field(init=False)
    coach: PodcastCoach = field(init=False)

    def __post_init__(self):
        self.catalog = FileCatalog(self.root)
        self.catalog.scan()
        self.coach = PodcastCoach(model=self.model, tts=self.tts)

    def route_and_start_by_text(self, command_text: str) -> Optional[str]:
        path = self.catalog.resolve(command_text)
        if not path:
            print(self.catalog.list_human())
            print("Couldn't resolve that request. Try 'lecture14' or 'transcript27.txt'.")
            return None
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        print(f"\n📄 Using transcript: {os.path.basename(path)}\n")
        self.coach.start_session(transcript, max_turns=self.turns)
        return path

    def route_and_start_by_audio(self, transcribe_once_callable) -> Optional[str]:
        """
        Plug in your own STT function that returns one utterance of text.
        Example:
            def my_stt():
                # block until you have one utterance
                return asr_text
            router.route_and_start_by_audio(my_stt)
        """
        if not callable(transcribe_once_callable):
            raise ValueError("transcribe_once_callable must be a callable returning a string utterance.")
        utterance = transcribe_once_callable()
        return self.route_and_start_by_text(utterance)


# -----------------------------
# CLI
# -----------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Podcast Router (software-only)")
    ap.add_argument("--root", type=str, default=".", help="project folder with transcript files (txt)")
    ap.add_argument("--say", type=str, default=None, help="free-form command like 'help me understand lecture27'")
    ap.add_argument("--interactive", action="store_true", help="prompt to type your command")
    ap.add_argument("--tts", choices=["on", "off"], default="off", help="text-to-speech via pyttsx3")
    ap.add_argument("--model", type=str, default="gpt-4", help="OpenAI model name")
    ap.add_argument("--turns", type=int, default=4, help="max conversation turns")
    args = ap.parse_args()

    router = PodcastRouter(root=args.root, model=args.model, tts=(args.tts == "on"), turns=args.turns)

    if args.say:
        router.route_and_start_by_text(args.say)
    elif args.interactive:
        print(router.catalog.list_human())
        cmd = input("\nWhat do you want to study? ")
        router.route_and_start_by_text(cmd)
    else:
        print("Nothing to do. Pass --say 'help me understand lecture27' or --interactive.")

if __name__ == "__main__":
    _cli()
