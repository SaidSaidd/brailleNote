# podcast_mode_router.py
"""
Route-by-command to the right transcript, then (optionally) start a podcast-style session.
- Software-only (no GPIO).
- Auto-loads .env (OPENAI_API_KEY).
- Resolves: exact filename (e.g., transcript29.txt), "lecture 29", "transcript 29",
  a bare number "29", or "latest"/"most recent".
- Includes a text/TTY coach for keyboard interaction (kept for convenience).
- Exported helper: resolve_only(command_text) returns a path without launching a session.
"""

import os
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

# --- Soft deps ---
try:
    import openai  # works with openai.chat.completions.create
except Exception:
    openai = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


# -----------------------------
# Utilities
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
# Coach (text I/O; optional)
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
            try:
                student = input("\n👤 You: ").strip()
            except EOFError:
                break
            if not student:
                break
            self.history.append({"role": "student", "text": student})

            host, mentor = self.brain.respond(transcript_text, self.history, student)
            self.history.append({"role": "host", "text": host})
            self.history.append({"role": "mentor", "text": mentor})
            self.speaker.duet(host, mentor)

        self.speaker.say("That's a wrap. Want to continue later, just run this again.")


# -----------------------------
# File catalog + routing
# -----------------------------

@dataclass
class FileCatalog:
    root: str = "."
    by_filename: Dict[str, str] = field(default_factory=dict)
    by_number: Dict[int, List[Tuple[int, str, str]]] = field(default_factory=dict)  # n -> list of (priority, path, label)

    def scan(self) -> None:
        self.by_filename.clear()
        self.by_number.clear()

        try:
            entries = os.listdir(self.root)
        except FileNotFoundError:
            entries = []

        for fname in entries:
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(self.root, fname)
            base = os.path.splitext(fname)[0].lower()
            self.by_filename[fname.lower()] = path
            self.by_filename[base + ".txt"] = path  # normalized

            label, n = self._extract_label_and_number(base)
            if n is not None:
                prio = {"transcript": 0, "lecture": 1, "notes": 2, "generic": 3}.get(label, 3)
                self.by_number.setdefault(n, []).append((prio, path, label))

    @staticmethod
    def _extract_label_and_number(base: str) -> Tuple[str, Optional[int]]:
        # Strong pattern: lecture/transcript/notes + number
        m = re.search(r"\b(lecture|transcript|notes?)\s*[-_ ]?\s*(\d+)\b", base)
        if m:
            return (m.group(1).rstrip('s'), int(m.group(2)))
        # Fallback: any trailing number token
        m2 = re.search(r"(\d+)\b", base)
        if m2:
            return ("generic", int(m2.group(1)))
        return ("generic", None)

    def latest_path(self) -> Optional[str]:
        candidates = [os.path.join(self.root, f) for f in os.listdir(self.root) if f.lower().endswith(".txt")]
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    def list_human(self) -> str:
        try:
            items = [f for f in os.listdir(self.root) if f.lower().endswith(".txt")]
        except FileNotFoundError:
            items = []
        items.sort()
        lines = ["Found transcripts:"]
        for p in items:
            lines.append(f" - {p}")
        return "\n".join(lines)

    def resolve(self, utterance: str) -> Optional[str]:
        """
        Map a free-form utterance to a transcript file path, with strict rules:
          1) 'latest'/'most recent' -> newest by mtime
          2) explicit filename *.txt -> exact match
          3) '(lecture|transcript|notes) N' -> number match with label preference
          4) bare number 'N' -> number match (prioritize transcript, then lecture, then notes, then generic)
          otherwise: None
        """
        if not utterance:
            return None
        u = utterance.lower().strip()

        # 1) latest
        if re.search(r"\b(latest|most\s+recent)\b", u):
            return self.latest_path()

        # 2) explicit filename
        m = re.search(r"([a-z0-9_\-]+\.txt)\b", u)
        if m:
            key = m.group(1).lower()
            if key in self.by_filename:
                return self.by_filename[key]

        # 3) label + number
        m = re.search(r"\b(lecture|transcript|notes?)\s*[-_ ]?\s*(\d+)\b", u)
        if m:
            label = m.group(1).rstrip('s')
            n = int(m.group(2))
            candidates = self.by_number.get(n, [])
            if not candidates:
                return None
            # Prefer same label; else priority order
            same_label = [p for pr, p, lbl in candidates if lbl == label]
            if same_label:
                return same_label[0]
            best = sorted(candidates, key=lambda t: t[0])[0]
            return best[1]

        # 4) bare number
        m = re.search(r"\b(\d{1,5})\b", u)
        if m:
            n = int(m.group(1))
            candidates = self.by_number.get(n, [])
            if candidates:
                best = sorted(candidates, key=lambda t: t[0])[0]
                return best[1]

        return None


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

    # NEW: just resolve, don't start a session
    def resolve_only(self, command_text: str) -> Optional[str]:
        self.catalog.scan()
        return self.catalog.resolve(command_text)

    def route_and_start_by_text(self, command_text: str) -> Optional[str]:
        self.catalog.scan()
        path = self.catalog.resolve(command_text)
        if not path:
            print(self.catalog.list_human())
            print("Couldn't resolve that request. Try 'lecture 29', 'transcript29.txt', or 'latest'.")
            return None
        with open(path, "r", encoding="utf-8") as f:
            transcript = f.read()
        print(f"\n📄 Using transcript: {os.path.basename(path)}\n")
        self.coach.start_session(transcript, max_turns=self.turns)
        return path


# -----------------------------
# CLI
# -----------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Podcast Router (software-only)")
    ap.add_argument("--root", type=str, default=".", help="project folder with transcript files (txt)")
    ap.add_argument("--say", type=str, default=None, help="free-form command like 'help me understand lecture 29'")
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
        print("Nothing to do. Pass --say 'help me understand lecture 29' or --interactive.")

if __name__ == "__main__":
    _cli()
