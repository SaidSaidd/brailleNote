import os

API_KEY = "API-KEY"

import assemblyai as aai
aai.settings.api_key = API_KEY

import glob
import re              # for normalization
import wave
import logging
from typing import Type

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
import pyaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1) PyAudio generator to capture raw audio chunks
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2) Globals to collect transcript and dedupe repeated lines
# -----------------------------------------------------------------------------
transcript_lines = []

# -----------------------------------------------------------------------------
# 3) Event handlers
# -----------------------------------------------------------------------------
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    # capture *only* the formatted, finalized turn
    if event.end_of_turn and event.turn_is_formatted:
        transcript_lines.append(event.transcript.strip())

    # still print everything for debug
    conf = getattr(event, "end_of_turn_confidence", None)
    print(
        f"Turn: '{event.transcript}'"
        + (f" (Confidence: {conf:.2f})" if conf is not None else "")
    )

    # tell the service to format future turns
    if event.end_of_turn and not event.turn_is_formatted:
        self.set_params(StreamingSessionParameters(format_turns=True))

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"Session terminated: {event.audio_duration_seconds:.2f}s of audio")

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Error: {error}")

# -----------------------------------------------------------------------------
# 4) Helpers & main loop
# -----------------------------------------------------------------------------
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
        idx = get_next_index()
        wav_filename = f"session{idx}.wav"
        txt_filename = f"transcript{idx}.txt"

        # 1) write out the raw audio
        mic.close()
        with wave.open(wav_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(mic.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(mic.rate)
            wf.writeframes(b"".join(mic.frames))

        # 2) write out the transcript
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))

        # --- 3) now that session{idx}.wav exists, generate AI summary ---
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

        # 4) finally disconnect
        client.disconnect(terminate=True)
        print(f"Done — saved {wav_filename}, {txt_filename}", end="")
        if os.path.exists(summary_filename):
            print(f", and {summary_filename}")
        else:
            print("")


if __name__ == "__main__":
    main()
