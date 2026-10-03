"""Offline voice loop: mic -> faster-whisper -> fuzzy match -> replay skill.

    python listen.py                      # voice + typed commands
    python listen.py --model small.en     # better accuracy, slower
    python listen.py --dry-run            # no robot, just prints what it would do

Say e.g. "give me the screwdriver", "hand me the tweezers", "clean up", "stop".
You can also just TYPE the command and press Enter (demo plan B).

Alias config: commands.json maps spoken phrases -> skill name (or a list of skills
to chain). Skill names themselves are always accepted.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz, process

import arm

SAMPLE_RATE = 16000
COMMANDS_FILE = Path(__file__).parent / "commands.json"
STOP_WORDS = {"stop", "halt", "freeze"}


# ---------- command routing ----------

def load_commands() -> dict[str, list[str]]:
    """phrase -> [skill, skill, ...]"""
    table: dict[str, list[str]] = {}
    if COMMANDS_FILE.exists():
        for phrase, target in json.loads(COMMANDS_FILE.read_text()).items():
            table[phrase.lower()] = target if isinstance(target, list) else [target]
    for s in arm.list_skills():
        table.setdefault(s.replace("_", " "), [s])
    return table


def route(text: str, table: dict[str, list[str]], threshold: int = 70) -> list[str] | None:
    text = text.lower().strip(" .,!?")
    if not text:
        return None
    if any(w in text.split() for w in STOP_WORDS):
        return ["__stop__"]
    # partial_ratio: "can you give me the screwdriver please" still matches "screwdriver"
    match = process.extractOne(text, list(table), scorer=fuzz.partial_ratio)
    if match and match[1] >= threshold:
        phrase, score, _ = match
        print(f"  ↳ matched '{phrase}' ({score:.0f})")
        return table[phrase]
    return None


# ---------- audio capture with crude energy VAD ----------

def utterances(energy_thresh: float = 0.01, silence_s: float = 0.7, max_s: float = 6.0):
    """Yields float32 numpy chunks, one per spoken utterance."""
    q: queue.Queue[np.ndarray] = queue.Queue()
    block = int(SAMPLE_RATE * 0.05)  # 50 ms

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block, callback=cb):
        buf, speaking, silent_for = [], False, 0.0
        while True:
            x = q.get()
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > energy_thresh:
                speaking, silent_for = True, 0.0
                buf.append(x)
            elif speaking:
                silent_for += 0.05
                buf.append(x)
                if silent_for >= silence_s or len(buf) * 0.05 >= max_s:
                    yield np.concatenate(buf)
                    buf, speaking, silent_for = [], False, 0.0


def meter(seconds: float = 8.0):
    """Print live mic RMS so --energy can be chosen from measurement, not guesswork."""
    block = int(SAMPLE_RATE * 0.05)
    levels: list[float] = []

    def cb(indata, frames, t, status):
        rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
        levels.append(rms)
        print(f"\r  rms={rms:.4f}  {'#' * min(60, int(rms * 1000))}".ljust(80), end="", flush=True)

    print(f"Mic meter for {seconds:.0f}s: stay quiet first, then say a command…")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block, callback=cb):
        time.sleep(seconds)
    quiet = float(np.percentile(levels, 20)); loud = float(np.percentile(levels, 95))
    print(f"\n  room noise ~{quiet:.4f}   speech peaks ~{loud:.4f}   suggested --energy {max(0.005, quiet * 3):.3f}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en", help="tiny.en | base.en | small.en")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--energy", type=float, default=0.01, help="VAD threshold; raise in a noisy room")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--meter", action="store_true", help="print live mic RMS for 8 s and exit")
    args = ap.parse_args()
    if args.meter:
        return meter()

    print(f"Loading whisper '{args.model}' (CPU int8)…")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    table = load_commands()
    print("Commands:", ", ".join(sorted(table)) or "(none — record some skills first)")

    robot = None if args.dry_run else arm.connect_follower()
    stop_flag = threading.Event()
    busy = threading.Lock()

    def run(skills: list[str]):
        if skills == ["__stop__"]:
            stop_flag.set()
            print("  ⏹ stop")
            return
        if not busy.acquire(blocking=False):
            print("  (busy — say 'stop' first)")
            return
        try:
            stop_flag.clear()
            for s in skills:
                print(f"  ▶ {s}")
                if robot is None:
                    time.sleep(1.0)
                    continue
                if not arm.play(robot, arm.load_skill(s), speed=args.speed, should_stop=stop_flag.is_set):
                    print("  interrupted")
                    break
            print("  ✓ done")
        finally:
            busy.release()

    def dispatch(text: str):
        print(f"heard: {text!r}")
        skills = route(text, table)
        if skills is None:
            print("  (no match)")
            return
        threading.Thread(target=run, args=(skills,), daemon=True).start()

    # keyboard fallback
    def keyboard():
        for line in sys.stdin:
            dispatch(line)
    threading.Thread(target=keyboard, daemon=True).start()

    print("Listening. Speak, or type a command + Enter. Ctrl-C to quit.")
    try:
        for audio in utterances(energy_thresh=args.energy):
            segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                           vad_filter=True, without_timestamps=True)
            text = " ".join(s.text for s in segments).strip()
            if text:
                dispatch(text)
    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
