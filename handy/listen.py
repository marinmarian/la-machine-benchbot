"""Offline voice loop: mic -> faster-whisper -> fuzzy match -> replay skill.

    python listen.py                      # voice + typed commands
    python listen.py --model small.en     # better accuracy, slower
    python listen.py --dry-run            # no robot, just prints what it would do
    python listen.py --arms right         # drive only one arm (two-arm skills play their right half)

Say e.g. "give me the screwdriver", "hand me the tweezers", "clean up", "stop".
You can also just TYPE the command and press Enter (demo plan B).

Alias config: commands.json maps spoken phrases -> skill name (or a list of skills
to chain). Skill names themselves are always accepted.
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz, process

import arm
import vision

SAMPLE_RATE = 16000
COMMANDS_FILE = Path(__file__).parent / "commands.json"
STOP_WORDS = {"stop", "halt", "freeze"}
SAY = shutil.which("say")          # macOS offline TTS; None elsewhere -> print only
speaking = threading.Event()       # set while we talk so the mic ignores our own voice


def speak(msg: str) -> None:
    print(f"  🗣 {msg}")
    if not SAY:
        return
    speaking.set()
    try:
        subprocess.run([SAY, msg], check=False)
        time.sleep(0.4)            # let the room tail die before listening again
    finally:
        speaking.clear()


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
        buf, in_speech, silent_for = [], False, 0.0
        while True:
            x = q.get()
            if speaking.is_set():                 # that is us talking, not the user
                buf, in_speech, silent_for = [], False, 0.0
                continue
            rms = float(np.sqrt(np.mean(x ** 2)))
            if rms > energy_thresh:
                in_speech, silent_for = True, 0.0
                buf.append(x)
            elif in_speech:
                silent_for += 0.05
                buf.append(x)
                if silent_for >= silence_s or len(buf) * 0.05 >= max_s:
                    yield np.concatenate(buf)
                    buf, in_speech, silent_for = [], False, 0.0


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
    ap.add_argument("--no-vision", action="store_true", help="skip the camera slot check before skills")
    ap.add_argument("--show", action="store_true", help="keep a live camera window with slot boxes open")
    ap.add_argument("--arms", help=f"comma list of arms to drive, default all: {','.join(arm.ARMS)}")
    args = ap.parse_args()
    if args.meter:
        return meter()

    print(f"Loading whisper '{args.model}' (CPU int8)…")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    table = load_commands()
    print("Commands:", ", ".join(sorted(table)) or "(none — record some skills first)")

    use_vision = vision.available() and not args.no_vision
    if use_vision:
        print(f"vision: on, backend {vision.backend()}, checks {vision.names()} before skills")
    else:
        print("vision: off" + ("" if args.no_vision else "  (collect.py + train.py for a detector, or vision.py setup for slots)"))

    cam = vision.LiveCamera(vision.load_config() or {"camera_index": 0}) if (use_vision and args.show) else None

    arms = arm.parse_arms(args.arms)
    rig = None if args.dry_run else arm.connect_followers(arms)   # {arm_name: follower}
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
                if not arm.skill_path(s).exists():
                    print(f"  skill '{s}' not recorded yet (python record.py {s})")
                    break
                if use_vision:
                    ok, why = vision.requirement_ok(s, frame=cam.latest() if cam else None)
                    if not ok:
                        speak(f"Sorry, {why}.")
                        break
                skill = arm.load_skill(s)
                print(f"  ▶ {s} ({', '.join(arm.skill_arms(skill))})")
                if rig is None:
                    time.sleep(1.0)
                    continue
                if not arm.play(rig, skill, speed=args.speed, should_stop=stop_flag.is_set):
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

    def listen_loop():
        for audio in utterances(energy_thresh=args.energy):
            segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                           vad_filter=True, without_timestamps=True)
            text = " ".join(s.text for s in segments).strip()
            if text:
                dispatch(text)

    print("Listening. Speak, or type a command + Enter. Ctrl-C to quit." + ("  (q in the camera window also quits)" if cam else ""))
    try:
        if cam:
            threading.Thread(target=listen_loop, daemon=True).start()
            vision.watch(cam, vision.load_config() or {})  # window must run on the main thread (macOS)
        else:
            listen_loop()
    except KeyboardInterrupt:
        pass
    finally:
        if cam:
            cam.close()
        if rig is not None:
            arm.disconnect(rig)


if __name__ == "__main__":
    main()
