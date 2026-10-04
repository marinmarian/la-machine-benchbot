"""Offline voice loop: mic -> faster-whisper -> fuzzy match -> replay skill.

    python listen.py                      # Enter, speak, Enter: only what is said in between counts; or type a command
    python listen.py --model small.en     # better accuracy, slower
    python listen.py --dry-run            # no robot, just prints what it would do
    python listen.py --arms right         # drive only one arm (two-arm skills play their right half)
    python listen.py --ptt                # push-to-talk: only hears you while right Option is held
    python listen.py --ptt page_down      # ... or another key (a presentation clicker sends page_down/page_up)
    python listen.py --open-mic           # always listening (cuts at pauses); fine in a quiet room, not while presenting

Skills set up for the wrist camera (hover_target etc., see visual_servoing.py; e.g. black_motor2) are
always played lining up on the tool with SAM 3 + SAM 2, never blind. Those models load at startup
(~20 s); without them (--no-wrist, or no wrist camera) such a skill is refused out loud.

Say e.g. "give me the screwdriver", "hand me the tweezers", "clean up", "stop".
You can also just TYPE the command and press Enter (demo plan B).

Alias config: commands.json maps spoken phrases -> skill name (or a list of skills
to chain). Skill names themselves are always accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
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
PHRASES_FILE = Path(__file__).parent / "phrases.json"
STOP_WORDS = {"stop", "halt", "freeze"}
SAY = shutil.which("say")          # macOS offline TTS; None elsewhere -> print only
TTS_VOICE = os.environ.get("TTS_VOICE", "")   # e.g. Ava or Zoe after downloading a Premium voice; "" = system default
speaking = threading.Event()       # set while we talk so the mic ignores our own voice
_speak_lock = threading.Lock()     # one utterance at a time
PHRASES: dict = json.loads(PHRASES_FILE.read_text()) if PHRASES_FILE.exists() else {}


def phrase(kind: str, skill: str | None = None, **fmt) -> str | None:
    """Pick a random line from phrases.json: PHRASES[kind][skill] -> [kind]['_default'] -> [kind]."""
    table = PHRASES.get(kind)
    if isinstance(table, dict):
        table = table.get(skill) or table.get("_default")
    if not table:
        return None
    return random.choice(table).format(skill=(skill or "").replace("_", " "), **fmt)


def speak(msg: str | None, wait: bool = False) -> None:
    """Say `msg` (non-blocking by default so the arm starts moving while we talk).
    The mic is muted for the whole utterance plus a short tail so we never hear ourselves."""
    if not msg:
        return
    print(f"  🗣 {msg}")
    if not SAY:
        return

    def _say():
        with _speak_lock:
            speaking.set()
            try:
                subprocess.run([SAY] + (["-v", TTS_VOICE] if TTS_VOICE else []) + [msg], check=False)
                time.sleep(0.4)            # let the room tail die before listening again
            finally:
                speaking.clear()

    t = threading.Thread(target=_say, daemon=True)
    t.start()
    if wait:
        t.join()


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


def gated_utterances(gate: threading.Event, min_s: float = 0.3, max_s: float = 20.0):
    """One float32 chunk per opening of `gate`: everything the mic hears while it is set, nothing else
    (push-to-talk and tap-to-talk). Over max_s the gate closes by itself."""
    q: queue.Queue[np.ndarray] = queue.Queue()
    block = int(SAMPLE_RATE * 0.05)  # 50 ms

    def cb(indata, frames, t, status):
        if gate.is_set():
            q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block, callback=cb):
        buf: list[np.ndarray] = []
        while True:
            try:
                buf.append(q.get(timeout=0.1))
            except queue.Empty:
                pass
            if gate.is_set() and len(buf) * 0.05 >= max_s:
                print(f"  ({max_s:.0f} s, mic off)")
                gate.clear()
            if not gate.is_set() and q.empty() and buf:     # closed: one utterance
                if len(buf) * 0.05 >= min_s:
                    yield np.concatenate(buf)
                buf = []


def ptt_utterances(key_name: str):
    """Push-to-talk: one chunk per press of `key_name` (a pynput Key name like alt_r, or a single
    character). The key works from any window, which needs macOS Accessibility permission."""
    from pynput import keyboard
    key = getattr(keyboard.Key, key_name, None) or keyboard.KeyCode.from_char(key_name)
    held = threading.Event()

    def on_press(k):
        if k == key and not held.is_set():
            held.set()
            print("  🎙 listening…", flush=True)

    def on_release(k):
        if k == key:
            held.clear()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.wait()
    if not getattr(listener, "IS_TRUSTED", True):
        print("  ⚠ macOS is not passing key presses to this terminal: System Settings > Privacy & Security > "
              "Accessibility (and Input Monitoring) > enable your terminal app, then restart listen.py")
    yield from gated_utterances(held)


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


# ---------- wrist camera line-up (visual_servoing) ----------

def vs_ready(skill: dict) -> bool:
    """Has everything visual_servoing's line-up needs; such a skill is never replayed blind."""
    return all(k in skill for k in ("hover_frame", "hover_target", "hover_px_per_mm", "rejoin"))


def load_wrist(servo_skills: set[str]) -> dict | None:
    """Open the wrist camera and load + warm up SAM 3 / SAM 2 now, so a command never waits for them."""
    index = os.environ.get("WRIST_CAMERA")
    if index is None:
        print("wrist camera: WRIST_CAMERA not set in .env")
        return None
    try:
        import visual_servoing as vs
        from wristrec import Camera
        cam = Camera(int(index), 640, 480)
        first = sorted(servo_skills)[0]
        print(f"Loading SAM 3 + SAM 2 for the wrist camera (camera {index})…")
        aligner = vs.Aligner(cam, vs.prompt_for(first, arm.load_skill(first)), show=False)
        return {"vs": vs, "cam": cam, "aligner": aligner, "kin": vs.Kinematics()}
    except Exception as e:  # noqa: BLE001  (camera missing, model download, ...) -> refuse those skills later
        print(f"wrist camera: failed to start ({type(e).__name__}: {e})")
        return None


def servo_run(wrist: dict, rig, name: str, skill: dict, should_stop) -> bool:
    """Play a wrist-camera skill: hover, line up on the tool, offset grasp. False if it didn't finish."""
    vs, a = wrist["vs"], arm.DEFAULT_ARM
    solo = arm.solo(skill, a)
    wrist["aligner"].prompt = vs.prompt_for(name, solo)
    try:
        vs.servo_grasp(rig, a, solo, wrist["aligner"], wrist["kin"], should_stop=should_stop)
        return True
    except vs.Stopped:
        print("  interrupted")
    except vs.ServoFailed as e:
        print(f"  gave up lining up: {e}")
        speak(phrase("servo_failed", name))
    return False


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en", help="tiny.en | base.en | small.en")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--energy", type=float, default=0.01, help="VAD threshold; raise in a noisy room")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--meter", action="store_true", help="print live mic RMS for 8 s and exit")
    ap.add_argument("--no-bench-check", "--no-vision", dest="no_vision", action="store_true",
                    help="skip the bench camera's 'is the tool there?' check before skills (not the wrist camera)")
    ap.add_argument("--show", action="store_true", help="keep a live camera window with slot boxes open")
    ap.add_argument("--arms", help=f"comma list of arms to drive, default all: {','.join(arm.ARMS)}")
    ap.add_argument("--ptt", nargs="?", const="alt_r", metavar="KEY",
                    help="push-to-talk: only listen while KEY is held (default right Option; pynput name, e.g. "
                         "page_down, f13, or a letter)")
    ap.add_argument("--open-mic", action="store_true",
                    help="always listen and cut at pauses, instead of the default: Enter (empty line) starts "
                         "listening, Enter again stops and runs what was said, the mic is ignored otherwise")
    ap.add_argument("--no-wrist", action="store_true",
                    help="don't load the wrist camera + SAM models; skills that need them are refused")
    args = ap.parse_args()
    if args.meter:
        return meter()

    print(f"Loading whisper '{args.model}' (CPU int8)…")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    table = load_commands()
    print("Commands:", ", ".join(sorted(table)) or "(none — record some skills first)")

    use_vision = vision.available() and not args.no_vision
    if use_vision:
        print(f"bench camera check: on, backend {vision.backend()}, checks {vision.names()} before skills")
    else:
        print("bench camera check: off" + ("" if args.no_vision else "  (collect.py + train.py for a detector, or vision.py setup for slots)"))

    cam = vision.LiveCamera(vision.load_config() or {"camera_index": 0}) if (use_vision and args.show) else None

    arms = arm.parse_arms(args.arms)
    # skills that line up on the tool with the wrist camera (on the first arm, which carries it)
    servo_skills = {s for s in arm.list_skills() if arm.DEFAULT_ARM in arm.skill_arms(arm.load_skill(s))
                    and vs_ready(arm.load_skill(s))}
    wrist = load_wrist(servo_skills) if servo_skills and not (args.dry_run or args.no_wrist) else None
    if servo_skills:
        print(f"wrist camera: {'ready' if wrist else 'off'} for {', '.join(sorted(servo_skills))}")

    rig = None if args.dry_run else arm.connect_followers(arms)   # {arm_name: follower}
    stop_flag = threading.Event()
    busy = threading.Lock()

    def run(skills: list[str]):
        if skills == ["__stop__"]:
            stop_flag.set()
            print("  ⏹ stop")
            speak(phrase("stop"))
            return
        if not busy.acquire(blocking=False):
            print("  (busy — say 'stop' first)")
            speak(phrase("busy"))
            return
        try:
            stop_flag.clear()
            speak(phrase("start", skills[0]))          # acknowledge right away, before any motion
            for s in skills:
                if not arm.skill_path(s).exists():
                    print(f"  skill '{s}' not recorded yet (python record.py {s})")
                    speak(phrase("missing", s))
                    break
                if use_vision:
                    ok, why = vision.requirement_ok(s, frame=cam.latest() if cam else None)
                    if not ok:
                        speak(f"Sorry, {why}.")
                        break
                skill = arm.load_skill(s)
                print(f"  ▶ {s} ({', '.join(arm.skill_arms(skill))})" + ("  [wrist camera]" if s in servo_skills else ""))
                if rig is None:
                    time.sleep(1.0)
                    speak(phrase("done", s))
                    continue
                if s in servo_skills:                    # only ever with the camera, never blind
                    if wrist is None or arm.DEFAULT_ARM not in rig:
                        print("  wrist camera not ready (see startup), refusing")
                        speak(phrase("no_wrist", s))
                        break
                    if not servo_run(wrist, rig, s, skill, stop_flag.is_set):
                        break
                    speak(phrase("done", s))
                    continue
                if not arm.play(rig, skill, speed=args.speed, should_stop=stop_flag.is_set):
                    print("  interrupted")
                    break
                speak(phrase("done", s))
            print("  ✓ done")
        finally:
            busy.release()

    def dispatch(text: str):
        print(f"heard: {text!r}")
        skills = route(text, table)
        if skills is None:
            print("  (no match)")
            if len(text.split()) >= 2:
                speak(phrase("unknown"))
            return
        threading.Thread(target=run, args=(skills,), daemon=True).start()

    # keyboard fallback
    armed = None if (args.open_mic or args.ptt) else threading.Event()   # tap-to-talk gate

    def keyboard():
        for line in sys.stdin:
            if armed is not None and not line.strip():
                if armed.is_set():
                    armed.clear()
                    print("  mic off", flush=True)
                else:
                    armed.set()
                    print("  🎙 listening… Enter to stop", flush=True)
            else:
                dispatch(line)
    threading.Thread(target=keyboard, daemon=True).start()

    def listen_loop():
        source = (ptt_utterances(args.ptt) if args.ptt else gated_utterances(armed) if armed
                  else utterances(energy_thresh=args.energy))
        for audio in source:
            segments, _ = model.transcribe(audio, language="en", beam_size=1,
                                           vad_filter=True, without_timestamps=True)
            text = " ".join(s.text for s in segments).strip()
            if text:
                dispatch(text)

    print((f"Hold {args.ptt} and speak" if args.ptt else "Press Enter, speak, press Enter" if armed else "Listening. Speak") + ", or type a command + Enter. Ctrl-C to quit." + ("  (q in the camera window also quits)" if cam else ""))
    speak(phrase("hello"))
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
        if wrist:
            wrist["cam"].close()
        if rig is not None:
            arm.disconnect(rig)


if __name__ == "__main__":
    main()
