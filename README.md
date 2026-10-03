# benchbot — voice-triggered SO-101 bench helper

Record a motion once with the leader arm, replay it by name with your voice. Fully offline.

## Setup (do this while you still have mobile data — whisper model downloads once)

```bash
cd benchbot
uv venv && source .venv/bin/activate
uv pip install -e .            # or: uv pip install -e "/path/to/lerobot[feetech]" then uv pip install faster-whisper sounddevice rapidfuzz numpy
lerobot-find-port              # run twice, once per arm
export FOLLOWER_PORT=/dev/tty.usbmodemXXXX LEADER_PORT=/dev/tty.usbmodemYYYY
export FOLLOWER_ID=<id you used in lerobot-calibrate> LEADER_ID=<same for leader>
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"   # pre-download
```

macOS: give the terminal microphone permission the first time `listen.py` runs.

## Workflow

1. **Record** each skill. Start AND end in the same parked "home" pose.
   ```bash
   uv run record.py home          # a tiny clip that just sits in the parked pose
   uv run record.py screwdriver   # home -> grab screwdriver from slot -> hold out -> release -> home
   uv run record.py return_tool
   uv run record.py cleanup
   ```
2. **Test** each one: `uv run replay.py screwdriver` (use `--speed 0.7` if it's jerky).
3. **Edit** `commands.json` for the phrases you'll actually say.
4. **Run** the demo: `uv run listen.py` (first check `uv run listen.py --dry-run`).

## Demo-day tips

- Tape the tool slots to the bench. Replay is a player piano: it plays the same roll every time, so the tools must be where they were when you recorded.
- Lower the follower's speed/accel if grasps slip; smooth beats fast on video.
- Keep a hand on the power switch. `stop` (voice or typed) interrupts between frames.
- Noisy room: run with `--energy 0.02` or `0.03`, and hold the laptop mic close. Typing the command is an acceptable plan B — the judges care about the arm, not the mic.
- `--model small.en` if `base.en` mishears tool names; add misheard variants to `commands.json` instead of fighting the model (e.g. `"screw driver"`, `"screwdrivers"`).

## If the lerobot import path differs

Your installed version may use different module paths. Check with:
```bash
python -c "import lerobot, pkgutil; print(lerobot.__version__); print([m.name for m in pkgutil.iter_modules(lerobot.robots.__path__)])"
```
Only `arm.py` touches lerobot, so that's the only file to adjust.
