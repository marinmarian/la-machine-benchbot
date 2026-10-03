# benchbot — voice-triggered SO-101 bench helper

Record a motion once with the leader arm, replay it by name with your voice. Fully offline.

## Setup (done 2026-10-03 on Marin's Mac — kept here for reference)

lerobot 0.5.1 is installed editable from `~/Work/lerobot` in the `lerobot` conda env
(Python 3.12, torch 2.10). The speech deps were added into that same env with uv so
nothing heavy had to be re-downloaded:

```bash
conda activate lerobot
uv pip install --python "$(which python)" faster-whisper sounddevice rapidfuzz
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"   # pre-download, verified with HF_HUB_OFFLINE=1
```

Ports and calibration ids live in `.env` (loaded by `arm.py`; shell env vars override).
After plugging both arms in, confirm with `ls /dev/tty.usbmodem*`. If the names changed, edit `.env`.
Calibration files: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/follower_so101.json`
and `.../teleoperators/so_leader/leader_so101.json`.

Run scripts with the env active: `python record.py home`, `python replay.py home`, `python listen.py`.
(`uv run` is not used: the pyproject's `lerobot[feetech]` line would pull a second lerobot + torch from PyPI.)

Note: `connect()` compares the motors' stored calibration with the file and, on mismatch, drops into
the interactive `lerobot-calibrate` prompt. If a script seems to hang right after connecting, look for that prompt.

macOS: give the terminal microphone permission the first time `listen.py` runs.

## Workflow

1. **Record** each skill. Start AND end in the same parked "home" pose.
   ```bash
   python record.py home          # a tiny clip that just sits in the parked pose
   python record.py screwdriver   # home -> grab screwdriver from slot -> hold out -> release -> home
   python record.py return_tool
   python record.py cleanup
   ```
2. **Test** each one: `python replay.py screwdriver` (use `--speed 0.7` if it's jerky; speed is capped at 1.0).
3. **Edit** `commands.json` for the phrases you'll actually say.
4. **Run** the demo: `python listen.py` (first check `python listen.py --dry-run`).

## Camera slot check (optional, offline)

Before a skill runs, `listen.py` can look at the bench and refuse out loud when the tool is not there
("Sorry, there is nothing in the screwdriver slot"). No ML model: two reference photos per setup.

```bash
python vision.py setup            # camera fixed, tools IN their slots; draw a tight box per slot, name it like the skill
                                  # (screwdriver, microphone, …). This frame becomes the "full" reference.
python vision.py capture empty    # every slot empty, camera untouched
python vision.py check --show     # PRESENT/absent + margin per slot, boxes drawn on the frame
python vision.py watch            # live window, every slot boxed + labelled, green = present, red = absent
python listen.py --show           # same live window during the demo
```

- A skill named like a slot requires that slot to be present. Other rules go in `slots.json` under
  `requires`, e.g. `"return_tool": {"screwdriver": "empty"}`.
- Checks run before every skill in a chain. `--no-vision` disables them. If `slots.json` or the photos
  are missing, listen.py says `vision: off` and runs as before.
- Re-capture `full` and `empty` in the demo room under demo lighting, right before the demo (20 s).
- Boxes must not be covered by the arm in its home pose. Margins under 10 mean the box is ambiguous: make it tighter around where the tool actually sits, or redo setup.
- Real test: take one tool out and run `check --show`. Only that slot should turn red.
- The robot talks via macOS `say`; the mic is muted while it speaks so it cannot trigger itself.

## Demo-day tips

- Tape the tool slots to the bench. Replay is a player piano: it plays the same roll every time, so the tools must be where they were when you recorded.
- Lower the follower's speed/accel if grasps slip; smooth beats fast on video.
- Re-capture the two vision reference photos in the final room/lighting.
- Keep a hand on the power switch. `stop` (voice or typed) interrupts between frames.
- Noisy room: run with `--energy 0.02` or `0.03`, and hold the laptop mic close. Typing the command is an acceptable plan B — the judges care about the arm, not the mic.
- `--model small.en` if `base.en` mishears tool names; add misheard variants to `commands.json` instead of fighting the model (e.g. `"screw driver"`, `"screwdrivers"`).

## If the lerobot import path differs (already fixed for 0.5.x: `lerobot.robots.so_follower`, `lerobot.teleoperators.so_leader`)

Your installed version may use different module paths. Check with:
```bash
python -c "import lerobot, pkgutil; print(lerobot.__version__); print([m.name for m in pkgutil.iter_modules(lerobot.robots.__path__)])"
```
Only `arm.py` touches lerobot, so that's the only file to adjust.
