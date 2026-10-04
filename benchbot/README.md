# benchbot — voice-triggered SO-101 bench helper

Record a motion once with the leader arm(s), replay it by name with your voice. Fully offline.
One arm or two: a skill file holds every arm that was recording, keyed `right_…`/`left_…`.

## Setup (done 2026-10-03 on Marin's Mac — kept here for reference)

lerobot 0.5.1 is installed editable from `~/Work/lerobot` in the `lerobot` conda env
(Python 3.12, torch 2.10). The speech deps were added into that same env with uv so
nothing heavy had to be re-downloaded:

```bash
conda activate lerobot
uv pip install --python "$(which python)" faster-whisper sounddevice rapidfuzz
python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"   # pre-download, verified with HF_HUB_OFFLINE=1
```

Ports and calibration ids live in `.env` (loaded by `arm.py`; shell env vars override), one block per arm
(`ARMS=right,left`, then `RIGHT_FOLLOWER_PORT`, `RIGHT_LEADER_PORT`, `RIGHT_FOLLOWER_ID`, … and the same for `LEFT_`).
After plugging the arms in, run `python ports.py`: it pings every `/dev/tty.usbmodem*` and says which looks like a
leader and which a follower. If the names changed, edit `.env`.
Calibration files live in `~/.cache/huggingface/lerobot/calibration/` and a copy is committed under
`benchbot/calibration/` (same layout; ids `follower_so101`, `leader_so101`, plus `*_left` for the second pair).
Restore them on a fresh machine with:

```bash
cp -R benchbot/calibration/. ~/.cache/huggingface/lerobot/calibration/
```

Recorded skills (`benchbot/skills/*.json`) are committed too; they only replay correctly on the arms they were recorded with.

Run scripts with the env active: `python record.py home`, `python replay.py home`, `python listen.py`.
Every script takes `--arms right` (or `left`, or `right,left`) to drive a subset; the default is all arms in `ARMS`.
`listen.py --arms right` is the one to use while the second pair is not calibrated yet, otherwise startup fails
on the missing calibration file.
(`uv run` is not used: the pyproject's `lerobot[feetech]` line would pull a second lerobot + torch from PyPI.)

Note: `connect()` compares the motors' stored calibration with the file and, on mismatch, drops into
the interactive `lerobot-calibrate` prompt. If a script seems to hang right after connecting, look for that prompt.

macOS: give the terminal microphone permission the first time `listen.py` runs.

## Second arm (two hands working together)

The second pair plugs in as two more `/dev/tty.usbmodem*` ports. Set it up once:

1. Power it and run `python ports.py`. A silent port means motor power is off or the 3-pin bus cable to the
   first motor is loose; the USB board shows up without either. If only id 1 answers, the motors are a fresh kit:
   run `lerobot-setup-motors` for that arm first.
2. Decide which port is the leader (the one with the handle; torque off) and put the two ports in `.env`
   under `LEFT_FOLLOWER_PORT` / `LEFT_LEADER_PORT`. Confirm by unplugging one USB cable and re-running `ports.py`.
3. Calibrate both with the ids from `.env` (interactive: move each joint through its full range):
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=$LEFT_FOLLOWER_PORT --robot.id=follower_so101_left
   lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$LEFT_LEADER_PORT  --teleop.id=leader_so101_left
   ```
4. Check it mirrors: `python record.py scratch --arms left`, wave the leader, Enter twice, then `python replay.py scratch`.

Then record two-arm skills exactly like one-arm ones; `record.py` reads both leaders every tick:
```bash
python record.py handoff            # both arms (default) — e.g. left holds the board, right drives the screw
python record.py tweezers --arms left
python replay.py --list             # shows which arms each skill uses
```
Rules that now apply to both arms at once: start and end every skill with BOTH arms in their home pose, and
when a one-arm skill plays, the other arm simply keeps holding wherever it was left. The old one-arm skills
(`clean`, `screwdriver`, `microphone`, `test`) have unprefixed keys and are mapped to the first arm in `ARMS`,
so keep the original pair listed first.

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

## Tool detector (YOLO, trained on your own tools, offline)

Live camera window with a labelled box on every tool, and a spoken refusal when the tool a skill
needs is not on the bench ("Sorry, I don't see the screwdriver on the bench"). Zero-shot models
failed on these objects, so we train a small YOLO on auto-labelled frames. Budget ~1 h.

```bash
# camera fixed in its final spot, arms parked OUT of view, bench empty
python collect.py --background                 # (re)writes vision/empty.jpg + empty frames
# park the arms IN view, bench still empty (teaches "arms are not tools")
python collect.py --background --keep-empty
# one object at a time, fully inside the frame; move + rotate it around the bench for 45 s
python collect.py screwdriver
python collect.py microphone                   # class name = skill name
python collect.py motor  ...
python train.py                                # ~40 epochs on the Mac GPU -> weights/benchbot.pt
python vision.py watch                         # boxes now come from the detector
```

- Labels come from background subtraction against `vision/empty.jpg`: frames with no object, two
  objects, or an object touching the frame edge are skipped (the window says so).
- Weak class? 30 more seconds of `collect.py <class>` then `train.py` again beats more epochs.
- `weights/benchbot.pt` (~6 MB) is committed so a fresh clone demos. `dataset/` and `runs/` are not.
- Re-check `vision.py watch` in the demo room; retrain only if a class drops out.

## Camera slot check (fallback, no model)

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
Only `arm.py` and `ports.py` touch lerobot, so those are the only files to adjust.
