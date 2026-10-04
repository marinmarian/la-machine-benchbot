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

Arms are identified by their controller board's USB serial, so ports never need editing:
`python arm.py` lists what is connected (role, calibration id, port). The serial -> calibration id
map is `arms.json`. A board it hasn't seen is matched once by reading the calibration stored in its
motors (read-only) and comparing with the calibration files, then saved there.
`FOLLOWER_ID` / `LEADER_ID` in `.env` pin which arm to use: a script refuses to run on a different
arm (skills only replay correctly on the arm they were recorded on). Unset, it uses whichever
follower/leader is plugged in. `FOLLOWER_PORT` / `LEADER_PORT` are only a fallback for a board
that can't be identified.
Calibration files live in `~/.cache/huggingface/lerobot/calibration/` and a copy is committed under
`benchbot/calibration/` (same layout; ids `follower_so101`, `leader_so101`, plus `*_left` for the second pair).
Restore them on a fresh machine with:

```bash
cp -R benchbot/calibration/. ~/.cache/huggingface/lerobot/calibration/
```

Recorded skills (`benchbot/skills/*.json`) are committed too; they only replay correctly on the arms they were recorded with.

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

## Tool detector (YOLO, trained on your own tools, offline)

Live camera window with a labelled box on every tool, and a spoken refusal when the tool a skill
needs is not on the bench ("Sorry, I don't see the screwdriver on the bench"). Zero-shot models
failed on these objects, so we train a small YOLO on auto-labelled frames. Budget ~1 h.

```bash
# camera fixed in its final spot, arm parked OUT of view, bench empty
python collect.py --background                 # saves vision/empty.jpg + empty frames
# park the arm IN view, bench still empty, run it again (teaches "arm is not a tool")
python collect.py --background
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

## Wrist camera dataset (for SAM alignment experiments)

`wristrec.py` records wrist-camera frames with the follower's measured joints next to each one.

```bash
python wristrec.py --probe                        # every camera tiled with its index; put the wrist one in .env as WRIST_CAMERA
python wristrec.py hover1                         # teleop with the leader: drive to a hover pose, h to hold,
                                                  # move the tool around under the camera with s (snapshot) / r (record), j once
python wristrec.py screw1 --skill screwdriver --hold-at-hover   # replay up to 2 s before the grasp, hold there, c continues
```

A skill can store its hold point as `"hover_frame": N` in its JSON (screwdriver4 has 141: whole tool in view,
room to slide it under the open jaws; black_motor2 has 219, its pause about 6 cm above where the jaws close). Then `--hold-at-hover` stops there instead of guessing 2 s before the grasp; `--hover-frame N` overrides both.

- `j` (only while holding) nudges pan, lift, elbow and wrist_flex ±3° and saves one settled frame per
  position, tagged `jog/<joint>/<±deg>`. Those frames give the pixels-per-degree table.
- Scope (decided 2026-10-03): the alignment corrects the tool's **position only** (left/right, closer/further).
  Tool **rotation is out of scope**: it would mean also turning `wrist_roll` and solving for orientation,
  which is much harder. The tool still has to lie at roughly the angle it had when the skill was recorded.
  `wrist_roll` is therefore not jogged, and snapshots only need position offsets.
- Output goes to `wrist_data/<session>/`: `frames/*.jpg`, `samples.jsonl` (`t`, `file`, `mode`, `tag`, `q` measured, `cmd`
  commanded), and `meta.json`. Running the same session name again appends. `wrist_data/` is not committed.

## Nudging a recorded skill (`adjust.py`)

Shift part of a skill in space without re-recording, e.g. grip 3 mm lower because the tool slipped out:

```bash
python adjust.py screwdriver4 --dz -0.3 --window 164 180 250 282 --dry-run   # report only
python adjust.py screwdriver4 --dz -0.3 --window 164 180 250 282
```

- `--dx/--dy/--dz` are cm in the robot base frame (z up). `--window A B C E`: the shift eases in over
  frames A..B, is full from B to C, eases out over C..E. Pick them from the joint values: for screwdriver4,
  164-180 is the last descent, 180-250 the grip, and 250-282 the start of the lift.
- Each frame goes through forward kinematics, gets the shift added, and goes back through inverse kinematics
  (placo, `urdf/so101_new_calib_kinematics.urdf`). `wrist_roll` and the gripper stay as recorded.
  The dry run prints the achieved shift, any tilt and the joint changes per frame.
- The original goes to `skills/backup/` (not committed) and the skill's `"adjustments"` list logs what was done.
- Needs `uv pip install placo`.

## Lining up on the tool with the wrist camera (`visual_servoing.py`)

The skill plays to its `hover_frame` and stops. SAM 3 finds the tool by text, SAM 2 tracks it, and the
gripper steps sideways at the same height until the tool sits where it sat in the recording. Then the
grasp plays shifted by that offset, which eases back to the recorded path during the lift.

```bash
python visual_servoing.py screwdriver4 --fit-table screw4_2   # done: pixels per mm at the hover, from wristrec's jog frames
python visual_servoing.py screwdriver4 --set-target           # tool exactly where the recording grasps it: save where it looks
python visual_servoing.py screwdriver4 --align-only --show    # move the tool, check it lines up (stops at the hover)
python visual_servoing.py screwdriver4 --show                 # line up + grasp
```

- Skill keys: `hover_frame`, `rejoin` [c, e] (full offset until frame c, back on the recording by e;
  screwdriver4: 250, 282; black_motor2: 338, 370, the start of the lift), `hover_px_per_mm`, `hover_target`, optional `prompt`.
- Reach: any direction up to 3 cm, 4 cm except straight outward. Beyond `--max-offset` (40 mm) it stops at the hover.
- The joints sag (a commanded 3° jog moves about 1°), so one big move is never trusted: each step is measured again.
  About 2.3 s for the first detection on the Mac GPU, then about 0.4 s per tracked frame.
- Moving sideways also turns the gripper a little, because pan is the only joint that can do it. The IK target
  (`Kinematics.target` in `adjust.py`) includes that turn.

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
Only `arm.py` touches lerobot, so that's the only file to adjust.
