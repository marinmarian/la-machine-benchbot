# Handy — voice-triggered SO-101 bench helper

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

Arms are named in `.env` (`ARMS=right,left`) and each name gets its calibration ids
(`RIGHT_FOLLOWER_ID`, `RIGHT_LEADER_ID`, and the same for `LEFT_`). Ports never need editing: each controller
board is recognised by its USB serial (`arms.json`, serial -> calibration id). `python arm.py` lists what is
connected (arm, role, calibration id, port). A board it hasn't seen is matched once by reading the calibration
stored in its motors (read-only) and comparing with the calibration files, then saved there. An id set in
`.env` must be the arm that is plugged in: a script refuses to drive a different one (skills only replay
correctly on the arm they were recorded on). `{ARM}_FOLLOWER_PORT` / `{ARM}_LEADER_PORT` are only a fallback
for a board that can't be identified. `python ports.py` pings every `/dev/tty.usbmodem*` and guesses leader vs
follower; useful for a board whose motors match no calibration yet.
Calibration files live in `~/.cache/huggingface/lerobot/calibration/` and a copy is committed under
`handy/calibration/` (same layout; ids `follower_so101`, `leader_so101`, plus `*_left` for the second pair).
Restore them on a fresh machine with:

```bash
cp -R handy/calibration/. ~/.cache/huggingface/lerobot/calibration/
```

Recorded skills (`handy/skills/*.json`) are committed too; they only replay correctly on the arms they were recorded with.

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
2. Decide which port is the leader (the one with the handle; torque off). Confirm by unplugging one USB cable
   and re-running `ports.py`.
3. Calibrate both with the ids from `.env` (interactive: move each joint through its full range); after that
   `python arm.py` recognises the two boards by themselves:
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodem… --robot.id=follower_so101_left
   lerobot-calibrate --teleop.type=so101_leader  --teleop.port=/dev/tty.usbmodem… --teleop.id=leader_so101_left
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

## Spoken replies

Handy talks back through macOS `say` (offline). It acknowledges a command before the arm moves
("Screwdriver, coming up."), announces when a skill is done ("Here you go."), confirms stop,
says when it is busy, and apologises when it did not understand a sentence. Lines live in
`phrases.json`, a few variants per skill picked at random; `_default` covers new skills.
The mic is muted while it speaks so it never triggers itself.

Voice: the stock voice is dated. Download "Ava (Premium)" or "Zoe (Premium)" once (System Settings >
Accessibility > Spoken Content > System Voice > Manage Voices, ~400 MB), then set `TTS_VOICE=Ava` in `.env`.
Try it: `say -v Ava "Screwdriver, coming up."`

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
python train.py                                # ~40 epochs on the Mac GPU -> weights/handy.pt
python vision.py watch                         # boxes now come from the detector
```

- Labels come from background subtraction against `vision/empty.jpg`: frames with no object, two
  objects, or an object touching the frame edge are skipped (the window says so).
- Weak class? 30 more seconds of `collect.py <class>` then `train.py` again beats more epochs.
- `weights/handy.pt` (~6 MB) is committed so a fresh clone demos. `dataset/` and `runs/` are not.
- Re-check `vision.py watch` in the demo room; retrain only if a class drops out.

## Wrist camera dataset (for SAM alignment experiments)

`wristrec.py` records wrist-camera frames with the follower's measured joints next to each one.

`wristrec.py`, `adjust.py` and `visual_servoing.py` work on one arm: `--arm right|left`, default the first in
`ARMS` (the one carrying the wrist camera). They read and write only that arm's part of a skill.

```bash
python wristrec.py --probe                        # every camera tiled with its index; put the wrist one in .env as WRIST_CAMERA
python wristrec.py hover1                         # teleop with the leader: drive to a hover pose, h to hold,
                                                  # move the tool around under the camera with s (snapshot) / r (record), j once
python wristrec.py screw1 --skill screwdriver --hold-at-hover   # replay up to 2 s before the grasp, hold there, c continues
```

A skill can store its hold point as `"hover_frame": N` in its JSON (screwdriver4 has 141: whole tool in view,
room to slide it under the open jaws; black_motor2 has 219, its pause about 6 cm above where the jaws close). Then `--hold-at-hover` stops there instead of guessing 2 s before the grasp; `--hover-frame N` overrides both.

- `k` (only while holding) slides the gripper ±30 mm (`--slide-mm`) in base x and y at the same height, the way
  the line-up moves, and saves one settled frame per position, tagged `slide/<x|y>/<±mm>`. Those frames give the
  pixels-per-mm table directly. Press it twice for two runs.
- `j` (only while holding) nudges pan, lift, elbow and wrist_flex ±3° instead (`jog/<joint>/<±deg>`). The table can
  be fitted from these too, but it has to separate sliding from tilting and only covers small moves.
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
python visual_servoing.py screwdriver4 --fit-table screw4_2   # pixels per mm at the hover, from wristrec's k (or j) frames
python visual_servoing.py screwdriver4 --set-target           # tool exactly where the recording grasps it: save where it looks
python visual_servoing.py screwdriver4 --align-only --show    # move the tool, check it lines up (stops at the hover)
python visual_servoing.py screwdriver4 --show                 # line up + grasp
```

- Skill keys: `hover_frame`, `rejoin` [c, e] (full offset until frame c, back on the recording by e;
  screwdriver4: 250, 282; black_motor2: 338, 370, the start of the lift), `hover_px_per_mm`, `hover_target`, optional `prompt`.
- Reach: before every correction step the whole offset grasp is computed (about 60 ms); if any frame is out of reach,
  it stops. From black_motor2's hover that is 5.5 cm straight out (wrist_flex at its model limit),
  12 cm toward the base and 9.5-15 cm sideways.
  It also stops beyond `--max-offset` (100 mm) or when the error grows twice in a row (wrong object or bad table).
- On any stop after reaching the hover it goes home: back to the recorded hover, then the approach in reverse.
  Ctrl-C never moves the arm.
- `listen.py` loads SAM 3 + SAM 2 and opens the wrist camera at startup (~12 s), and plays every skill that has
  these keys (black_motor2, screwdriver4) only through this line-up, never blind: "pick up the motor" lines up on the
  motor first. If the line-up gives up, it says so and goes home; "stop" freezes the arm where it is.
  If the wrist camera is unplugged or sends no frames, `listen.py` won't start (it checks that `WRIST_CAMERA` is the
  USB camera, not the laptop's, which takes its index when the wrist one is unplugged). If it drops out mid-demo,
  the arm goes home and the demo stops. `--no-wrist` runs without it and refuses those skills.
- The joints sag (a commanded 3° jog moves about 1°), so one big move is never trusted: each step is measured again.
- The table is linear but the view isn't: sliding sideways turns the camera with pan, so px/mm falls from 3.4 at 3 cm
  to 2.8 at 9 cm, and the joints' backlash moves the settled pose ~4.5 mm with the approach direction. black_motor2
  keeps the 3 cm table (exact near the target; far away its steps fall short, which only costs an extra step).
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
- Checks run before every skill in a chain. `--no-bench-check` disables them. If `slots.json` or the photos
  are missing, listen.py says `bench camera check: off` and runs as before. (This is not the wrist camera.)
- Re-capture `full` and `empty` in the demo room under demo lighting, right before the demo (20 s).
- Boxes must not be covered by the arm in its home pose. Margins under 10 mean the box is ambiguous: make it tighter around where the tool actually sits, or redo setup.
- Real test: take one tool out and run `check --show`. Only that slot should turn red.
- The robot talks via macOS `say`; the mic is muted while it speaks so it cannot trigger itself.

## Demo-day tips

- Tape the tool slots to the bench. Replay is a player piano: it plays the same roll every time, so the tools must be where they were when you recorded.
- Lower the follower's speed/accel if grasps slip; smooth beats fast on video.
- Re-capture the two vision reference photos in the final room/lighting.
- Keep a hand on the power switch. `stop` (voice or typed) interrupts between frames.
- `python listen.py` only hears you between two Enters: press Enter (terminal in front), say the command, press Enter
  again. Talking to the audience never triggers it (20 s cap per command). `--ptt` is hold-to-talk on right Option from
  any window instead (needs the terminal allowed under Accessibility); `--open-mic` is the old always-listening mode.
- Noisy room: run with `--energy 0.02` or `0.03`, and hold the laptop mic close. Typing the command is an acceptable plan B — the judges care about the arm, not the mic.
- `--model small.en` if `base.en` mishears tool names; add misheard variants to `commands.json` instead of fighting the model (e.g. `"screw driver"`, `"screwdrivers"`).

## If the lerobot import path differs (already fixed for 0.5.x: `lerobot.robots.so_follower`, `lerobot.teleoperators.so_leader`)

Your installed version may use different module paths. Check with:
```bash
python -c "import lerobot, pkgutil; print(lerobot.__version__); print([m.name for m in pkgutil.iter_modules(lerobot.robots.__path__)])"
```
Only `arm.py` and `ports.py` touch lerobot, so those are the only files to adjust.
