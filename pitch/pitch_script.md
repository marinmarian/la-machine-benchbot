# BenchBot — 3-minute pitch + demo script

Working name: **BenchBot** (alternatives: Benchmate, Second Hand, Handoff). Pick one before you print anything.

Stage setup before you're called: arm powered, `listen.py` running, laptop mic facing the speaker, tools in taped slots, one phone on the bench with its back cover off. One teammate talks, one stands next to the arm with a hand near the power switch, one drives slides.

---

## 0:00 — Hook (15 s) · Slide 1

> "Every repair technician has two hands. One holds the phone. The other spends the day walking to the tool rack."

Turn to the arm. Say clearly: **"Give me the screwdriver."**
Let the arm run while you talk. Don't narrate the arm; let the judges watch it.

## 0:15 — Problem (30 s) · Slide 2

> "A repair bench is a one-person factory. A screen or battery swap takes about 45 minutes, and a chunk of that isn't repair at all: fetching, putting back, cleaning the bench for the next job. Shops can't hire their way out of it; margins are thin and parts are getting more expensive. The technician's hands are the bottleneck, and half the time one hand is doing a courier's job."

Take the screwdriver from the gripper when it's offered. Do one visible turn on the phone.

## 0:45 — Product (40 s) · Slide 3

> "BenchBot is a second pair of hands that lives on the bench. It hands you tools, takes them back, and clears the bench when you're done. You talk to it, because your eyes are on the magnifier and your hands are full."

Say: **"Take it back."** Then: **"Clean up."**

> "The important part is how you teach it. You don't program it. You grab the leader arm, do the motion once, say what it's called. Done. A shop owner can add a new tool in ninety seconds. No code, no training run, no cloud. Everything you see runs offline on this laptop."

Hold up the leader arm when you say "grab the leader arm."

## 1:25 — Why now (30 s) · Slide 4

> "Three things changed this year. The EU Right to Repair directive started applying in July, so repair volume is going up, not down. Arms like this one went from a twenty-five-thousand-euro cobot to a couple of hundred euros. And speech recognition that used to need a server now runs on a laptop with the wifi off, which it is right now."

## 1:55 — Business (40 s) · Slide 5

> "We sell the bench, not the robot: the arm, the fixture, the mic, and a subscription for the skill library. Think of it like a label printer. Cheap hardware, and the value is in what it does every day.
>
> The math for a shop: if BenchBot saves eight minutes on a forty-five-minute repair, at ten repairs a day that's over an hour of technician time per day, roughly 750 euros a month. We charge a fraction of that.
>
> Phone and laptop repair is the wedge because the work is standardized and the shops are everywhere: in the UK alone the mobile repair industry is about 690 million pounds. But the product is a bench helper. Watch repair, dental labs, electronics assembly, bike shops, teaching labs, the same bench, different tools.
>
> And every skill a shop records can go into a shared catalog. 'iPhone 15 battery flow' recorded once in Rotterdam, replayed in Lisbon. The arm is the first app. The library is the company."

## 2:35 — Close (20 s) · Slide 6

> "This weekend: one arm, three skills, no training, no internet. Next: ten repair shops in Amsterdam, three months, measure minutes saved per repair. We're BenchBot. Thanks."

Say: **"Go home."** The arm parks. Stop talking.

---

## Demo plan B / C

- If voice mishears twice: type the command. Say "it's loud in here" and move on. Judges score the arm, not the mic.
- If a grasp slips: say "stop", put the tool back in the slot by hand, repeat the command. One retry max, then skip to the next skill.
- If the arm won't connect: play the recorded video (record one tonight when it works!) and demo the record workflow with the leader arm unplugged from the follower.
- Record a clean 60-second video of the full sequence tonight regardless. Put it on the laptop, not in the cloud.

## Q&A prep

**"What if the tool isn't exactly in the slot?"**
Today: fixtures, like a surgical tray. Everything has a home and the arm knows where home is. That's how real repair benches already work. Next version adds a wrist camera and a small learned policy so it tolerates a few centimeters of slop; the record workflow stays the same, the recordings become training data.

**"Why replay instead of a learned policy?"**
Deliberate. Replay is deterministic, debuggable by a shop owner, and works with one demonstration instead of fifty. The learned version is the roadmap, and every replay session is a labeled demo we can train on later.

**"Is it safe next to a person's hands?"**
Low-torque hobby servos: it can't hurt you, it can at most drop a screwdriver. Voice stop word, and the joint-step limiter in software caps how far it can move per tick. For the product we'd add a torque threshold stop.

**"Why voice? Why not a foot pedal or a button?"**
Hands are full, eyes are on a magnifier, and the vocabulary is tiny, which is why it works offline. A pedal is a fine addition for "take it back"; voice is what lets you name which tool.

**"Isn't this just a fancy tool rack?"**
The handoff and the clean-up are the point. The tech never looks up or puts the device down. Multiply by thirty repairs a day.

**"How big is the market, really?"**
Consumer electronics repair in Europe is roughly €2.9bn (IBISWorld, 2025). UK mobile phone repair alone is about £690m in 2026 with ~640 businesses; Germany's consumer electronics repair is about €400m. Those are the wedge. The bench-helper market is any small-batch manual workstation.

**"What's the moat?"**
The skill library plus the fixture standard. Hardware is commodity on purpose. Whoever has the most recorded, rated, shop-specific skills wins, same as app stores.

**"What did you build this weekend?"**
Record-and-replay pipeline on the SO-101, offline speech with fuzzy command matching, skill chaining, voice stop, and three bench skills. Everything on the laptop, nothing in the cloud.

## Numbers and where they come from

| Claim | Source / basis |
|---|---|
| Screen/battery repair ≈ 45 min | Repair-shop FAQ (FixStop) |
| EU Right to Repair directive applies from 31 July 2026 | Engadget / Thommessen |
| Europe consumer electronics repair ≈ €2.9bn (2025) | IBISWorld |
| UK mobile phone repair ≈ £689m, ~636 businesses (2026) | IBISWorld |
| Germany consumer electronics repair ≈ €404m (2026) | IBISWorld |
| Arm hardware ≈ €250–300 | Seeed SO-ARM101 Pro street price (~$247 on deal) |
| Technician cost ≈ €26–30/h | US repair-tech salary data (RepairDesk); EU wage lower, so ROI is conservative in € |
| 8 min saved per repair, 10 repairs/day → ~€750/month | **Our assumption.** Say so if asked; offer to measure in the pilot. |
