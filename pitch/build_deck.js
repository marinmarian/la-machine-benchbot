const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const OUT = process.env.OUT || "benchbot_pitch.pptx";

// Workbench palette: graphite dominant, safety-orange accent, steel grey support
const THEME = {
  name: "BenchBot",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1D21", lt1: "FFFFFF", dk2: "2E3238", lt2: "EEF0F2",
    accent1: "F26B1D", accent2: "8A94A3", accent3: "C8CED6",
    accent4: "F2A05A", accent5: "4A5260", accent6: "9DA6B3",
    hlink: "F26B1D", folHlink: "8A94A3",
  },
};


(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9"; // 10 x 5.625
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.title = "BenchBot pitch";
  const C = pres.SchemeColor;

  // --- layouts ---
  pres.defineSlideMaster({
    title: "DARK", background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.4, w: 8.8, h: 1.7, fontSize: 40, bold: true, color: C.background1, margin: 0, align: "left", valign: "bottom" } } },
      { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.3, w: 8.0, h: 1.0, fontSize: 20, color: C.accent3, margin: 0, align: "left" } } },
    ],
  });
  pres.defineSlideMaster({
    title: "LIGHT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 8.8, h: 0.8, fontSize: 32, bold: true, color: C.text1, margin: 0, align: "left" } } },
      { text: { text: "BenchBot · La Machine 2026", options: { x: 0.6, y: 5.2, w: 5, h: 0.3, fontSize: 10, color: C.accent2, margin: 0 } } },
    ],
    slideNumber: { x: 9.2, y: 5.2, w: 0.4, h: 0.3, fontSize: 10, color: C.accent2 },
  });

  const icons = { hand:"↩", screw:"✦", broom:"∿", mic:"🎙", robot:"⚙", wifi:"◉", gavel:"§", tag:"#", play:"▶", hands:"✦" };

  const iconCircle = (s, glyph, x, y, d = 0.7) => {
    s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "icon circle" });
    s.addText(glyph, { x, y, w: d, h: d, fontSize: d * 26, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: "icon glyph" });
  };

  // ---- 1. Title ----
  pres.addSection({ title: "Pitch" });
  let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Pitch" });
  iconCircle(s, "B", 0.6, 0.8, 0.8);
  s.addText("BenchBot", { placeholder: "title" });
  s.addText("A second pair of hands for the repair bench", { placeholder: "body" });
  s.addText("Teach it by showing. Call it by voice. Runs offline.", { x: 0.6, y: 4.5, w: 8, h: 0.4, fontSize: 14, color: C.accent1, italic: true, margin: 0, isTextBox: true, objectName: "tagline" });
  s.addNotes("Hook: every technician has two hands. One holds the phone. Say 'give me the screwdriver' and let the arm run.");

  // ---- 2. Problem ----
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Pitch" });
  s.addText("A repair bench is a one-person factory", { placeholder: "title" });
  s.addText("45", { x: 0.6, y: 1.5, w: 2.6, h: 1.3, fontSize: 84, bold: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "stat" });
  s.addText("minutes for a typical screen or battery swap", { x: 0.6, y: 2.85, w: 2.7, h: 0.8, fontSize: 14, color: C.accent5, margin: 0, isTextBox: true, objectName: "stat label" });
  s.addText("Not all of it is repair. One hand does a courier's job all day.", { x: 0.6, y: 3.8, w: 2.9, h: 1.0, fontSize: 14, italic: true, color: C.text2, margin: 0, isTextBox: true, objectName: "stat note" });
  const rows = [
    ["1", "Fetch", "Walk to the rack, find the right bit, come back. Eyes leave the device."],
    ["2", "Return", "Put it down somewhere. Lose it. Look for it on the next job."],
    ["3", "Clean", "Clear screws, swarf and adhesive before the next phone can go down."],
  ];
  rows.forEach(([ic, h, d], i) => {
    const y = 1.5 + i * 1.15;
    iconCircle(s, ic, 4.0, y);
    s.addText(h, { x: 4.9, y: y - 0.02, w: 4.5, h: 0.35, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "row head " + i });
    s.addText(d, { x: 4.9, y: y + 0.33, w: 4.5, h: 0.6, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: "row desc " + i });
  });
  s.addNotes("Shops can't hire their way out. Margins thin, parts getting pricier. Take the screwdriver from the gripper here.");

  // ---- 3. Product ----
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Pitch" });
  s.addText("Show it once. Name it. Say it.", { placeholder: "title" });
  const steps = [
    ["1", "Show", "Grab the leader arm and do the motion once. The follower mirrors and records every joint at 30 Hz."],
    ["2", "Name", "Call it \"screwdriver\", \"take it back\", \"clean up\". Chain them into routines."],
    ["3", "Say", "Local speech on the laptop matches your words to a skill and replays it. Say \"stop\" and it stops."],
  ];
  steps.forEach(([ic, h, d], i) => {
    const x = 0.6 + i * 3.0;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w: 2.75, h: 2.6, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "card " + i });
    iconCircle(s, ic, x + 0.25, 1.75, 0.6);
    s.addText(h, { x: x + 1.0, y: 1.8, w: 1.6, h: 0.5, fontSize: 22, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "step head " + i });
    s.addText(d, { x: x + 0.25, y: 2.5, w: 2.25, h: 1.5, fontSize: 13, color: C.text2, margin: 0, isTextBox: true, objectName: "step desc " + i });
  });
  s.addText([
    { text: "No code. No training run. No cloud. ", options: { bold: true, color: C.text1 } },
    { text: "A shop owner adds a new tool in ninety seconds, and the whole thing runs with the wifi off.", options: { color: C.text2 } },
  ], { x: 0.6, y: 4.35, w: 8.8, h: 0.6, fontSize: 14, margin: 0, isTextBox: true, objectName: "product note" });
  s.addNotes("Say 'take it back', then 'clean up'. Hold up the leader arm on 'Show'.");

  // ---- 4. Why now ----
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Pitch" });
  s.addText("Why this works in 2026 and didn't in 2022", { placeholder: "title" });
  const now = [
    ["§", "Right to Repair", "EU directive applies since 31 July 2026. Manuals, parts pricing, longer warranties. More repairs, more benches."],
    ["€", "€25k → €250", "A 6-axis arm with a leader controller is now hobby-kit money (Hugging Face SO-101). Hardware stops being the business."],
    ["))", "Speech, offline", "Whisper-class recognition runs on a laptop CPU. A ten-word vocabulary needs no server and no account."],
  ];
  now.forEach(([ic, h, d], i) => {
    const x = 0.6 + i * 3.0;
    iconCircle(s, ic, x, 1.5, 0.7);
    s.addText(h, { x, y: 2.4, w: 2.7, h: 0.5, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "now head " + i });
    s.addText(d, { x, y: 2.95, w: 2.7, h: 1.6, fontSize: 14, color: C.text2, margin: 0, isTextBox: true, objectName: "now desc " + i });
  });
  s.addNotes("Three things changed this year. Point at the arm on the second one. Mention the wifi is off right now.");

  // ---- 5. Business ----
  s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Pitch" });
  s.addText("We sell the bench, not the robot", { placeholder: "title" });
  // left: ROI callout
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.45, w: 4.1, h: 3.5, rectRadius: 0.12, fill: { color: C.text1 }, line: { color: C.text1 }, objectName: "roi card" });
  s.addText("~€750", { x: 0.9, y: 1.65, w: 3.5, h: 0.9, fontSize: 54, bold: true, color: C.accent1, margin: 0, isTextBox: true, objectName: "roi stat" });
  s.addText("technician time saved per bench, per month", { x: 0.9, y: 2.55, w: 3.5, h: 0.4, fontSize: 14, color: C.accent3, margin: 0, isTextBox: true, objectName: "roi label" });
  s.addText([
    { text: "8 min saved × 10 repairs/day × €26/h", options: { breakLine: true } },
    { text: "Hardware + fixture ≈ €600 to build", options: { breakLine: true } },
    { text: "Subscription for the skill library", options: {} },
  ], { x: 0.9, y: 3.15, w: 3.5, h: 1.6, fontSize: 13, color: C.accent3, margin: 0, isTextBox: true, paraSpaceAfter: 6, objectName: "roi lines" });
  // right: market + expansion
  s.addText("Wedge", { x: 5.1, y: 1.45, w: 4.3, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0, isTextBox: true, objectName: "wedge label" });
  s.addText([
    { text: "Phone & laptop repair. ", options: { bold: true, color: C.text1 } },
    { text: "Standardised work, shops everywhere. Europe consumer electronics repair ≈ €2.9bn; UK mobile repair alone ≈ £690m across ~640 businesses.", options: { color: C.text2 } },
  ], { x: 5.1, y: 1.8, w: 4.3, h: 1.3, fontSize: 14, margin: 0, isTextBox: true, objectName: "wedge text" });
  s.addText("Then any bench", { x: 5.1, y: 3.15, w: 4.3, h: 0.35, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0, isTextBox: true, objectName: "expand label" });
  s.addText([
    { text: "Watch repair · dental labs · electronics assembly · bike shops · teaching labs. ", options: { color: C.text2 } },
    { text: "Every recorded skill joins a shared catalog. The arm is the first app; the library is the company.", options: { bold: true, color: C.text1 } },
  ], { x: 5.1, y: 3.5, w: 4.3, h: 1.5, fontSize: 14, margin: 0, isTextBox: true, objectName: "expand text" });
  s.addNotes("Like a label printer: cheap hardware, value is what it does every day. ROI line is our assumption; the pilot measures it. Sources: IBISWorld 2025/2026.");

  // ---- 6. Close ----
  s = pres.addSlide({ masterName: "DARK", sectionTitle: "Pitch" });
  s.addText("One arm. Three skills. No internet.", { placeholder: "title" });
  s.addText("Next: ten repair shops in Amsterdam, three months, minutes saved per repair measured on the bench.", { placeholder: "body" });
  s.addText("Say \"go home\". Let it park.", { x: 0.6, y: 4.5, w: 8, h: 0.4, fontSize: 14, color: C.accent1, italic: true, margin: 0, isTextBox: true, objectName: "cue" });
  s.addNotes("Say 'go home'. Stop talking when the arm parks.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
