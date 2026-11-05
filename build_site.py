"""Build the thesis's results site from the K v2b 12.5k gallery (GitHub Pages, branch gh-pages of two repos).

    python3 build_site.py <gallery dir> <site repo dir> <samples repo dir>

The project page (<site repo>, https://antoniomacaronio.github.io/aligning-video-motion/): index.html, a blog-style
write-up of the thesis rendered from post_template.html (its prose, the thesis figures that make_post_figures.py renders
into figures/, and the Chapter 4 tables), with the thesis's twelve Section 4.4 examples as interactive scenes. Each
example opens on its figure's task and has tabs for the same clip's other two tasks (make_thesis_scenes.py writes
recordings/thesis/ and media/thesis/; thesis_examples.json has the titles and captions). gallery.html redirects old
links to the gallery, and #results to the page's Results section.
The gallery (<samples repo>, https://antoniomacaronio.github.io/aligning-video-motion-samples/): index.html, the
gallery's 64 clips rewritten for an outside reader (a header in the thesis's terms, without internal paths, commit
hashes or run codenames; tab labels; a link back to the project page), its viser recordings, and its videos re-encoded
once (H.264 CRF 26, every frame a keyframe; media/.encoded.json). The two are separate repos because GitHub Pages caps
each site at 1 GB.

On both pages a 3D scene loads, and the videos beside it play, only while the example (or gallery row) is fully on
screen, or fills the screen when it is taller than it; the videos then follow the scene's playback clock (the viewer's
timeline, read from the same-origin iframe: video frame k is motion frame 3k).
"""
import html
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

src, site, samples = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
for d in (site, samples):
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", f"{src / 'viser'}/", f"{d / 'viser'}/"], check=True)
    (d / ".nojekyll").write_text("")
subprocess.run(["rsync", "-a", "--delete", f"{src / 'recordings'}/", f"{samples / 'recordings'}/"], check=True)
for old in ("recordings/training", "recordings/validation", "media/training", "media/validation"):  # moved to samples
    shutil.rmtree(site / old, ignore_errors=True)
(site / "media" / ".encoded.json").unlink(missing_ok=True)

ENCODING = "crf26-g1"
manifest_path = samples / "media" / ".encoded.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
sources = {p.relative_to(src / "media").as_posix(): p for p in (src / "media").rglob("*") if p.is_file()}


def encode(rel):
    source, out = sources[rel], samples / "media" / rel
    stamp = [source.stat().st_size, int(source.stat().st_mtime), ENCODING]
    if manifest.get(rel) == stamp and out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".mp4":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:v", "libx264", "-crf", "26", "-preset", "slow",
                        "-g", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)
    else:
        shutil.copy2(source, out)
    manifest[rel] = stamp


with ThreadPoolExecutor(8) as pool:
    list(pool.map(encode, sorted(sources)))
for out in (samples / "media").rglob("*"):  # media the gallery no longer has
    rel = out.relative_to(samples / "media").as_posix()
    if out.is_file() and out != manifest_path and rel not in sources:
        out.unlink()
manifest_path.write_text(json.dumps({k: manifest[k] for k in sorted(manifest) if k in sources}, indent=0))

FIGURES_URL, SAMPLES_URL = "https://antoniomacaronio.github.io/aligning-video-motion/", \
    "https://antoniomacaronio.github.io/aligning-video-motion-samples/"
# Relative links between the two sites, so that both also work side by side under one local server.
FIGURES_REL, SAMPLES_REL = "../aligning-video-motion/", "../aligning-video-motion-samples/"
BACKBAR = (f'<div class="backbar"><a href="{FIGURES_REL}">&larr; Aligning Egocentric Video and Human Motion Generation'
           '</a></div>')
BACKBAR_CSS = (".backbar { background: #fff; border-bottom: 1px solid #ddd; padding: 9px 22px; font-size: 14px; }\n"
               ".backbar a { color: #3182bd; text-decoration: none; }\n")


# Shared page script: the viewer's playback time, and "fully on screen" for an element.
SYNC_JS = """const VIDEO_FPS = 10, LAST_FRAME = 48;  // 49 video frames at 10 fps; video frame k is motion frame 3k
function sceneTime(frame) {
  if (frame.dataset.loaded !== "1") return null;
  try {
    const slider = frame.contentDocument && frame.contentDocument.querySelector('[role="slider"][aria-valuemax]');
    const t = slider ? parseFloat(slider.getAttribute("aria-valuenow")) : NaN;
    return Number.isFinite(t) ? t : null;
  } catch (error) {
    return null;
  }
}
function sceneFrame(t) { return Math.min(Math.max(Math.floor(t * VIDEO_FPS + 1e-6), 0), LAST_FRAME); }
// Start a video, first asking for its data if the browser has deferred loading it (it sometimes does).
function startVideo(v) {
  if (v.readyState === 0 && v.networkState !== HTMLMediaElement.NETWORK_LOADING) { v.preload = "auto"; v.load(); }
  v.play().catch(() => {});
}
// Fully on screen, or filling the screen when the element is taller than it.
function fullyVisible(entry) {
  const height = Math.max(entry.boundingClientRect.height, 1);
  return entry.isIntersecting && entry.intersectionRatio >= Math.min(0.98, 0.95 * innerHeight / height);
}
const THRESHOLDS = Array.from({ length: 41 }, (_, i) => i / 40);
"""

# ---------------------------------------------------------------- the samples site: the gallery page
s = (src / "index.html").read_text()

HEADER = f"""<header>
<h1>Gallery</h1>
<p class="byline">Samples for <a href="{FIGURES_REL}">Aligning Egocentric Video and Human Motion Generation</a> &middot;
Anthony Zhang &middot; M.S. thesis, Electrical Engineering and Computer Sciences, UC Berkeley, 2026</p>
<div class="notes">
<p>Samples from the final model, a HunyuanVideo&nbsp;1.5 video diffusion transformer extended with a motion stream and trained
with joint finetuning, a text curriculum and geometric post-training. Each row is one clip, sampled once per task with the
clip's fixed seed (50 Euler steps; text guidance 6 for the video and 2 for the motion).
<b>Joint generation</b> produces the video and the motion from the first frame, the first pose and the text prompt;
<b>video-conditioned motion</b> is given the recorded video and generates the motion;
<b>motion-conditioned video</b> is given the recorded motion and generates the video.
Below them, the joint and motion-conditioned videos are sampled again from the same seed with a <b>negative prompt</b>
in place of the empty prompt in text guidance.</p>
<p>The <b>3D scene</b> shows the recorded motion (gray), the jointly generated motion (blue) and the video-conditioned motion
(orange) as MHR bodies with the clip subject's body shape, posed from the motion's joint rotations. Each body's head is drawn
with the orientation of the Aria glasses, because NymeriaPlus's MHR fit tilts the head back relative to the glasses.
The camera frustums are estimated by VGGT-&Omega; from each video and anchored at the recorded camera's first pose.
Drag to orbit; the toggles below apply to every scene. A row's scene loads, and its videos play, when the row is fully on
screen, and the videos follow the scene's play bar, so pausing or scrubbing the scene moves all five.</p>
<p><b>Training-split clips</b>: 16 clips from the training split. <b>Held-out clips</b>: 48 clips from validation participants
who never appear in training. The method and the quantitative results are on the
<a href="{FIGURES_REL}">project page</a>.</p>
</div>
</header>"""

a = s.index("<header>")
b = s.index("</header>") + len("</header>")
s = s[:a] + HEADER + s[b:]
s = re.sub(r"<title>.*?</title>", "<title>Aligning Egocentric Video and Human Motion Generation: gallery</title>",
           s, count=1, flags=re.S)
# The project page's tab icon (make_favicon.py), from its URL.
s = s.replace("</title>", f'</title><link rel="icon" href="{FIGURES_URL}favicon.ico" sizes="any">'
              f'<link rel="icon" type="image/png" sizes="32x32" href="{FIGURES_URL}favicon-32.png">'
              f'<link rel="apple-touch-icon" href="{FIGURES_URL}apple-touch-icon.png">', 1)

for old, new in (('<button data-tab="training">Training (16/16)</button>', '<button data-tab="training">Training-split clips (16)</button>'),
                 ('<button data-tab="validation">Validation (48/48)</button>', '<button data-tab="validation">Held-out clips (48)</button>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
old = '<button data-tab="results">Results</button>'
assert s.count(old) == 1
s = s.replace(old, "")  # the results are on the project page
s = s.replace(" &middot; diagnostic clip</h3>", " &middot; training-split clip</h3>")
s = s.replace("<figcaption>GT video</figcaption>", "<figcaption>recorded video</figcaption>")
s = s.replace('<i class="chip" style="background:#c8c8c8"></i>GT</label>', '<i class="chip" style="background:#c8c8c8"></i>recorded</label>')
s = s.replace('></i>GT video</label>', '></i>recorded video</label>')
old = '<button onclick="playAll(this)">&#9654; play all five videos from the start</button> &nbsp;'
assert s.count(old) == 64, s.count(old)
s = s.replace(old, "")  # the scene's play bar drives all five videos
s = s.replace('<div class="hint">3D scene loads when scrolled into view</div>',
              '<div class="hint">the 3D scene loads when the row is fully on screen</div>')

a = s.index('<section class="tab" id="tab-results">')
b = s.index("</section>", a) + len("</section>")
s = s[:a] + s[b:]  # the gallery's own Results tab: the project page has the thesis's tables
s = s.replace("</style></head>", ".byline { color: #555; margin: 2px 0 10px; }\n"
              "footer { color: #666; font-size: 12.5px; padding: 0 22px 30px; }\n" + BACKBAR_CSS + "</style></head>", 1)
for old, new in (('<input type="radio" name="layout" value="overlay" checked>', '<input type="radio" name="layout" value="overlay">'),
                 ('<input type="radio" name="layout" value="side">', '<input type="radio" name="layout" value="side" checked>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)   # side by side is the default 3D layout
FOOTER = ("<footer>Data: NymeriaPlus, built on the Nymeria dataset of Project Aria recordings (Meta Reality Labs "
          "Research). Video model: HunyuanVideo&nbsp;1.5. Camera estimates: VGGT-&Omega;. 3D viewer: viser.</footer>")
s = s.replace("<script>", FOOTER + "\n<script>", 1)
s = s.replace("<header>", BACKBAR + "\n<header>", 1)

# The gallery's loader loaded every scene near the viewport; now a row's scene loads, and its five videos play, only
# while the row is fully on screen, and the videos follow the scene's playback clock.
a = s.index("const observer = new IntersectionObserver((entries) => {")
b = s.index('addEventListener("resize", schedule);') + len('addEventListener("resize", schedule);')
GALLERY_JS = SYNC_JS + """const rows = [...document.querySelectorAll(".row")].filter((row) => row.querySelector("iframe[data-src]")).map((row) => ({
  body: row.querySelector(".body"), frame: row.querySelector("iframe[data-src]"), videos: [...row.querySelectorAll("video")],
  shown: -1, on: false,
}));
function setOn(r, on) {
  if (r.on === on) return;
  r.on = on;
  r.shown = -1;
  if (on) { load(r.frame); r.videos.forEach(startVideo); }
  else { unload(r.frame); r.videos.forEach((v) => v.pause()); }
}
const observer = new IntersectionObserver((entries) => entries.forEach((e) => {
  const r = rows.find((row) => row.body === e.target);
  if (r) setOn(r, fullyVisible(e));
}), { threshold: THRESHOLDS });
rows.forEach((r) => observer.observe(r.body));
function refresh() {}
function tick() {
  for (const r of rows) {
    if (!r.on) continue;
    const t = sceneTime(r.frame);
    if (t === null) {  // the scene is still loading: the videos play on their own
      r.videos.forEach((v) => { if (v.readyState === 0 && v.networkState !== HTMLMediaElement.NETWORK_LOADING) startVideo(v); });
      continue;
    }
    const k = sceneFrame(t);
    if (k === r.shown) continue;
    r.shown = k;
    for (const v of r.videos) {
      if (!v.paused) v.pause();
      if (v.readyState >= 1) v.currentTime = (k + 0.5) / VIDEO_FPS;
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);"""
s = s[:a] + GALLERY_JS + s[b:]
old = 'showTab(location.hash.slice(1) || "training");'
assert s.count(old) == 1
s = s.replace(old, f'if (location.hash === "#results") location.replace("{FIGURES_URL}#results");\n' + old
              + '\naddEventListener("hashchange", () => showTab(location.hash.slice(1) || "training"));')
(samples / "index.html").write_text(s)
(samples / "README.md").write_text(
    "# Gallery: Aligning Egocentric Video and Human Motion Generation\n\n"
    f"The 64-clip gallery of the thesis's project page ({FIGURES_URL}), served at {SAMPLES_URL}.\n"
    "Built by `build_site.py` in the project page's repo (AntonioMacaronio/aligning-video-motion); do not edit "
    "the files here by hand.\n")

# ---------------------------------------------------------------- the project page
(site / "gallery.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Gallery</title>'
    f'<meta http-equiv="refresh" content="0; url={SAMPLES_URL}">'
    f'<script>location.replace(location.hash === "#results" ? "{FIGURES_URL}#results" : "{SAMPLES_URL}" + location.hash);'
    f'</script></head><body><p>The gallery moved to <a href="{SAMPLES_URL}">{SAMPLES_URL}</a>.</p></body></html>\n')

TASKS = {
    "joint": dict(label="Joint generation", note="The video and the motion are both generated from the text prompt and the "
                  "first video frame.", video="Generated video", motion="Generated motion"),
    "motion_cond": dict(label="Motion-conditioned video", note="The video is generated from the text prompt, the first frame "
                        "and the recorded motion shown.", video="Generated video", motion="Input motion (recorded)"),
    "video_cond": dict(label="Video-conditioned motion", note="The motion is generated from the text prompt and the recorded "
                       "video shown.", video="Input video (recorded)", motion="Generated motion"),
}
SECTIONS = (("joint", "Figures 4.1 and 4.2"), ("motion_cond", "Figures 4.3 and 4.4"), ("video_cond", "Figures 4.5 and 4.6"))
examples = json.loads((site / "thesis_examples.json").read_text())
for video in sorted((site / "media" / "thesis").glob("*.mp4")):
    poster = video.with_suffix(".jpg")
    if not poster.exists() or poster.stat().st_mtime < video.stat().st_mtime:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(poster)], check=True)


def pane(e, task, active):
    """One task of an example: its video beside its 3D scene (the figure's task under the example's own name)."""
    name = e["name"] if task == e["mode"] else f"{e['name']}__{task}"
    info = json.loads((site / "recordings" / "thesis" / f"{name}.json").read_text())
    # The site camera frames the content like the figure's tight crop (make_thesis_scenes.site_camera). Without a
    # field of view the viewer would use three.js's 75 deg, wider than the figure's 50.
    position, look_at = info.get("site_position", info["position"]), info.get("site_look_at", info["look_at"])
    camera = (f"initialCameraPosition={','.join(map(str, position))}"
              f"&amp;initialCameraLookAt={','.join(map(str, look_at))}&amp;initialCameraUp=0,0,1"
              f"&amp;initialCameraFov={info.get('site_fov', info['fov']):.4f}")
    t = TASKS[task]
    return (f'<div class="pane" data-task="{task}"{"" if active else " hidden"}><p class="task-note">{t["note"]}</p>'
            f'<div class="body" style="--aspect: {info["aspect"]:.3f}"><div class="video"><p class="panel-label">{t["video"]}</p>'
            f'<video src="media/thesis/{name}.mp4" poster="media/thesis/{name}.jpg" muted loop playsinline preload="none">'
            f'</video></div><div class="motion"><p class="panel-label">{t["motion"]}</p><div class="scene">'
            f'<div class="hint">the 3D scene loads when the example is fully on screen</div>'
            f'<iframe data-src="viser/index.html?playbackPath=../recordings/thesis/{name}.viser&amp;{camera}" '
            f'title="{html.escape(e["title"])}: {t["label"].lower()}"></iframe></div></div></div></div>')


def card(e, shown, opens=None):
    """One example: its title and caption, a tab per task and a pane per task, open on `opens` (default: its figure's
    task, which its tab marks)."""
    mode, opens = e["mode"], opens or e["mode"]
    tabs = "".join(
        f'<button role="tab" data-task="{task}" aria-selected="{"true" if task == opens else "false"}">{t["label"]}'
        f'{" <span class=fig-mark>thesis figure</span>" if task == mode else ""}</button>' for task, t in TASKS.items())
    return (f'<div class="example" id="{e["name"]}"{"" if shown else " hidden"}><h4>{html.escape(e["title"])}</h4>'
            f'<p class="caption">{html.escape(e["caption"])}</p><div class="tabs" role="tablist">{tabs}</div>'
            + "".join(pane(e, task, task == opens) for task in TASKS) + "</div>")


PAGE_JS = """const RAMP = [[46, 184, 114], [52, 136, 219], [142, 68, 173]];
function ramp(f) {
  const x = Math.min(Math.max(f, 0), 1) * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), t = x - i;
  return "rgb(" + RAMP[i].map((c, k) => Math.round(c + t * (RAMP[i + 1][k] - c))).join(",") + ")";
}
const MAX_LIVE = 3, live = [];
function unload(frame) {
  const i = live.indexOf(frame);
  if (i >= 0) live.splice(i, 1);
  if (frame.dataset.loaded === "1") { frame.src = "about:blank"; frame.dataset.loaded = "0"; }
  frame.previousElementSibling.style.display = "";
}
function load(frame) {
  if (frame.dataset.loaded === "1") { live.splice(live.indexOf(frame), 1); live.push(frame); return; }
  frame.src = frame.dataset.src; frame.dataset.loaded = "1"; frame.previousElementSibling.style.display = "none";
  live.push(frame);
  while (live.length > MAX_LIVE) unload(live[0]);
}
// One entry per task pane; only the selected pane of an example is displayed, so only it can become visible.
const panes = [...document.querySelectorAll(".pane")].map((pane) => ({
  pane, body: pane.querySelector(".body"), video: pane.querySelector("video"), frame: pane.querySelector("iframe"),
  shown: -1, on: false,
}));
function setOn(p, on) {
  if (p.on === on) return;
  p.on = on;
  p.shown = -1;
  if (on) { load(p.frame); startVideo(p.video); }
  else { unload(p.frame); p.video.pause(); }
}
const observer = new IntersectionObserver((entries) => entries.forEach((e) => {
  const p = panes.find((x) => x.body === e.target);
  if (p) setOn(p, !p.pane.hidden && fullyVisible(e));
}), { threshold: THRESHOLDS });
panes.forEach((p) => observer.observe(p.body));
document.querySelectorAll(".example").forEach((example) => {
  example.querySelectorAll('.tabs [role="tab"]').forEach((tab) => tab.addEventListener("click", () => {
    example.querySelectorAll('.tabs [role="tab"]').forEach((b) => b.setAttribute("aria-selected", String(b === tab)));
    for (const p of panes) {
      if (!example.contains(p.pane)) continue;
      const selected = p.pane.dataset.task === tab.dataset.task;
      if (!selected) setOn(p, false);
      p.pane.hidden = !selected;
    }
  }));
});
function tick() {
  for (const p of panes) {
    if (!p.on) continue;
    const v = p.video, t = sceneTime(p.frame);
    if (t !== null && v.readyState >= 1) {
      if (!v.paused) v.pause();
      const k = sceneFrame(t);
      if (k !== p.shown) { v.currentTime = (k + 0.5) / VIDEO_FPS; p.shown = k; }
      v.style.borderColor = ramp(k / LAST_FRAME);  // the body's color at motion frame 3k
    } else {
      if (v.readyState === 0 && v.networkState !== HTMLMediaElement.NETWORK_LOADING) startVideo(v);
      if (v.duration) v.style.borderColor = ramp(v.currentTime / v.duration);  // the scene is loading: the video plays on its own
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
// Contents: mark the section being read, and fade the list while a wide figure or example passes beside it.
const tocLinks = [...document.querySelectorAll(".toc a")];
const tocTargets = tocLinks.map((a) => document.getElementById(a.hash.slice(1)));
function markToc() {
  let current = 0;
  tocTargets.forEach((h, i) => { if (h.getBoundingClientRect().top < innerHeight * 0.35) current = i; });
  tocLinks.forEach((a, i) => a.classList.toggle("current", i === current));
}
addEventListener("scroll", markToc, { passive: true });
markToc();
const toc = document.querySelector(".toc"), besideToc = new Set();
const tocObserver = new IntersectionObserver((entries) => {
  entries.forEach((e) => (e.isIntersecting ? besideToc.add(e.target) : besideToc.delete(e.target)));
  toc.classList.toggle("dim", besideToc.size > 0);
}, { rootMargin: "-90px 0px -35% 0px" });
document.querySelectorAll("main .wide").forEach((el) => tocObserver.observe(el));
// The thesis figures viewer: one example on the stage at a time, picked from the strip; #<example> opens one.
const stageCards = [...document.querySelectorAll(".stage .example")], thumbs = [...document.querySelectorAll(".thumb")];
function showExample(name) {
  const card = stageCards.find((c) => c.id === name) || stageCards[0];
  for (const c of stageCards) {
    if (c !== card) for (const p of panes) if (c.contains(p.pane)) setOn(p, false);
    c.hidden = c !== card;
  }
  thumbs.forEach((t) => t.setAttribute("aria-pressed", String(t.dataset.example === card.id)));
}
thumbs.forEach((t) => t.addEventListener("click", () => {
  showExample(t.dataset.example);
  history.replaceState(null, "", "#" + t.dataset.example);
}));
const linked = stageCards.find((c) => "#" + c.id === location.hash);
showExample(linked ? linked.id : stageCards[0].id);
if (linked) document.getElementById("examples").scrollIntoView();"""

MAIN = (site / "post_template.html").read_text()
# The viewer's examples in thesis order (Figures 4.1 to 4.6); it opens on the first. To lead with another example, name
# it here and the task its tab opens on, e.g. LEAD, LEAD_TASK = "motion_cond_3", "joint".
LEAD, LEAD_TASK = None, None
order = sorted(examples, key=lambda e: e["name"] != LEAD) if LEAD else examples
strip = "".join(
    f'<button type="button" class="thumb" data-example="{e["name"]}" aria-pressed="{"true" if i == 0 else "false"}" '
    f'title="{html.escape(e["title"])}"><img src="figures/thumbs/{e["name"]}.jpg" width="176" height="176" '
    f'alt="{html.escape(e["title"])}"></button>' for i, e in enumerate(order))
slots = {"<!--STAGE-->": "\n".join(card(e, i == 0, LEAD_TASK if i == 0 else None) for i, e in enumerate(order)),
         "<!--THUMBS-->": strip,
         "<!--FOOTER-->": FOOTER, "<!--SCRIPT-->": SYNC_JS + PAGE_JS}
for slot, value in slots.items():
    assert MAIN.count(slot) == 1, slot
    MAIN = MAIN.replace(slot, value)
assert MAIN.count("<!--GALLERY_URL-->") == 1, MAIN.count("<!--GALLERY_URL-->")
MAIN = MAIN.replace("<!--GALLERY_URL-->", SAMPLES_REL)
assert not re.search(r"<!--[A-Z_:a-z]+-->", MAIN), "unfilled slot"
# Animated overlays (make_post_figures.py's OVERLAID) reach data-pad drawing units past their figure on every side;
# extend each <img class="anim"> past the figure's box by as much, in percent of that box.
def _place_overlay(m):
    head = (site / m.group(1)).read_text()[:600]
    w, h, pad = (float(re.search(rf'{key}="([\d.]+)"', head).group(1)) for key in ("width", "height", "data-pad"))
    fw, fh = w - 2 * pad, h - 2 * pad
    style = (f"left: {-100 * pad / fw:.4f}%; top: {-100 * pad / fh:.4f}%; width: {100 * w / fw:.4f}%; "
             f"height: {100 * h / fh:.4f}%")
    return f'{m.group(0)} style="{style}"'


MAIN = re.sub(r'<img class="anim" src="(figures/[^"]+_anim\.svg)"', _place_overlay, MAIN)
(site / "index.html").write_text(MAIN)

for page, text in (("samples index.html", s), ("site index.html", MAIN)):
    leftover = [p for p in (r"\bLVP\b", r"c1f6727|480f69b", r"Run [A-Z]\b", r"\bv2b\b",
                            r"A-cont|I-cc|I-cont", r"docs/", r"vis/", r"launch-records", r"s3://", r"grid ?2",
                            r"Hunyuan viewer")
                if re.search(p, re.sub(r"<script.*?</script>", "", text, flags=re.S))]
    print(page, len(text), "bytes; leftover internal markers:", leftover)
