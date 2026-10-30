"""Build the thesis's two results sites from the K v2b 12.5k gallery (GitHub Pages, branch gh-pages of each repo).

    python3 build_site.py <gallery dir> <site repo dir> <samples repo dir>

The thesis figures site (<site repo>, https://antoniomacaronio.github.io/aligning-video-motion/):
- index.html: the thesis's Section 4.4 examples. Each opens on its figure's task and has tabs for the same clip's
  other two tasks (make_thesis_scenes.py writes recordings/thesis/ and media/thesis/; thesis_examples.json has the
  titles and captions);
- gallery.html: a redirect to the samples site, so old links (including #results) keep working.
The samples site (<samples repo>, https://antoniomacaronio.github.io/aligning-video-motion-samples/): index.html, the
gallery's 64 clips rewritten for an outside reader (a header in the thesis's terms, without internal paths, commit
hashes or run codenames; tab labels; a Results tab with the thesis's Chapter 4 tables), its viser recordings, and its
videos re-encoded once (H.264 CRF 26, every frame a keyframe; media/.encoded.json). The two are separate repos
because GitHub Pages caps each site at 1 GB.

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
SITENAV_CSS = (".sitenav { background: #fff; border-bottom: 1px solid #ddd; padding: 0 22px; display: flex; gap: 2px; }\n"
               ".sitenav a { font-size: 14px; padding: 9px 14px; color: #3182bd; text-decoration: none; }\n"
               ".sitenav a.active { color: #222; font-weight: 600; }\n")


def sitenav(active):
    """Links between the two sites; on the samples site, #results opens its Results tab."""
    here = {"figures": ("index.html", "../aligning-video-motion-samples/"),
            "gallery": ("../aligning-video-motion/", "index.html")}[active]
    links = ((here[0], "Thesis figures", "figures"), (here[1], "More samples", "gallery"),
             (here[1] + "#results", "Results", "results"))
    marked = ' class="active"'
    return '<div class="sitenav">' + "".join(
        f'<a href="{href}"{marked if key == active else ""}>{label}</a>' for href, label, key in links) + "</div>"


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

HEADER = """<header>
<h1>Aligning Egocentric Video and Human Motion Generation</h1>
<p class="byline">Anthony Zhang &middot; M.S. thesis, Electrical Engineering and Computer Sciences, UC Berkeley, 2026</p>
<div class="notes">
<p>Samples from the final model, a HunyuanVideo&nbsp;1.5 video diffusion transformer extended with a motion stream and trained
with joint finetuning, a text curriculum and geometric post-training. Each row is one clip, sampled once per task with the
clip's fixed seed (50 Euler steps; text guidance 6 for the video and 2 for the motion).
<b>Joint generation</b> produces the video and the motion from the first frame, the first pose and the caption;
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
who never appear in training. <b>Results</b>: the thesis's quantitative comparison of the training phases.</p>
</div>
</header>"""

a = s.index("<header>")
b = s.index("</header>") + len("</header>")
s = s[:a] + HEADER + s[b:]
s = re.sub(r"<title>.*?</title>", "<title>Aligning Egocentric Video and Human Motion Generation: more samples</title>",
           s, count=1, flags=re.S)

for old, new in (('<button data-tab="training">Training (16/16)</button>', '<button data-tab="training">Training-split clips (16)</button>'),
                 ('<button data-tab="validation">Validation (48/48)</button>', '<button data-tab="validation">Held-out clips (48)</button>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
s = s.replace(" &middot; diagnostic clip</h3>", " &middot; training-split clip</h3>")
s = s.replace("<figcaption>GT video</figcaption>", "<figcaption>recorded video</figcaption>")
s = s.replace('<i class="chip" style="background:#c8c8c8"></i>GT</label>', '<i class="chip" style="background:#c8c8c8"></i>recorded</label>')
s = s.replace('></i>GT video</label>', '></i>recorded video</label>')
old = '<button onclick="playAll(this)">&#9654; play all five videos from the start</button> &nbsp;'
assert s.count(old) == 64, s.count(old)
s = s.replace(old, "")  # the scene's play bar drives all five videos
s = s.replace('<div class="hint">3D scene loads when scrolled into view</div>',
              '<div class="hint">the 3D scene loads when the row is fully on screen</div>')

ROWS = [("", ""), ("&#10003;", ""), ("", "&#10003;"), ("&#10003;", "&#10003;")]


def table(title, note, columns, rows, groups=None):
    head = ""
    if groups:
        head += "<tr><th></th><th></th>" + "".join(f'<th colspan="{n}">{g}</th>' for g, n in groups) + "</tr>"
    head += "<tr><th>Curriculum</th><th>Geometry</th>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    body = ""
    for (c, g), values in zip(ROWS, rows):
        body += f"<tr><td>{c}</td><td>{g}</td>" + "".join(f"<td>{v}</td>" for v in values) + "</tr>"
    return f'<div class="chart"><h3>{title}</h3><p>{note}</p><table class="results">{head}{body}</table></div>'


RESULTS = '<section class="tab" id="tab-results"><div class="notes" style="margin-bottom:12px">' + (
    "Four models, each trained for 12,500 updates: joint finetuning alone, with the curriculum, with geometric "
    "post-training, and with both (the final model). A model that leaves out a phase trains on the joint-finetuning "
    "loss for those updates instead. All four are evaluated on the same 64 clips from validation participants, one "
    "sample per clip and task with the clip's fixed seed; intervals are 95% clip-bootstrap intervals. Bold marks the "
    "best value in each column.</div>") + '<div class="charts">' + "".join([
    table("Rotation error (&deg;)",
          "Relative to the first frame (lower is better): the joint video's camera against the joint head "
          "(self-consistency); the generated head against the recorded head given the real video; the generated "
          "video's camera against the recorded head given the real motion. VGGT-&Omega;'s camera from the "
          "recorded video lies 3.8&deg; from the recorded head, the lowest error the camera-based columns can reach.",
          ["Self-consistency", "Video-conditioned", "Motion-conditioned"],
          [["21.5", "29.7", "15.8"], ["19.5", "31.4", "14.8"], ["18.6", "<b>27.6</b>", "<b>14.1</b>"],
           ["<b>16.7</b>", "29.6", "<b>14.1</b>"]]),
    table("Fr&eacute;chet distances",
          "Lower is better. Video FID on Inception-v3 features of the generated frames after the first, against the "
          "same clips' recordings through the VAE. Motion FD directly on the standardized motion features (pose) and "
          "their frame-to-frame differences (velocity), against the recorded motion; it uses no learned motion "
          "encoder, unlike an M-FID.",
          ["Joint", "Motion-cond.", "Joint", "Video-cond.", "Joint", "Video-cond."],
          [["71.7", "65.8", "42.6", "30.9", "0.60", "0.62"], ["63.5", "59.5", "34.1", "32.1", "0.64", "0.60"],
           ["<b>63.4</b>", "57.7", "<b>23.8</b>", "21.3", "0.42", "0.38"],
           ["63.8", "<b>55.6</b>", "28.2", "<b>21.0</b>", "<b>0.37</b>", "<b>0.37</b>"]],
          groups=[("Video FID", 2), ("Motion FD, pose", 2), ("Motion FD, velocity", 2)]),
    table("HandScore",
          "F-score, averaged over the two hands, between the wearer's hands Qwen3-VL-32B finds in each frame and the "
          "hands the motion places in view of the headset's camera (higher is better). Recorded pairs score 0.83 "
          "[0.77, 0.88]; pairing a video with another clip's motion scores about 0.43.",
          ["Joint", "Motion-cond. video", "Video-cond. motion"],
          [["0.65 [0.58, 0.71]", "0.72 [0.64, 0.78]", "0.72 [0.63, 0.79]"],
           ["<b>0.73 [0.64, 0.80]</b>", "<b>0.76 [0.67, 0.84]</b>", "0.73 [0.65, 0.80]"],
           ["0.69 [0.60, 0.77]", "<b>0.76 [0.67, 0.82]</b>", "0.72 [0.64, 0.80]"],
           ["0.69 [0.60, 0.76]", "0.69 [0.59, 0.77]", "<b>0.74 [0.66, 0.81]</b>"]]),
]) + "</div></section>"

a = s.index('<section class="tab" id="tab-results">')
b = s.index("</section>", a) + len("</section>")
s = s[:a] + RESULTS + s[b:]
s = s.replace("</style></head>", ".results { border-collapse: collapse; width: 100%; font-size: 13px; }\n"
              ".results th, .results td { padding: 4px 8px; text-align: center; border-bottom: 1px solid #e5e5e5; }\n"
              ".results th { font-weight: 600; color: #333; }\n.byline { color: #555; margin: 2px 0 10px; }\n"
              "footer { color: #666; font-size: 12.5px; padding: 0 22px 30px; }\n" + SITENAV_CSS + "</style></head>", 1)
for old, new in (('<input type="radio" name="layout" value="overlay" checked>', '<input type="radio" name="layout" value="overlay">'),
                 ('<input type="radio" name="layout" value="side">', '<input type="radio" name="layout" value="side" checked>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)   # side by side is the default 3D layout
FOOTER = ("<footer>Data: NymeriaPlus, built on the Nymeria dataset of Project Aria recordings (Meta Reality Labs "
          "Research). Video model: HunyuanVideo&nbsp;1.5. Camera estimates: VGGT-&Omega;. 3D viewer: viser.</footer>")
s = s.replace("<script>", FOOTER + "\n<script>", 1)
s = s.replace("<header>", sitenav("gallery") + "\n<header>", 1)

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
s = s.replace(old, old + '\naddEventListener("hashchange", () => showTab(location.hash.slice(1) || "training"));')
(samples / "index.html").write_text(s)
(samples / "README.md").write_text(
    "# More samples: Aligning Egocentric Video and Human Motion Generation\n\n"
    f"The 64-clip gallery of the thesis's results site ({FIGURES_URL}), served at {SAMPLES_URL}.\n"
    "Built by `build_site.py` in the thesis figures site's repo (AntonioMacaronio/aligning-video-motion); do not edit "
    "the files here by hand.\n")

# ---------------------------------------------------------------- the thesis figures site
(site / "gallery.html").write_text(
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>More samples</title>'
    f'<meta http-equiv="refresh" content="0; url={SAMPLES_URL}">'
    f'<script>location.replace("{SAMPLES_URL}" + location.hash);</script></head>'
    f'<body><p>The samples moved to <a href="{SAMPLES_URL}">{SAMPLES_URL}</a>.</p></body></html>\n')

TASKS = {
    "joint": dict(label="Joint generation", note="The video and the motion are both generated from the caption and the "
                  "first video frame.", video="Generated video", motion="Generated motion"),
    "motion_cond": dict(label="Motion-conditioned video", note="The video is generated from the caption, the first frame "
                        "and the recorded motion shown.", video="Generated video", motion="Input motion (recorded)"),
    "video_cond": dict(label="Video-conditioned motion", note="The motion is generated from the caption and the recorded "
                       "video shown.", video="Input video (recorded)", motion="Generated motion"),
}
SECTIONS = (("joint", "Figures 4.1 and 4.2"), ("motion_cond", "Figures 4.3 and 4.4"), ("video_cond", "Figures 4.5 and 4.6"))
examples = json.loads((site / "thesis_examples.json").read_text())


def pane(e, task, active):
    """One task of an example: its video beside its 3D scene (the figure's task under the example's own name)."""
    name = e["name"] if task == e["mode"] else f"{e['name']}__{task}"
    info = json.loads((site / "recordings" / "thesis" / f"{name}.json").read_text())
    camera = (f"initialCameraPosition={','.join(map(str, info['position']))}"
              f"&amp;initialCameraLookAt={','.join(map(str, info['look_at']))}&amp;initialCameraUp=0,0,1")
    t = TASKS[task]
    return (f'<div class="pane" data-task="{task}"{"" if active else " hidden"}><p class="task-note">{t["note"]}</p>'
            f'<div class="body"><div class="video"><p class="panel-label">{t["video"]}</p>'
            f'<video src="media/thesis/{name}.mp4" muted loop playsinline preload="none"></video></div>'
            f'<div class="motion"><p class="panel-label">{t["motion"]}</p>'
            f'<div class="scene" style="aspect-ratio: {info["aspect"]:.3f}">'
            f'<div class="hint">the 3D scene loads when the example is fully on screen</div>'
            f'<iframe data-src="viser/index.html?playbackPath=../recordings/thesis/{name}.viser&amp;{camera}" '
            f'title="{html.escape(e["title"])}: {t["label"].lower()}"></iframe></div></div></div></div>')


cards = []
for mode, figures in SECTIONS:
    cards.append(f'<h2>{TASKS[mode]["label"]} <span class="figref">thesis {figures}</span></h2>')
    for e in (e for e in examples if e["mode"] == mode):
        tabs = "".join(
            f'<button role="tab" data-task="{task}" aria-selected="{"true" if task == mode else "false"}">{t["label"]}'
            f'{" <span class=fig-mark>thesis figure</span>" if task == mode else ""}</button>' for task, t in TASKS.items())
        cards.append(f'<div class="example"><h3>{html.escape(e["title"])}</h3>'
                     f'<p class="caption">{html.escape(e["caption"])}</p><div class="tabs" role="tablist">{tabs}</div>'
                     + "".join(pane(e, task, task == mode) for task in TASKS) + "</div>")
MAIN = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Aligning Egocentric Video and Human Motion Generation</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; margin: 0; color: #222; background: #f6f7f9; }}
header {{ padding: 18px 28px 8px; background: #fff; border-bottom: 1px solid #ddd; }}
h1 {{ font-size: 20px; margin: 0 0 6px; }}
.byline {{ color: #555; margin: 2px 0 10px; }}
.notes {{ font-size: 13px; line-height: 1.5; color: #444; max-width: 1400px; }}
{SITENAV_CSS}.sitenav {{ position: sticky; top: 0; z-index: 10; }}
main {{ padding: 4px 22px 30px; max-width: 1500px; }}
h2 {{ font-size: 17px; margin: 22px 0 8px; }}
.figref {{ font-size: 12.5px; color: #777; font-weight: 400; margin-left: 6px; }}
.example {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 0 0 16px; padding: 10px 12px; }}
.example h3 {{ font-size: 14.5px; margin: 0 0 2px; }}
.example .caption {{ font-size: 12.5px; color: #555; margin: 0 0 6px; }}
.tabs {{ display: flex; flex-wrap: wrap; gap: 2px; border-bottom: 1px solid #e3e3e3; margin: 0 0 8px; }}
.tabs button {{ font: inherit; font-size: 13px; padding: 6px 12px; border: 0; border-bottom: 2px solid transparent; margin-bottom: -1px; background: none; color: #3182bd; cursor: pointer; }}
.tabs button[aria-selected="true"] {{ color: #222; font-weight: 600; border-bottom-color: #3182bd; }}
.fig-mark {{ font-size: 11px; color: #666; background: #eef0f3; border-radius: 3px; padding: 0 5px; margin-left: 4px; font-weight: 400; }}
.task-note {{ font-size: 12.5px; color: #555; margin: 0 0 8px; }}
.pane .body {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2.25fr); gap: 14px; align-items: start; }}
.panel-label {{ font-size: 12px; color: #444; margin: 0 0 4px; font-weight: 600; }}
.video video {{ width: 100%; display: block; box-sizing: border-box; border: 6px solid rgb(46,184,114); border-radius: 6px; background: #000; }}
.scene {{ position: relative; background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; overflow: hidden; }}
.scene iframe {{ position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }}
.scene .hint {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #889; font-size: 13px; pointer-events: none; }}
.ramp {{ display: inline-block; width: 110px; height: 10px; border-radius: 2px; vertical-align: -1px; background: linear-gradient(90deg, rgb(46,184,114), rgb(52,136,219), rgb(142,68,173)); }}
footer {{ color: #666; font-size: 12.5px; padding: 0 22px 30px; }}
@media (max-width: 900px) {{ .pane .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
{sitenav("figures")}
<header>
<h1>Aligning Egocentric Video and Human Motion Generation</h1>
<p class="byline">Anthony Zhang &middot; M.S. thesis, Electrical Engineering and Computer Sciences, UC Berkeley, 2026</p>
<div class="notes">
<p>One model generates egocentric video and the wearer's full-body motion together, or either one given the other. It extends
the HunyuanVideo&nbsp;1.5 video diffusion transformer with a motion stream and is trained with joint finetuning, a text curriculum
and geometric post-training. These are the examples of the thesis's Section 4.4, sampled from the final model. Each example
opens on the task of its thesis figure; its tabs show the evaluation's samples of the same clip for the other two tasks.</p>
<p>Each example shows the video on the left and the motion on the right: an MHR body with the subject's body shape, animated
over the clip inside a VGGT-&Omega; point cloud of the video. As in the thesis figures, the body's color moves from green through
blue to purple over the clip <span class="ramp"></span>, and the video's border follows the same colors. The point cloud is placed
with the motion's own camera trajectory, so the body and the scene agree only as far as the video and the motion do, and the
body's head follows the orientation of the Aria glasses. Each scene opens at its figure's camera; drag to orbit. A scene loads,
and its video plays, when the example is fully on screen; the video follows the scene's play bar, so pausing or scrubbing a
scene moves its video with it.</p>
<p><a href="{SAMPLES_URL}">More samples</a> has the evaluation's 64 clips with every task, the recorded motion and
VGGT-&Omega;'s cameras; <a href="{SAMPLES_URL}#results">Results</a> has the thesis's quantitative comparison of the training
phases.</p>
</div>
</header>
<main>
{"".join(cards)}
</main>
{FOOTER}
<script>
{SYNC_JS}const RAMP = [[46, 184, 114], [52, 136, 219], [142, 68, 173]];
function ramp(f) {{
  const x = Math.min(Math.max(f, 0), 1) * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), t = x - i;
  return "rgb(" + RAMP[i].map((c, k) => Math.round(c + t * (RAMP[i + 1][k] - c))).join(",") + ")";
}}
const MAX_LIVE = 3, live = [];
function unload(frame) {{
  const i = live.indexOf(frame);
  if (i >= 0) live.splice(i, 1);
  if (frame.dataset.loaded === "1") {{ frame.src = "about:blank"; frame.dataset.loaded = "0"; }}
  frame.previousElementSibling.style.display = "";
}}
function load(frame) {{
  if (frame.dataset.loaded === "1") {{ live.splice(live.indexOf(frame), 1); live.push(frame); return; }}
  frame.src = frame.dataset.src; frame.dataset.loaded = "1"; frame.previousElementSibling.style.display = "none";
  live.push(frame);
  while (live.length > MAX_LIVE) unload(live[0]);
}}
// One entry per task pane; only the selected pane of an example is displayed, so only it can become visible.
const panes = [...document.querySelectorAll(".pane")].map((pane) => ({{
  pane, body: pane.querySelector(".body"), video: pane.querySelector("video"), frame: pane.querySelector("iframe"),
  shown: -1, on: false,
}}));
function setOn(p, on) {{
  if (p.on === on) return;
  p.on = on;
  p.shown = -1;
  if (on) {{ load(p.frame); startVideo(p.video); }}
  else {{ unload(p.frame); p.video.pause(); }}
}}
const observer = new IntersectionObserver((entries) => entries.forEach((e) => {{
  const p = panes.find((x) => x.body === e.target);
  if (p) setOn(p, !p.pane.hidden && fullyVisible(e));
}}), {{ threshold: THRESHOLDS }});
panes.forEach((p) => observer.observe(p.body));
document.querySelectorAll(".example").forEach((example) => {{
  example.querySelectorAll('.tabs [role="tab"]').forEach((tab) => tab.addEventListener("click", () => {{
    example.querySelectorAll('.tabs [role="tab"]').forEach((b) => b.setAttribute("aria-selected", String(b === tab)));
    for (const p of panes) {{
      if (!example.contains(p.pane)) continue;
      const selected = p.pane.dataset.task === tab.dataset.task;
      if (!selected) setOn(p, false);
      p.pane.hidden = !selected;
    }}
  }}));
}});
function tick() {{
  for (const p of panes) {{
    if (!p.on) continue;
    const v = p.video, t = sceneTime(p.frame);
    if (t !== null && v.readyState >= 1) {{
      if (!v.paused) v.pause();
      const k = sceneFrame(t);
      if (k !== p.shown) {{ v.currentTime = (k + 0.5) / VIDEO_FPS; p.shown = k; }}
      v.style.borderColor = ramp(k / LAST_FRAME);  // the body's color at motion frame 3k
    }} else {{
      if (v.readyState === 0 && v.networkState !== HTMLMediaElement.NETWORK_LOADING) startVideo(v);
      if (v.duration) v.style.borderColor = ramp(v.currentTime / v.duration);  // the scene is loading: the video plays on its own
    }}
  }}
  requestAnimationFrame(tick);
}}
requestAnimationFrame(tick);
</script>
</body></html>
"""
(site / "index.html").write_text(MAIN)

for page, text in (("samples index.html", s), ("site index.html", MAIN)):
    leftover = [p for p in (r"\bLVP\b", r"c1f6727|480f69b", r"\bB200\b", r"Run [A-Z]\b", r"\bv2b\b",
                            r"A-cont|I-cc|I-cont", r"docs/", r"vis/", r"launch-records", r"s3://", r"grid ?2",
                            r"Hunyuan viewer")
                if re.search(p, re.sub(r"<script.*?</script>", "", text, flags=re.S))]
    print(page, len(text), "bytes; leftover internal markers:", leftover)
