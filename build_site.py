"""Build the thesis's results site from the K v2b 12.5k gallery (GitHub Pages, branch gh-pages).

    python3 build_site.py <gallery dir> <site repo dir>

Copies the gallery's static viser client and viser recordings, re-encodes its videos, and writes two pages:
- index.html, the main page: the thesis's Section 4.4 examples as animated scenes (make_thesis_scenes.py writes their
  recordings/thesis/ and media/thesis/; thesis_examples.json has their titles and captions);
- gallery.html, the gallery's page rewritten for an outside reader: a header in the thesis's terms (no internal paths,
  commit hashes or run codenames), tab labels, and a Results tab with the thesis's Chapter 4 tables (64 held-out clips
  at 12,500 updates) in place of the internal evaluation charts.
The gallery's videos are re-encoded once (H.264 CRF 26, same size and frame rate, media/.encoded.json) so the site
stays under GitHub Pages' 1 GB.
"""
import html
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
subprocess.run(["rsync", "-a", "--delete", f"{src / 'viser'}/", f"{dst / 'viser'}/"], check=True)
subprocess.run(["rsync", "-a", "--delete", "--exclude", "thesis/", f"{src / 'recordings'}/", f"{dst / 'recordings'}/"],
               check=True)
(dst / ".nojekyll").write_text("")

manifest_path = dst / "media" / ".encoded.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
sources = {p.relative_to(src / "media").as_posix(): p for p in (src / "media").rglob("*") if p.is_file()}


def encode(rel):
    source, out = sources[rel], dst / "media" / rel
    stamp = [source.stat().st_size, int(source.stat().st_mtime)]
    if manifest.get(rel) == stamp and out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".mp4":
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-c:v", "libx264", "-crf", "26", "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)], check=True)
    else:
        shutil.copy2(source, out)
    manifest[rel] = stamp


with ThreadPoolExecutor(8) as pool:
    list(pool.map(encode, sorted(sources)))
for out in (dst / "media").rglob("*"):  # gallery media the gallery no longer has (the thesis page's are kept)
    rel = out.relative_to(dst / "media").as_posix()
    if out.is_file() and not rel.startswith("thesis/") and out != manifest_path and rel not in sources:
        out.unlink()
manifest_path.write_text(json.dumps({k: manifest[k] for k in sorted(manifest) if k in sources}, indent=0))

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
Drag to orbit; each scene has its own play controls, and the toggles below apply to every scene.</p>
<p><b>Training-split clips</b>: 16 clips from the training split. <b>Held-out clips</b>: 48 clips from validation participants
who never appear in training. <b>Results</b>: the thesis's quantitative comparison of the training phases.</p>
</div>
</header>"""

a = s.index("<header>")
b = s.index("</header>") + len("</header>")
s = s[:a] + HEADER + s[b:]
s = re.sub(r"<title>.*?</title>", "<title>Aligning Egocentric Video and Human Motion Generation</title>", s, count=1, flags=re.S)

for old, new in (('<button data-tab="training">Training (16/16)</button>', '<button data-tab="training">Training-split clips (16)</button>'),
                 ('<button data-tab="validation">Validation (48/48)</button>', '<button data-tab="validation">Held-out clips (48)</button>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)
s = s.replace(" &middot; diagnostic clip</h3>", " &middot; training-split clip</h3>")
s = s.replace("<figcaption>GT video</figcaption>", "<figcaption>recorded video</figcaption>")
s = s.replace('<i class="chip" style="background:#c8c8c8"></i>GT</label>', '<i class="chip" style="background:#c8c8c8"></i>recorded</label>')
s = s.replace('></i>GT video</label>', '></i>recorded video</label>')

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
              "footer { color: #666; font-size: 12.5px; padding: 0 22px 30px; }\n</style></head>", 1)
for old, new in (('<input type="radio" name="layout" value="overlay" checked>', '<input type="radio" name="layout" value="overlay">'),
                 ('<input type="radio" name="layout" value="side">', '<input type="radio" name="layout" value="side" checked>')):
    assert s.count(old) == 1, old
    s = s.replace(old, new)   # side by side is the default 3D layout
FOOTER = ("<footer>Data: NymeriaPlus, built on the Nymeria dataset of Project Aria recordings (Meta Reality Labs "
          "Research). Video model: HunyuanVideo&nbsp;1.5. Camera estimates: VGGT-&Omega;. 3D viewer: viser.</footer>")
s = s.replace("<script>", FOOTER + "\n<script>", 1)

# Links between the two pages, above the gallery's own tabs; #results opens the Results tab, also from the main page.
SITENAV_CSS = (".sitenav { background: #fff; border-bottom: 1px solid #ddd; padding: 0 22px; display: flex; gap: 2px; }\n"
               ".sitenav a { font-size: 14px; padding: 9px 14px; color: #3182bd; text-decoration: none; }\n"
               ".sitenav a.active { color: #222; font-weight: 600; }\n")


def sitenav(active):
    links = (("index.html", "Thesis figures", "figures"), ("gallery.html", "All samples", "gallery"),
             ("gallery.html#results", "Results", "results"))
    marked = ' class="active"'
    return '<div class="sitenav">' + "".join(
        f'<a href="{href}"{marked if key == active else ""}>{label}</a>' for href, label, key in links) + "</div>"


s = s.replace("</style></head>", SITENAV_CSS + "</style></head>", 1)
s = s.replace("<header>", sitenav("gallery") + "\n<header>", 1)
old = 'showTab(location.hash.slice(1) || "training");'
assert s.count(old) == 1
s = s.replace(old, old + '\naddEventListener("hashchange", () => showTab(location.hash.slice(1) || "training"));')
(dst / "gallery.html").write_text(s)

# The main page: the thesis's Section 4.4 examples.
TASKS = (("joint", "Joint generation", "The video and the motion are both generated from the caption and the first video "
          "frame.", "Generated video", "Generated motion"),
         ("motion_cond", "Motion-conditioned video", "The video is generated from the caption, the first frame and the "
          "recorded motion shown.", "Generated video", "Input motion (recorded)"),
         ("video_cond", "Video-conditioned motion", "The motion is generated from the caption and the recorded video shown.",
          "Input video (recorded)", "Generated motion"))
examples = json.loads((dst / "thesis_examples.json").read_text())
cards = []
for mode, heading, note, video_label, motion_label in TASKS:
    cards.append(f'<h2>{heading}</h2><p class="task-note">{note}</p>')
    for e in (e for e in examples if e["mode"] == mode):
        info = json.loads((dst / "recordings" / "thesis" / f"{e['name']}.json").read_text())
        camera = (f"initialCameraPosition={','.join(map(str, info['position']))}"
                  f"&amp;initialCameraLookAt={','.join(map(str, info['look_at']))}&amp;initialCameraUp=0,0,1")
        tag = f' <span class="tag">{e["tag"]}</span>' if e["tag"] else ""
        cards.append(
            f'<div class="example"><h3>{html.escape(e["title"])}{tag}</h3>'
            f'<p class="caption">{html.escape(e["caption"])}</p><div class="body">'
            f'<div class="video"><p class="panel-label">{video_label}</p>'
            f'<video src="media/thesis/{e["name"]}.mp4" muted loop playsinline preload="metadata"></video>'
            f'<button class="replay">restart video</button></div>'
            f'<div class="motion"><p class="panel-label">{motion_label}</p>'
            f'<div class="scene" style="aspect-ratio: {info["aspect"]:.3f}"><div class="hint">loading the 3D scene&hellip;</div>'
            f'<iframe data-src="viser/index.html?playbackPath=../recordings/thesis/{e["name"]}.viser&amp;{camera}" '
            f'title="{html.escape(e["title"])}"></iframe></div></div></div></div>')
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
h2 {{ font-size: 17px; margin: 22px 0 2px; }}
.task-note {{ font-size: 13px; color: #555; margin: 0 0 10px; }}
.example {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; margin: 0 0 16px; padding: 10px 12px; }}
.example h3 {{ font-size: 14.5px; margin: 0 0 2px; }}
.tag {{ font-size: 11.5px; color: #666; background: #eef0f3; border-radius: 3px; padding: 1px 6px; margin-left: 6px; font-weight: 400; }}
.example .caption {{ font-size: 12.5px; color: #555; margin: 0 0 8px; }}
.example .body {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2.25fr); gap: 14px; align-items: start; }}
.panel-label {{ font-size: 12px; color: #444; margin: 0 0 4px; font-weight: 600; }}
.video video {{ width: 100%; display: block; box-sizing: border-box; border: 6px solid rgb(46,184,114); border-radius: 6px; background: #000; }}
.replay {{ font-size: 12px; margin-top: 6px; padding: 3px 10px; cursor: pointer; }}
.scene {{ position: relative; background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; overflow: hidden; }}
.scene iframe {{ position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }}
.scene .hint {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #889; font-size: 13px; pointer-events: none; }}
.ramp {{ display: inline-block; width: 110px; height: 10px; border-radius: 2px; vertical-align: -1px; background: linear-gradient(90deg, rgb(46,184,114), rgb(52,136,219), rgb(142,68,173)); }}
footer {{ color: #666; font-size: 12.5px; padding: 0 22px 30px; }}
@media (max-width: 900px) {{ .example .body {{ grid-template-columns: 1fr; }} }}
</style></head><body>
{sitenav("figures")}
<header>
<h1>Aligning Egocentric Video and Human Motion Generation</h1>
<p class="byline">Anthony Zhang &middot; M.S. thesis, Electrical Engineering and Computer Sciences, UC Berkeley, 2026</p>
<div class="notes">
<p>One model generates egocentric video and the wearer's full-body motion together, or either one given the other. It extends
the HunyuanVideo&nbsp;1.5 video diffusion transformer with a motion stream and is trained with joint finetuning, a text curriculum
and geometric post-training. These are the examples of the thesis's Section 4.4, sampled from the final model.</p>
<p>Each example shows the video on the left and the motion on the right: an MHR body with the subject's body shape, animated
over the clip inside a VGGT-&Omega; point cloud of the video. As in the thesis figures, the body's color moves from green through
blue to purple over the clip <span class="ramp"></span>, and the video's border follows the same colors. The point cloud is placed
with the motion's own camera trajectory, so the body and the scene agree only as far as the video and the motion do, and the
body's head follows the orientation of the Aria glasses. Each scene opens at its figure's camera; drag to orbit.</p>
<p><a href="gallery.html">All samples</a> has the evaluation's 64 clips with every task, the recorded motion and VGGT-&Omega;'s
cameras; <a href="gallery.html#results">Results</a> has the thesis's quantitative comparison of the training phases.</p>
</div>
</header>
<main>
{"".join(cards)}
</main>
{FOOTER}
<script>
const RAMP = [[46, 184, 114], [52, 136, 219], [142, 68, 173]];
function ramp(f) {{
  const x = Math.min(Math.max(f, 0), 1) * (RAMP.length - 1), i = Math.min(Math.floor(x), RAMP.length - 2), t = x - i;
  return "rgb(" + RAMP[i].map((c, k) => Math.round(c + t * (RAMP[i + 1][k] - c))).join(",") + ")";
}}
const videos = [...document.querySelectorAll("video")];
function tick() {{
  for (const v of videos) if (v.duration) v.style.borderColor = ramp(v.currentTime / v.duration);
  requestAnimationFrame(tick);
}}
requestAnimationFrame(tick);
const seen = new IntersectionObserver((entries) => entries.forEach((e) => {{
  if (e.isIntersecting) e.target.play().catch(() => {{}}); else e.target.pause();
}}), {{ threshold: 0.25 }});
videos.forEach((v) => seen.observe(v));
document.querySelectorAll(".replay").forEach((b) => b.addEventListener("click", () => {{
  const v = b.parentElement.querySelector("video"); v.currentTime = 0; v.play();
}}));
const MAX_LIVE = 4, live = [];
function unload(frame) {{
  frame.src = "about:blank"; delete frame.dataset.loaded; frame.previousElementSibling.style.display = "";
  live.splice(live.indexOf(frame), 1);
}}
function load(frame) {{
  if (frame.dataset.loaded) {{ live.splice(live.indexOf(frame), 1); live.push(frame); return; }}
  frame.src = frame.dataset.src; frame.dataset.loaded = "1"; frame.previousElementSibling.style.display = "none";
  live.push(frame);
  while (live.length > MAX_LIVE) unload(live[0]);
}}
const near = new IntersectionObserver((entries) => entries.forEach((e) => {{ if (e.isIntersecting) load(e.target); }}),
                                      {{ rootMargin: "250px 0px" }});
document.querySelectorAll("iframe[data-src]").forEach((f) => near.observe(f));
</script>
</body></html>
"""
(dst / "index.html").write_text(MAIN)

for page, text in (("gallery.html", s), ("index.html", MAIN)):
    leftover = [p for p in (r"\bLVP\b", r"c1f6727|480f69b", r"\bB200\b", r"Run [A-Z]\b", r"\bv2b\b",
                            r"A-cont|I-cc|I-cont", r"docs/", r"vis/", r"launch-records", r"s3://", r"grid ?2",
                            r"Hunyuan viewer")
                if re.search(p, re.sub(r"<script.*?</script>", "", text, flags=re.S))]
    print(page, len(text), "bytes; leftover internal markers:", leftover)
