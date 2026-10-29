"""Build the public results site for the thesis from the K v2b 12.5k gallery (local only; nothing is published).

    python3 build_site.py <gallery dir> <site repo dir>

Copies the gallery's static viser client, media and viser recordings, and rewrites index.html for an outside reader:
a header in the thesis's terms (no internal paths, commit hashes or run codenames), tab labels, and a Results tab with
the thesis's Chapter 4 tables (64 held-out clips at 12,500 updates) in place of the internal evaluation charts.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for name in ("viser", "media", "recordings"):
    subprocess.run(["rsync", "-a", "--delete", f"{src / name}/", f"{dst / name}/"], check=True)
(dst / ".nojekyll").write_text("")

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
    "best value in a column of the last three tables.</div>") + '<div class="charts">' + "".join([
    table("Cross-modal reliance",
          "How much closer each stream's sample lands to the recording when it is given the recorded other modality "
          "instead of generating both: the rotation error (&deg;, relative to the first frame) against the recorded "
          "head of VGGT-&Omega;'s camera on the video, and of the generated head, jointly and given the other modality. "
          "Removed is the share of the joint error that the given modality removes.",
          ["Joint", "Given", "Removed (%)", "Joint", "Given", "Removed (%)"],
          [["39.4", "15.8", "60 [53, 66]", "40.2", "29.7", "26 [20, 32]"],
           ["35.3", "14.8", "58 [48, 66]", "38.6", "31.4", "19 [14, 23]"],
           ["33.5", "14.1", "58 [51, 64]", "34.6", "27.6", "20 [13, 27]"],
           ["34.2", "14.1", "59 [50, 65]", "37.3", "29.6", "21 [14, 28]"]],
          groups=[("Video uses motion", 3), ("Motion uses video", 3)]),
    table("Rotation error (&deg;)",
          "Relative to the first frame (lower is better): the joint video's camera against the joint head "
          "(self-consistency); the generated head against the recorded head given the real video; the generated "
          "video's camera against the recorded head given the real motion.",
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
FOOTER = ("<footer>Data: NymeriaPlus, built on the Nymeria dataset of Project Aria recordings (Meta Reality Labs "
          "Research). Video model: HunyuanVideo&nbsp;1.5. Camera estimates: VGGT-&Omega;. 3D viewer: viser.</footer>")
s = s.replace("<script>", FOOTER + "\n<script>", 1)
(dst / "index.html").write_text(s)

leftover = [p for p in (r"\bLVP\b", r"c1f6727|480f69b", r"\bB200\b", r"Run [A-Z]\b", r"\bv2b\b", r"A-cont|I-cc|I-cont",
                        r"docs/", r"vis/", r"launch-records", r"s3://", r"grid ?2", r"Hunyuan viewer")
            if re.search(p, re.sub(r"<script.*?</script>", "", s, flags=re.S))]
print("leftover internal markers:", leftover)
print("index.html", len(s), "bytes")
