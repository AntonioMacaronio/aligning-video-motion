# Aligning Egocentric Video and Human Motion Generation: project page

Static project page for Anthony Zhang's M.S. thesis (UC Berkeley EECS, 2026), served by GitHub Pages from this repo's
`gh-pages` branch at https://antoniomacaronio.github.io/aligning-video-motion/. The gallery of all 64 evaluation clips is
a separate repo, `aligning-video-motion-samples` (GitHub Pages allows 1 GB per site).

- `post_template.html`: the page's source (prose, equations, tables). `build_site.py` fills its slots with the teaser
  video and the page's scripts (its video player among them), and writes `index.html`; edit the template, not
  `index.html`.
- `media/teaser.mp4`: the teaser under the title, from `make_teaser.py`: the three tasks, one thesis figure each, as
  inputs -> MM-DiT -> outputs, drawn at 2880 x 1440.
- `figures/`: the thesis figures the page shows, written by `make_post_figures.py` (macOS): JPEGs for the figures with
  photos and renders, SVGs for the line drawings. The geometry sidecar's SVG carries the animation from
  `sidecar_animation.py`; `arch_sidecar_static.svg` is its still version for readers who ask for reduced motion.
- `explainer/`: the interactive explainer at the top of "Geometric post-training" (`#gx` in the template). `geometry.js`
  draws the whole story as a function of time on one stage and runs its controls; the story's lines, their ids and
  their lengths in seconds are the `.gx-lede` elements in the template; `geometry.css` styles it. `frames.jpg` and
  `noise*.png` come from `make_explainer_assets.py` (the staircase example's video, and pixel noise).
- `media/thesis/`: the examples' videos, from `make_thesis_scenes.py`, and their recorded motion as clips
  (`<example>__input_motion.mp4`), from `make_task_inputs.py`.
- `media/misalignment/`: the Results section's camera-head comparison (joint finetuning alone against the final
  model on four held-out clips, `comparison.mp4` the four in turn), from `make_misalignment_videos.py`.
- `media/scenes/`: square clips of two examples' 3D scenes, from `make_scene_clips.py`, shown by the earlier
  three-column viewer (no longer on the page).
- `recordings/thesis/`: the examples' viser recordings, from `make_thesis_scenes.py` (the page no longer shows them).
- `favicon.ico`, `favicon-32.png`, `apple-touch-icon.png`: the tab icon, a body mesh through a fisheye lens framed like an egocentric frame, from `make_favicon.py` (the gallery links to the same files).
- `viser/`: the static viser client; each scene loads `viser/index.html?playbackPath=../recordings/thesis/<name>.viser`.
- `gallery.html` (from `figures_template.html`): the header's Gallery link, the twelve thesis figures as cards, each
  with a tab per task (inputs -> outputs, the 3D scene live); it links on to the 64-clip evaluation gallery.

- `site_lock.py`: the password page. `build_site.py` writes every page (this repo's `index.html` and `gallery.html`,
  and the samples gallery's `index.html`) encrypted, behind a page that asks for the password and decrypts it in the
  browser. The password is kept out of the repos: `SITE_PASSWORD`, or `~/.config/aligning-video-motion/site_password`.
  Without one the build stops; `SITE_PLAIN=1` builds unlocked pages for a local check only. To take the password
  off for good (the paper's release), set `LOCKED = False` at the top of `site_lock.py`, rebuild and publish both repos.

Rebuild: `python3 build_site.py <gallery dir> <this repo> <samples repo>`. Preview: serve the parent folder of both repos
with a server that supports HTTP Range requests (video seeking needs them), then open `/aligning-video-motion/`.
