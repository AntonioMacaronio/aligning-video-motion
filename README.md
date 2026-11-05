# Aligning Egocentric Video and Human Motion Generation: project page

Static project page for Anthony Zhang's M.S. thesis (UC Berkeley EECS, 2026), served by GitHub Pages from this repo's
`gh-pages` branch at https://antoniomacaronio.github.io/aligning-video-motion/. The gallery of all 64 evaluation clips is
a separate repo, `aligning-video-motion-samples` (GitHub Pages allows 1 GB per site).

- `post_template.html`: the page's source (prose, equations, tables). `build_site.py` fills its slots with the twelve
  interactive thesis examples and the playback script, and writes `index.html`; edit the template, not `index.html`.
- `figures/`: the thesis figures the page shows, written by `make_post_figures.py` (macOS): JPEGs for the figures with
  photos and renders, SVGs for the line drawings. The geometry sidecar's SVG carries the animation from
  `sidecar_animation.py`; `arch_sidecar_static.svg` is its still version for readers who ask for reduced motion.
- `explainer/`: the interactive explainer at the top of "Geometric post-training" (`#gx` in the template). `geometry.js`
  draws the whole story as a function of time on one stage and runs its controls; the story's lines, their ids and
  their lengths in seconds are the `.gx-lede` elements in the template; `geometry.css` styles it. `frames.jpg` and
  `noise*.png` come from `make_explainer_assets.py` (the staircase example's video, and pixel noise).
- `recordings/thesis/`, `media/thesis/`: the examples' viser recordings and videos, from `make_thesis_scenes.py`.
- `favicon.ico`, `favicon-32.png`, `apple-touch-icon.png`: the tab icon, a body mesh through a fisheye lens framed like an egocentric frame, from `make_favicon.py` (the gallery links to the same files).
- `viser/`: the static viser client; each scene loads `viser/index.html?playbackPath=../recordings/thesis/<name>.viser`.
- `gallery.html`: a redirect for old links to the gallery.

Rebuild: `python3 build_site.py <gallery dir> <this repo> <samples repo>`. Preview: serve the parent folder of both repos
with a server that supports HTTP Range requests (video seeking needs them), then open `/aligning-video-motion/`.
