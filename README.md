# Aligning Egocentric Video and Human Motion Generation: results site

Static companion site for Anthony Zhang's M.S. thesis (UC Berkeley EECS, 2026). It shows the final model's samples on
16 training-split and 48 held-out clips, with embedded, interactive viser scenes and the thesis's Chapter 4 tables.

- `index.html`: the page. `viser/`: the static viser client; each scene loads `recordings/<split>/<clip>.viser`
  through `viser/index.html?playbackPath=...`. `media/`: the videos.
- `build_site.py`: rebuilds this folder from the K v2b 12.5k gallery (`motion-v2-grid2-k2b12500-20261024`).

Preview locally: `python3 -m http.server 8765`, then open http://127.0.0.1:8765/.

Not yet published. To publish on GitHub Pages once the data and publication approvals are confirmed:

    gh repo create aligning-video-motion --public --source . --push
    gh api -X POST repos/AntonioMacaronio/aligning-video-motion/pages -f 'source[branch]=main' -f 'source[path]=/'

The site will then be at https://antoniomacaronio.github.io/aligning-video-motion/ (about 1 GB, within the Pages
limit; no file is over 100 MB).
