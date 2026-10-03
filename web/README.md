# web

The Hazard Intelligence site: the landing page (`index.html`), five hand-drawn diagrams of how the system works (`how.html`, rough.js + Excalidraw's Virgil font, both vendored so it works offline) and the Model API docs (`api.html`).

`bash web/build.sh` assembles everything deployable into `web/dist/`: these pages, their media, `hi_client.py` for download, and the ski game from `game/` built under `/game`. `vercel.json` at the repo root runs that build and rewrites the game's root-relative `/courses` and `/g1` fetches into `/game`.

`tools/build_figures.py` redraws `media/tracks.svg` (Marmolada GoPro descents on real contours) and `media/terrain.svg` (Bellevarde contours, race line and gates) from local data.
