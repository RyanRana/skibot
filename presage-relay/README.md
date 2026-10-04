# Presage relay

Heart rate and breathing for the live Hazard Intelligence game. Presage has no in-browser SDK and Vercel cannot host its native SDK or a long-lived WebSocket, so this small server does it: the game streams JPEG camera frames over `wss://…/presage`, the server runs the Presage SmartSpectra Node SDK and streams pulse and breathing back. The Presage key lives only in this server's environment.

| Variable | |
|---|---|
| `PRESAGE_API_KEY` | required, secret |
| `ALLOWED_ORIGINS` | the site's address(es), comma separated, e.g. `https://hazard.vercel.app`; without it only localhost may connect |
| `MAX_SESSION_SECONDS` | default 300 |
| `SESSIONS_PER_HOUR` | per visitor address, default 8 |
| `PORT` | default 8787 |

One player is measured at a time (the SDK's state is per process); others see "pulse busy". `GET /health` reports `{ ok, keyed, busy }`.

**On a host** (Render, Fly, Railway, AWS): deploy this folder with its `Dockerfile` (x86: `docker build --platform linux/amd64`), set the variables, and put `wss://<host>/presage` in the Vercel project as `VITE_PRESAGE_URL`, then redeploy the site.

**On a laptop** for a demo: `scripts/presage-relay.sh https://<site>` runs it (Docker or Node) and opens a free Cloudflare tunnel; put the printed address in Vercel as above. The laptop has to stay on while people play.

Without `VITE_PRESAGE_URL` the live game simply runs without vitals; locally the dev server measures (see GAME.md).
