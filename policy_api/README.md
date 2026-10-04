# policy_api

Serves trained skibot policies so any physics loop can run them without training: one robot, zero-shot.

- `export_policies.py` turns checkpoints from `runs_live/` into `policies/<id>/`: `weights.npz` (normalizer + MLP), `policy.onnx` (same network for local runtime) and `meta.json` (architecture, every observation field with its slice and scale, action layout, control rate, provenance).
- `hi_server.py` is the FastAPI server: `GET /v1/policies`, `GET /v1/policies/{id}`, `/onnx`, `/weights`, `POST /v1/policies/{id}/infer`, and a websocket at `/v1/policies/{id}/ws` that speaks openpi's protocol (metadata on connect, msgpack-numpy observations in, actions out), so `openpi_client.WebsocketClientPolicy` works against it.
- `mission.py` + `/v1/missions`: send the robot to one point on a route. `POST /v1/missions` takes a route polyline (local metres) and a target (`target_s` metres along it, or `target_xy`). Each `POST /v1/missions/{id}/step` sends the observation and pose `{x, y, yaw}`; the server writes the command slice itself (pure pursuit 6 m ahead, slowing over the last 20 m, stop inside `arrive_radius`) and returns actions plus progress. `GET /v1/missions/{id}?log=1` is the server's record of every step and the arrival time. Missions live in memory, so Modal runs one container.
- `hi_client.py`: `pull(host, "g1-ski")` downloads the spec and ONNX once and runs locally (about 0.05 ms per step); `HiPolicy` streams over the websocket.
- `modal_app.py` hosts it on Modal (CPU, scales to zero): https://ryanrana04--hazard-intelligence-api-api.modal.run
- Keys and usage live in the team's SpacetimeDB (`api_key`, `api_call`; see DATABASE.md). Pass a key as `Authorization: Bearer hi_...`, or `pull(host, "g1-ski", key=...)` / `HI_API_KEY`. Public policies need none; a wrong or revoked key gets 401. The Modal secret `hi-stdb` holds the API's SpacetimeDB token.
- `test_api.py` checks numpy, HTTP, websocket and pulled ONNX against rsl_rl's own actions on a recorded 300-step rollout (`parity.npz`, from `record_parity.py`); all agree within 4e-6. `example_sim.py` drives the CPU SkiEnv with a pulled policy.

| id | inputs → outputs | benchmark |
|---|---|---|
| g1-ski | 141 → 31 | 19 of 24 test slopes without a fall (SONIC alone: 1) |
| g1-ski-trees | 156 → 31 | adds the 15 tree rays; not scored yet |

Both steer NVIDIA GEAR-SONIC (NVIDIA Open Model License), which is not served here.
