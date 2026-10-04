# The database

Everything in Hazard Intelligence that is not code lives in one SpacetimeDB database, `ground-truth` on maincloud: the live game, the maps and terrain, every recording, the training inputs and the policy registry. The ONNX runtime and the weights stay on Modal (`policy_api/`); the registry here says what each policy is and where it runs.

The module is `spacetime/spacetimedb/src/index.ts`. `tools/stdb_load.py` loads everything on disk into it.

## What is in it

| Group | Tables | What |
|---|---|---|
| Live game | `player`, `skier`, `run`, `trace_chunk`, `feed`, `challenge`, `cheer`, `run_photo`, `dataset_stats`, `outbox` | The shared mountain. The game, the board and the iMessage agent read and write these directly. |
| Phone | `hike`, `hike_chunk`, `hike_stats` | Hikes from the Ground Truth iPhone app, written by the phone server (`ground/stdb.py`). |
| Organisations | `tenant`, `member`, `invite`, `account`, `platform_admin` | Who owns what, who belongs where. |
| Maps | `resort`, `course`, `terrain`, `terrain_chunk`, `trail_network`, `trail`, `area`, `point_set`, `point_chunk` | 26 resorts and 47 ski courses (centreline, gates, start pose, MuJoCo scene, 2 m heightfield), 6 Ann Arbor hike courses (1 m lidar, canopy height, 74,803 trees and bushes), 3,644 ground.py trails and roads with elevation profiles, the Streif far field. |
| Recordings | `recording`, `recording_chunk`, `segment` | One shape for every stream: named channels with units, flat f32 frames. YouTube GoPro follow-cam joints (link only, never the video), GoPro telemetry at 100 Hz split into descents, the hike walking clips. |
| Training | `reference_pose`, `training_set` | The World Cup reference poses the ski policy holds, and which terrains, poses, courses and clips each policy family trains on. The tile mosaics are `terrain` rows with layer `tiles`. |
| Models | `policy`, `train_run`, `evaluation` | The policy registry (runtime and endpoint on Modal), training runs, and 409 evaluations: the 24-tile benchmark, course rides and hike stretches. |
| App users | `app_user`, `app_message`, `app_invite`, `app_session` | The people using Ground Truth: phone number or email, join code, consent, opt-out, what the agent remembers, every text both ways, invite links, and each phone session. Their raw iPhone IMU, GPS and barometer streams are `recording` rows `phone/<session>/imu`, `/gps` and `/baro`. |
| Model API | `api_key`, `api_call`, `platform_service` | API keys per organisation (only their SHA-256 is stored) and a log of every API call. |

`capture`, `capture_chunk` and `capture_stats` are the first version of `recording`. They are kept exactly as they were so publishing never disconnects a client; nothing writes them.

## Organisations

Organisations (tenants) own every row outside the live game and phone tables. Hazard Intelligence (`hazard-intelligence`) is the one public tenant: our data is open. Every other organisation is private by default.

- Tenant tables are private. Nobody reads them directly; they read views.
- `public_<table>` serves the rows of public tenants to anyone, no login needed.
- `my_<table>` serves the rows of every organisation the caller belongs to.
- Roles: `owner` and `admin` manage members and invite codes; `member` and `service` read and write; `viewer` reads.
- Joining: an owner or admin calls `create_invite`, the code shows up in their `my_invite` view, and whoever calls `redeem_invite` with it joins with that role.
- Anyone can `create_tenant`; they become its owner.
- The app's people belong to `ground-truth-app`, a private organisation. Only its owners, admins and services see them, through `my_app_user`, `my_app_message`, `my_app_invite` and `my_app_session`. Phone numbers never reach a public view.
- `delete_app_user` is "delete my data": the person, their messages, invites, sessions, raw phone streams and hikes, in one transaction.

## Reading it

Reads need no account. SQL over HTTP, from a shell or a browser (CORS is open):

```bash
curl -X POST https://maincloud.spacetimedb.com/v1/database/ground-truth/sql \
  -d "SELECT id, name, length_m, drop_m FROM public_course WHERE activity = 'ski'"
curl -X POST https://maincloud.spacetimedb.com/v1/database/ground-truth/sql \
  -d "SELECT row_0, heights FROM public_terrain_chunk WHERE terrain_id = 'kitzbuhel-streif'"
```

Rows come back as positional arrays next to their schema. Timestamps are `[micros]`, identities `["0x…"]`. There is no `ORDER BY`; sort on your side. With `Authorization: Bearer <token>` the `my_*` views answer as that identity.

## Writing it

- `tools/stdb_load.py` reads `out/` and `data/` and writes every dataset through the module's reducers, as the identity the spacetime CLI is logged in as. Re-running replaces rows; it never duplicates them.

  ```bash
  python tools/stdb_load.py --server local --root ~/mhacks-ski --hike-root ~/mhacks-hike     # a local `spacetime start`
  python tools/stdb_load.py --server maincloud courses --only kitzbuhel-streif                # one course
  ```

- Reducer arguments over HTTP are camelCase at the top level and snake_case inside nested objects (`t_0_ms`, `turn_x`).
- Requests are capped at 2 MiB, so big arrays go in chunks of about 40,000 floats.
- Every writer reducer names the tenant and needs a member, service, admin or owner role there.

## The app and the API

- **Phone server** (`ground/server.py --stdb ...`): its files in `out/ground` stay its working copy. Every invite, message, consent, upload, registration and label also goes to SpacetimeDB, plus each finished session's raw streams. At startup it copies over everything it already has.
- **Agent** (`ground/agent`): every person and every message goes to SpacetimeDB the same way. Its identity is kept in `out/ground/agent_stdb.json`.
- **Joining as a service.** Both need their identity to be a service of `ground-truth-app`. Either set `GT_SERVICE_INVITE` to a code from `python tools/stdb_admin.py invite --role service`, or run `python tools/stdb_admin.py add-service <identity>` with the identity they print at startup. The phone server's identity on maincloud is already a service.
- **Model API** (`policy_api/`, on Modal):
  - Keys go in `Authorization: Bearer hi_...`, `Api-Key hi_...` (openpi_client's `api_key`) or `X-Api-Key`.
  - Without a key, the public policies still work.
  - A wrong or revoked key gets 401. A revoked key is refused within a minute.
  - Every call is logged to `api_call`, counted for the key's organisation.
  - Make a key with `python tools/stdb_admin.py key create --label "<who>"`; it is shown once. List or revoke with `key list` and `key revoke <prefix>`.
  - The API's own identity is a platform service, and its token is the Modal secret `hi-stdb`.

## Publishing safely

- Publish from the repo root with explicit confirmations, never `--yes` alone and never `-c`:

  ```bash
  spacetime publish ground-truth --no-config -s maincloud -p spacetime/spacetimedb --yes=remote,migrate
  ```

  If SpacetimeDB says the change will break clients, the prompt appears and you can say no.
- Never run `spacetime dev` in `spacetime/`. Its config points at the live database, and `dev` wipes data on a schema conflict.
- Test every schema change on a local copy first: `spacetime start`, publish the current `main` module, then publish the new one on top. `Checking for breaking changes...` followed by `Updated database` with no prompt means it is safe.
- Do not add columns to a table the game, board or agent subscribe to. Old clients misread the rows until they are rebuilt. Add a table instead.
- Do not make a subscribed table private. The client's whole subscription fails.
- Views and reducers share one namespace, and index names come from table and column names. The local publish catches both.
- Only the owner's login can publish, see logs or read private tables. Teammates read and write data through organisations and invite codes. Publishing rights for others need the database to live in a maincloud organisation, which can only be set when a database is created.
