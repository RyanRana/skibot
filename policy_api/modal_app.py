"""Host the policy API on Modal: CPU only (the policies are small MLPs), scales to zero, websockets included.

    modal deploy policy_api/modal_app.py        # prints the public URL
"""
from pathlib import Path

import modal

HERE = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "numpy", "msgpack")
    .env({"HI_POLICY_DIR": "/root/policies"})
    .add_local_file(HERE / "hi_server.py", "/root/hi_server.py")
    .add_local_file(HERE / "mission.py", "/root/mission.py")
    .add_local_dir(HERE / "policies", "/root/policies")
)

app = modal.App("hazard-intelligence-api", image=image)


@app.function(cpu=1.0, memory=1024, scaledown_window=600, max_containers=1)  # missions live in memory
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def api():
    from hi_server import app as web

    return web
