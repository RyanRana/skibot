// Pipeline drawer for the hiking viewer: every park in the ground.py dataset, what sets its trails apart, and which
// stages (data, course, tiles, walk, viewer) are built. Reads pipeline.json from hikesim.pipeline. Self contained:
// it only touches the page through the DOM (a button in #panel, and clicks on .course rows to open a course).
(() => {
  const css = `
  #pipe { position:fixed; top:12px; right:12px; bottom:12px; width:390px; display:none; flex-direction:column; z-index:5;
          background:var(--panel); border:1px solid var(--line); border-radius:10px; backdrop-filter:blur(6px); }
  #pipe.on { display:flex; }
  #pipe header { padding:12px 14px 8px; border-bottom:1px solid var(--line); }
  #pipe h2 { font-size:16px; margin:0; display:flex; justify-content:space-between; align-items:center; }
  #pipe .gen { color:var(--dim); font-size:11px; margin-top:2px; }
  #pipe .stages { display:grid; grid-template-columns:repeat(7,1fr); gap:6px; margin-top:10px; }
  #pipe .stg { font-size:11px; color:var(--dim); line-height:1.25; }
  #pipe .stg b { display:block; color:var(--ink); font-size:15px; }
  #pipe .bar { height:4px; background:rgba(255,255,255,.08); border-radius:2px; margin-top:4px; overflow:hidden; display:flex; }
  #pipe .bar i { display:block; height:100%; }
  #pipe .filters { display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; }
  #pipe .filters button.on { border-color:var(--accent); color:var(--accent); }
  #pipe .list { overflow:auto; padding:6px 10px 10px; flex:1; }
  #pipe .park { border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin:6px 0; cursor:pointer; }
  #pipe .park:hover, #pipe .park.open { border-color:var(--accent); }
  #pipe .park .top { display:flex; justify-content:space-between; gap:8px; }
  #pipe .park .nums { color:var(--dim); font-size:12px; }
  #pipe .tags { margin-top:5px; display:flex; flex-wrap:wrap; gap:4px; }
  #pipe .tags span { font-size:11px; padding:1px 7px; border-radius:10px; border:1px solid var(--line); color:var(--ink); }
  #pipe .strip { display:grid; grid-template-columns:repeat(7,1fr); gap:3px; margin-top:7px; }
  #pipe .strip i { height:6px; border-radius:3px; display:block; }
  #pipe .strip-l { display:grid; grid-template-columns:repeat(7,1fr); gap:3px; font-size:10px; color:var(--dim); margin-top:2px; }
  #pipe .detail { display:none; margin-top:8px; font-size:12px; }
  #pipe .park.open .detail { display:block; }
  #pipe .detail .s { display:grid; grid-template-columns:62px 1fr; gap:6px; padding:3px 0; border-top:1px solid var(--line); }
  #pipe .detail .s span:first-child { color:var(--dim); }
  #pipe code { display:block; font-size:11px; background:rgba(0,0,0,.35); border:1px solid var(--line); border-radius:6px;
               padding:6px 8px; margin-top:6px; white-space:pre-wrap; word-break:break-all; }
  #pipe .acts { display:flex; gap:6px; margin-top:6px; }
  #pipe video { width:100%; border-radius:6px; margin-top:8px; background:#000; display:block; }
  #pipe .vids { display:flex; gap:4px; flex-wrap:wrap; margin-top:4px; }
  #pipe .vids button.on { border-color:var(--accent); color:var(--accent); }
  #pipe .key { display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:var(--dim); margin-top:8px; }
  #pipe .key i { display:inline-block; width:14px; height:6px; border-radius:3px; margin-right:4px; vertical-align:middle; }
  @media (max-width:640px){ #pipe { top:auto; left:12px; width:auto; height:60%; } }`;
  const C = { done: "#3fbf6a", stale: "#e8b04a", todo: "transparent", blocked: "rgba(255,255,255,.06)", skip: "rgba(255,255,255,.06)" };
  const LABEL = { done: "built", stale: "stale, rebuild", todo: "not built yet", blocked: "waiting on an earlier stage", skip: "not needed" };
  const KEYS = ["data", "course", "trees", "tiles", "walk", "video", "viewer"];
  const SHORT = { data: "data", course: "course", trees: "trees", tiles: "tiles", walk: "walk", video: "video", viewer: "viewer" };
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const seg = (st) => `<i style="background:${C[st]};${st === "todo" ? "box-shadow:inset 0 0 0 1px var(--dim)" : ""}"></i>`;

  async function init() {
    let P;
    try { P = await (await fetch("pipeline.json", { cache: "no-store" })).json(); } catch { return; }
    const style = document.createElement("style"); style.textContent = css; document.head.appendChild(style);
    const btn = document.createElement("button"); btn.textContent = "Pipeline"; btn.setAttribute("aria-expanded", "false");
    (document.querySelector("#panel .row") || document.getElementById("panel")).appendChild(btn);
    const box = document.createElement("div"); box.id = "pipe"; document.body.appendChild(box);
    const n = P.parks;
    const stages = P.stages.map((s) => {
      const done = s.done - s.stale;
      return `<div class="stg"><b>${s.done}<span style="color:var(--dim);font-size:11px;font-weight:400"> / ${n}</span></b>${esc(s.label)}
        ${s.stale ? `<div style="color:${C.stale}">${s.stale} stale</div>` : ""}
        <div class="bar"><i style="width:${(100 * done) / n}%;background:${C.done}"></i><i style="width:${(100 * s.stale) / n}%;background:${C.stale}"></i></div></div>`;
    }).join("");
    const groups = {
      all: () => true, viewer: (a) => a.in_viewer, built: (a) => a.built, todo: (a) => !a.built,
    };
    const count = (g) => P.areas.filter(groups[g]).length;
    box.innerHTML = `<header>
        <h2>Pipeline <button id="pipe-x" aria-label="Close">Close</button></h2>
        <div class="gen">${n} Ann Arbor parks with trails in ${esc(P.dataset)}, checked ${esc(P.generated)}</div>
        <div class="stages">${stages}</div>
        <div class="key">${["done", "stale", "todo", "blocked"].map((k) => `<span>${seg(k).replace("<i", "<i")}${LABEL[k]}</span>`).join("")}</div>
        <div class="filters">
          <button data-g="all" class="on">All ${count("all")}</button><button data-g="viewer">In viewer ${count("viewer")}</button>
          <button data-g="built">Built ${count("built")}</button><button data-g="todo">Not built yet ${count("todo")}</button>
        </div></header><div class="list"></div>`;
    const list = box.querySelector(".list");
    const render = (g) => {
      list.innerHTML = P.areas.filter(groups[g]).map((a, i) => {
        const s = a.sig;
        const nums = `${s.km.toFixed(2)} km, climb ${s.climb_m} m, steepest ${s.steepest_deg} deg`;
        const detail = KEYS.map((k) => {
          const x = a.stages[k];
          return `<div class="s"><span>${SHORT[k]}</span><span><b style="color:${x.state === "done" ? C.done : x.state === "stale" ? C.stale : "var(--ink)"}">${LABEL[x.state]}</b>${x.at ? `, ${esc(x.at)}` : ""}${x.detail ? `<br><span style="color:var(--dim)">${esc(x.detail)}</span>` : ""}</span></div>`;
        }).join("");
        const nxt = a.next ? a.stages[a.next] : null;
        const surf = s.surfaces.map(([k, v]) => `${k.replace("_", " ")} ${Math.round(v * 100)}%`).join(", ");
        return `<div class="park" data-i="${i}" data-slug="${esc(a.slug)}">
          <div class="top"><b>${esc(a.name)}</b><span class="nums">${s.km.toFixed(1)} km</span></div>
          <div class="nums">${nums}</div>
          ${a.different.length ? `<div class="tags">${a.different.map((t) => `<span>${esc(t)}</span>`).join("")}</div>` : ""}
          <div class="strip">${KEYS.map((k) => seg(a.stages[k].state)).join("")}</div>
          <div class="strip-l">${KEYS.map((k) => `<span>${SHORT[k]}</span>`).join("")}</div>
          <div class="detail">
            ${a.videos && a.videos.length ? (() => {
              const vs = [...a.videos].sort((p, q) => (q.trees - p.trees) || ((q.walked_m || 0) - (p.walked_m || 0)));
              return `<video data-v controls muted playsinline preload="none" poster="${esc(vs[0].poster)}" src="${esc(vs[0].file)}"></video>
                ${vs.length > 1 ? `<div class="vids">${vs.map((v, j) => `<button data-vid="${esc(v.file)}" data-poster="${esc(v.poster)}" class="${j ? "" : "on"}">${esc(v.run)}${v.trees ? "" : " (no trees)"}</button>`).join("")}</div>` : ""}`;
            })() : ""}
            <div class="nums" style="margin-bottom:4px">Surfaces: ${esc(surf || "none mapped")}. ${s.ways} ways, vertical ${s.vertical_m ?? "?"} m, ${Math.round(s.steep_share * 100)}% over 15 deg.</div>
            ${detail}
            ${nxt ? `<div style="margin-top:6px">Next: ${esc(nxt === a.stages.viewer ? "export to the viewer" : a.next === "video" ? "render the hike video" : a.next === "trees" ? "build the 8-bit trees" : "build the " + a.next)}</div><code>${esc(nxt.cmd)}</code>` : ""}
            <div class="acts">${a.in_viewer ? `<button data-open="${esc(a.slug)}">Open in 3D</button>` : ""}${nxt ? `<button data-copy="${esc(nxt.cmd)}">Copy command</button>` : ""}</div>
          </div></div>`;
      }).join("") || `<div class="nums" style="padding:10px">Nothing here yet.</div>`;
    };
    render("all");
    const open = (on) => { box.classList.toggle("on", on); btn.setAttribute("aria-expanded", String(on)); };
    btn.onclick = () => open(!box.classList.contains("on"));
    box.querySelector("#pipe-x").onclick = () => open(false);
    box.querySelectorAll(".filters button").forEach((b) => b.onclick = () => {
      box.querySelectorAll(".filters button").forEach((o) => o.classList.toggle("on", o === b)); render(b.dataset.g);
    });
    list.addEventListener("click", async (e) => {
      const o = e.target.closest("[data-open]"), c = e.target.closest("[data-copy]"), vb = e.target.closest("[data-vid]");
      if (e.target.closest("video")) return;
      if (vb) { const v = vb.closest(".detail").querySelector("video"); v.src = vb.dataset.vid; v.poster = vb.dataset.poster; v.play();
        vb.parentNode.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b === vb)); return; }
      if (o) { const row = document.querySelector(`.course[data-slug="${o.dataset.open}"]`); if (row) { row.click(); open(false); } return; }
      if (c) { try { await navigator.clipboard.writeText(c.dataset.copy); c.textContent = "Copied"; } catch { c.textContent = "Select the command above"; } return; }
      const p = e.target.closest(".park"); if (p) p.classList.toggle("open");
    });
    if (new URLSearchParams(location.search).has("pipeline")) open(true);
  }
  document.readyState === "loading" ? addEventListener("DOMContentLoaded", init) : init();
})();
