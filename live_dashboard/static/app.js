"use strict";

// Polls the JSON API and renders it. Text is only ever assigned through textContent:
// labels, errors and details come from published records and are never parsed as HTML.

// Run ids as published (live_dashboard/publication.py): label slug + 12 hex digest chars.
const RUN_ID = /^[a-z0-9][a-z0-9-]{0,39}-[0-9a-f]{12}$/;
const PLOT_CONFIG = { displaylogo: false, responsive: true };
const MIN_DELAY_SECONDS = 1;
// The Date header has a one-second resolution: smaller offsets are not clock skew.
const CLOCK_SKEW_TOLERANCE_MS = 2000;

const view = {
  runs: null, // latest run list, null until the store has been read once
  selected: null, // run id shown or awaited (it may not be published yet)
  drawn: { runId: null, etag: null },
  pollSeconds: 10,
  refreshedAt: null, // milliseconds since the epoch of the last successful store read
  clockOffsetMs: 0, // server clock minus browser clock
  selectorKey: null,
  timer: null,
  pollToken: 0,
};

function byId(id) {
  return document.getElementById(id);
}

function show(element, text) {
  element.textContent = text || "";
  element.hidden = !text;
}

function requestedRun() {
  const value = new URLSearchParams(window.location.search).get("run");
  return value !== null && RUN_ID.test(value) ? value : null;
}

function rememberRun(runId) {
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState(null, "", url);
}

async function request(path, headers) {
  // Revalidation is explicit (If-None-Match), so the browser cache must stay out of it.
  const response = await fetch(path, { cache: "no-store", headers: headers || {} });
  const serverTime = Date.parse(response.headers.get("Date") || "");
  if (!Number.isNaN(serverTime)) {
    const offset = serverTime - Date.now();
    view.clockOffsetMs = Math.abs(offset) > CLOCK_SKEW_TOLERANCE_MS ? offset : 0;
  }
  return response;
}

function formatAge(seconds) {
  if (seconds < 120) return `${seconds} s`;
  if (seconds < 7200) return `${Math.round(seconds / 60)} min`;
  return `${Math.round(seconds / 3600)} h`;
}

function updateAge() {
  const element = byId("refresh-age");
  if (view.refreshedAt === null) {
    element.textContent = "store not refreshed yet";
    return;
  }
  const ageMs = Date.now() + view.clockOffsetMs - view.refreshedAt;
  element.textContent = `store refreshed ${formatAge(Math.max(0, Math.round(ageMs / 1000)))} ago`;
}

function renderSelector() {
  const options = view.runs.map((run) => [
    run.run_id,
    `${run.label} · ${run.run_id} · ${run.state} · ${run.published_worlds}/${run.target_worlds} worlds` +
      (run.loading_tasks ? ` · ${run.loading_tasks} tasks loading` : ""),
  ]);
  if (view.selected !== null && !view.runs.some((run) => run.run_id === view.selected)) {
    options.unshift([view.selected, `${view.selected} (not published yet)`]);
  }
  const key = JSON.stringify(options);
  if (key === view.selectorKey) return; // rebuilding would close an open dropdown
  view.selectorKey = key;
  const select = byId("run-select");
  select.replaceChildren(
    ...options.map(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      return option;
    }),
  );
  select.disabled = options.length === 0;
  if (view.selected !== null) select.value = view.selected;
}

// Returns the number of seconds to wait before the next poll.
async function loadRuns() {
  const response = await request("/api/runs");
  if (response.status !== 200 && response.status !== 503) {
    throw new Error(`run list: HTTP ${response.status}`);
  }
  const body = await response.json();
  view.pollSeconds = body.poll_seconds;
  view.refreshedAt = body.refreshed_at === null ? null : Date.parse(body.refreshed_at);
  updateAge();
  if (response.status === 503) {
    const reason = body.last_error ? `; store unreadable: ${body.last_error}` : "";
    show(byId("store-message"), `waiting for the first store refresh${reason}`);
    return Number(response.headers.get("Retry-After")) || view.pollSeconds;
  }
  show(byId("store-message"), body.last_error ? `store unreadable: ${body.last_error}` : "");
  show(
    byId("collection-issues"),
    body.issues.length ? `collection issues: ${body.issues.join("; ")}` : "",
  );
  view.runs = body.runs;
  if (view.selected === null) {
    view.selected = requestedRun();
    if (view.selected === null && view.runs.length > 0) {
      view.selected = view.runs[0].run_id; // newest first
      rememberRun(view.selected);
    }
  }
  renderSelector();
  return view.pollSeconds;
}

function waitFor(runId) {
  show(byId("run-message"), `waiting for run ${runId}`);
  byId("run-content").hidden = true;
  view.drawn = { runId: null, etag: null };
}

async function loadRun() {
  const runId = view.selected;
  if (view.runs === null) return; // the store message explains why
  if (runId === null) {
    show(byId("run-message"), "waiting for the first run to be published");
    return;
  }
  if (!view.runs.some((run) => run.run_id === runId)) {
    waitFor(runId);
    return;
  }
  const known = view.drawn.runId === runId ? view.drawn.etag : null;
  const response = await request(
    `/api/runs/${encodeURIComponent(runId)}`,
    known === null ? {} : { "If-None-Match": known },
  );
  if (view.selected !== runId || response.status === 304 || response.status === 503) return;
  if (response.status === 404) {
    waitFor(runId);
    return;
  }
  if (!response.ok) throw new Error(`run ${runId}: HTTP ${response.status}`);
  const etag = response.headers.get("ETag");
  const body = await response.json();
  if (view.selected !== runId) return;
  show(byId("run-message"), "");
  if (view.drawn.runId !== runId || view.drawn.etag !== etag) render(body);
  view.drawn = { runId, etag };
}

function renderPanel(element, panel) {
  element.querySelector(".panel-title").textContent = panel.title;
  const plot = element.querySelector(".plot");
  const empty = element.querySelector(".empty");
  if (panel.figure === null) {
    if (window.Plotly) window.Plotly.purge(plot);
    plot.hidden = true;
    show(empty, panel.empty);
  } else if (window.Plotly) {
    show(empty, "");
    plot.hidden = false;
    window.Plotly.react(plot, panel.figure.data, panel.figure.layout, PLOT_CONFIG);
  } else {
    plot.hidden = true;
    show(empty, "the plotting library did not load; reload the page");
  }
  show(element.querySelector(".note"), panel.note);
}

function renderStatus(status) {
  byId("status-line").textContent = status.line.join(" · ");
  const details = byId("problems");
  const count = status.problems.length;
  details.hidden = count === 0;
  details.querySelector("summary").textContent =
    `${count} ${count === 1 ? "task needs" : "tasks need"} attention`;
  details.querySelector("ul").replaceChildren(
    ...status.problems.map((problem) => {
      const item = document.createElement("li");
      item.textContent = `${problem.task_id} ${problem.status}: ${problem.detail}`;
      return item;
    }),
  );
}

function render(body) {
  byId("run-content").hidden = false;
  const run = body.run;
  byId("run-info").textContent = [
    run.label,
    run.run_id,
    `preset ${run.preset}`,
    `seed ${run.seed}`,
    `${run.worlds_per_task} worlds per task`,
    `created ${run.created_at}`,
    `code ${run.code_revision || "unknown"}`,
  ].join(" · ");
  for (const element of document.querySelectorAll("[data-card]")) {
    const card = body.cards.find((item) => item.id === element.dataset.card);
    element.querySelector(".card-title").textContent = card ? card.title : "";
    element.querySelector(".card-value").textContent = card ? card.text : "";
    element.querySelector(".card-subtitle").textContent = card ? card.subtitle : "";
  }
  for (const element of document.querySelectorAll("[data-figure]")) {
    const panel = body.figures.find((item) => item.id === element.dataset.figure);
    if (panel) renderPanel(element, panel);
  }
  renderStatus(body.status);
}

async function poll() {
  const token = ++view.pollToken;
  clearTimeout(view.timer);
  let delaySeconds = view.pollSeconds;
  try {
    delaySeconds = await loadRuns();
    await loadRun();
  } catch (error) {
    show(byId("store-message"), `dashboard unreachable: ${error.message || error}`);
  }
  if (token === view.pollToken) {
    view.timer = setTimeout(poll, Math.max(MIN_DELAY_SECONDS, delaySeconds) * 1000);
  }
}

byId("run-select").addEventListener("change", (event) => {
  view.selected = event.target.value;
  rememberRun(view.selected);
  view.drawn = { runId: null, etag: null };
  poll();
});
setInterval(updateAge, 1000);
poll();
