/* Home page: the hero field (one network in one ink, dithered to a few
   pixel tones, resolved by a lens that follows the cursor) and the demo
   window that plays once, the first time it is seen, from an empty notebook
   to a written bundle's data open as a sheet. The window's contents come from
   a real run during the build; the script only reveals them. */

(() => {
  const NAVY = "#0C1F40";
  const AQUA = "#B4E7DD";
  const UI_FONT = '"Inter", system-ui, sans-serif';

  function prng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const smooth = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };

  /* =========================================================================
     The field. One network drawn in a single ink at a third of the screen's
     resolution, ordered-dithered to four tones, and scaled up without
     smoothing. The lens draws the same network at full resolution, magnified,
     with the direction of every edge. */
  const CELL = 3;
  const TONES = 4;
  const LEVELS = 12;
  const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map((v) => (v + 0.5) / 16);
  const DISTANCE = 3.2;
  const LENS_R = 96;
  const ZOOM = 2.2;

  /* One ink on the scheme's ground: navy on white, or aqua on navy. */
  function palette() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return dark ? { ground: NAVY, ink: AQUA } : { ground: "#FFFFFF", ink: NAVY };
  }

  function rgb(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  /* One network with two bodies. As a world, its nodes cover a globe evenly
     and join their nearest neighbours; broken apart, the same nodes sit on a
     jittered lattice that flows like cloth, its near end turned toward the
     viewer and its far end tapering into a thread toward the text. Each node
     takes the globe point nearest its lattice position, so the world unwraps
     rather than scatters. Every edge runs from a lower to a higher index, so
     both bodies are acyclic. */
  function network(random) {
    const NU = 64;
    const NV = 14;
    const N = NU * NV;
    const uv = new Float32Array(2 * N);
    for (let i = 0; i < NU; i++) {
      for (let j = 0; j < NV; j++) {
        uv[2 * (i * NV + j)] = (i + (random() - 0.5) * 0.7) / (NU - 1);
        uv[2 * (i * NV + j) + 1] = (j + (random() - 0.5) * 0.8) / (NV - 1);
      }
    }

    const edges = [];
    const kinds = [];
    const id = (i, j) => i * NV + j;
    const link = (a, b, p) => {
      if (random() >= p) return;
      edges.push(a, b);
      kinds.push(0);
    };
    for (let i = 0; i < NU; i++) {
      for (let j = 0; j < NV; j++) {
        if (i + 1 < NU) link(id(i, j), id(i + 1, j), 0.75);
        if (j + 1 < NV) link(id(i, j), id(i, j + 1), 0.6);
        if (i + 1 < NU && j + 1 < NV) link(id(i, j), id(i + 1, j + 1), 0.4);
        if (i + 1 < NU && j > 0) link(id(i, j), id(i + 1, j - 1), 0.4);
        if (i + 3 < NU) link(id(i, j), id(i + 2 + Math.floor(random() * 2), Math.floor(random() * NV)), 0.05);
      }
    }

    /* Globe points on a Fibonacci spiral, handed out in random node order. */
    const golden = Math.PI * (3 - Math.sqrt(5));
    const spiral = new Float32Array(3 * N);
    for (let p = 0; p < N; p++) {
      const y = 1 - (2 * (p + 0.5)) / N;
      const r = Math.sqrt(1 - y * y);
      const a = p * golden + (random() - 0.5) * 0.3;
      spiral[3 * p] = Math.cos(a) * r;
      spiral[3 * p + 1] = y;
      spiral[3 * p + 2] = Math.sin(a) * r;
    }
    const globe = new Float32Array(3 * N);
    const taken = new Uint8Array(N);
    const order = Array.from({ length: N }, (_, k) => k);
    for (let k = N - 1; k > 0; k--) {
      const j = Math.floor(random() * (k + 1));
      [order[k], order[j]] = [order[j], order[k]];
    }
    for (const k of order) {
      const lon = uv[2 * k] * Math.PI * 2;
      const lat = (uv[2 * k + 1] - 0.5) * Math.PI * 0.9;
      const tx = Math.cos(lat) * Math.cos(lon);
      const ty = Math.sin(lat);
      const tz = Math.cos(lat) * Math.sin(lon);
      let best = -1;
      let bestDot = -2;
      for (let p = 0; p < N; p++) {
        if (taken[p]) continue;
        const d = tx * spiral[3 * p] + ty * spiral[3 * p + 1] + tz * spiral[3 * p + 2];
        if (d > bestDot) {
          bestDot = d;
          best = p;
        }
      }
      taken[best] = 1;
      globe[3 * k] = spiral[3 * best];
      globe[3 * k + 1] = spiral[3 * best + 1];
      globe[3 * k + 2] = spiral[3 * best + 2];
    }

    /* The world's own edges: each node to its three nearest on the globe. */
    const seen = new Set();
    for (let k = 0; k < N; k++) {
      const best = [-1, -1, -1];
      const dots = [-2, -2, -2];
      for (let j = 0; j < N; j++) {
        if (j === k) continue;
        const d = globe[3 * k] * globe[3 * j] + globe[3 * k + 1] * globe[3 * j + 1] + globe[3 * k + 2] * globe[3 * j + 2];
        for (let s = 0; s < 3; s++) {
          if (d > dots[s]) {
            dots.splice(s, 0, d);
            best.splice(s, 0, j);
            dots.pop();
            best.pop();
            break;
          }
        }
      }
      for (const j of best) {
        const a = Math.min(j, k);
        const b = Math.max(j, k);
        if (seen.has(a * N + b) || random() > 0.75) continue;
        seen.add(a * N + b);
        edges.push(a, b);
        kinds.push(1);
      }
    }

    const yaw = 0.7;
    const roll = -0.45;
    const tilt = 0.38;
    const RADIUS = 0.9;
    return {
      count: N,
      edges: Uint16Array.from(edges),
      kinds: Uint8Array.from(kinds),
      /* How far along the ribbon each node sits; the world gathers near end first. */
      along: (k) => uv[2 * k],
      /* `mix` is each node's progress from ribbon (0) to world (1). */
      place(out, t, mix) {
        const c = Math.cos(t * 0.11);
        const s = Math.sin(t * 0.11);
        for (let k = 0; k < N; k++) {
          const u = uv[2 * k];
          const v = uv[2 * k + 1];
          const fold = 0.9 * Math.sin(u * 2.6 - t * 0.1) + 0.3;
          const half = (0.42 + 0.14 * Math.sin(u * 3.1 + 0.8)) * (0.08 + 0.92 * smooth(0, 0.5, u));
          const across = (v - 0.5) * 2 * half;
          const x = (u * 2 - 1) * 2.3;
          const y = 0.34 * Math.sin(u * 2.8 + t * 0.12) + across * Math.cos(fold);
          const z =
            0.35 * Math.sin(u * 2.2 - t * 0.09 + v * 1.4) +
            across * Math.sin(fold) +
            0.6 * across * across +
            0.03 * Math.sin(v * 9 + u * 11 + t * 0.5);
          const x1 = x * Math.cos(yaw) - z * Math.sin(yaw);
          const rx = x1 * Math.cos(roll) - y * Math.sin(roll);
          const ry = x1 * Math.sin(roll) + y * Math.cos(roll);
          const rz = z * Math.cos(yaw) + x * Math.sin(yaw);

          const gx = globe[3 * k] * c + globe[3 * k + 2] * s;
          const gz0 = globe[3 * k + 2] * c - globe[3 * k] * s;
          const gy = globe[3 * k + 1] * Math.cos(tilt) - gz0 * Math.sin(tilt);
          const gz = gz0 * Math.cos(tilt) + globe[3 * k + 1] * Math.sin(tilt);

          /* In flight, a node swings out from the globe before it lands. */
          const w = mix[k];
          const burst = 0.45 * Math.sin(Math.PI * w);
          out[3 * k] = rx + (gx * RADIUS - rx) * w + gx * burst;
          out[3 * k + 1] = ry + (gy * RADIUS - ry) * w + gy * burst;
          out[3 * k + 2] = rz + (gz * RADIUS - rz) * w + gz * burst;
        }
      },
      /* The globe's silhouette radius, for the shaded body and the click target. */
      radius: RADIUS,
    };
  }

  function mountField(canvas, { on, cleanup, reduceMotion }) {
    const hero = canvas.closest(".pg-hero");
    const copy = hero.querySelector(".pg-hero__copy");
    const g = canvas.getContext("2d");
    const hint = hero.querySelector("[data-pg-hint]");
    const model = network(prng(20260712));
    const n = model.count;
    const edges = model.edges;
    const kinds = model.kinds;
    const m = edges.length / 2;
    const mix = new Float32Array(n).fill(1);
    const stagger = prng(11);
    const delay = Float32Array.from({ length: n }, () => stagger());
    /* A click breaks the world into the ribbon; it flows, then slowly gathers
       back into the world, near end first. */
    const BREAK = 2.8;
    const HOLD = 9;
    const GATHER = 7;
    let phase = "world";
    let phaseAt = 0;
    let whole = mix[0];
    const pos = new Float32Array(n * 3);
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    const near = new Float32Array(n);
    const lit = new Float32Array(n);
    const edgeLevel = new Uint8Array(m);
    const nodeLevel = new Uint8Array(n);
    const white = Array.from({ length: LEVELS }, (_, b) => `rgba(255, 255, 255, ${(b + 1) / LEVELS})`);

    const lo = document.createElement("canvas");
    const lg = lo.getContext("2d", { willReadFrequently: true });
    const px = document.createElement("canvas");
    const pg = px.getContext("2d");
    let image = null;
    let tones = new Uint32Array(TONES + 1);
    let P = palette();
    let W = 0;
    let H = 0;
    let GW = 0;
    let GH = 0;
    let dpr = 1;
    let view = null;
    let pointer = null;
    let lastPointer = 0;
    let lens = null;
    let focus = -1;
    let focusSince = 0;
    const pick = prng(7);
    let clock = 0;
    let frame = 0;
    let visible = true;
    let lastTime = performance.now();

    /* Where the figure sits: right of the copy on wide screens, below it on
       narrow ones. It fades out toward the text and toward the hero's top and
       bottom, so it never meets a hard edge. `ink` thins strokes on a small
       figure so it stays a network rather than a blot. */
    function frameView() {
      const box = hero.getBoundingClientRect();
      const rim = H * 0.1;
      const ends = (y) => smooth(0, rim, y) * smooth(H, H - rim, y);
      const ink = (scale) => Math.min(1, Math.max(0.55, scale / 360));
      if (W < 960) {
        const top = copy ? copy.getBoundingClientRect().bottom - box.top + 24 : H * 0.5;
        const room = H - top;
        const scale = Math.min(W * 0.4, room * 0.38);
        return {
          left: 0,
          cx: W / 2,
          cy: top + room * 0.43,
          scale,
          ink: ink(scale),
          fade: (x, y) => smooth(top - 20, top + 80, y) * smooth(H, H - rim, y),
        };
      }
      const right = copy ? copy.getBoundingClientRect().right - box.left : W * 0.42;
      const x0 = Math.max(W * 0.44, right + 40);
      const scale = Math.min(H * 0.42, (W - x0) * 0.46);
      return {
        left: x0,
        cx: x0 + (W - x0) * 0.56,
        cy: H * 0.47,
        scale,
        ink: ink(scale),
        fade: (x, y) => smooth(x0 - W * 0.16, x0 + W * 0.06, x) * ends(y),
      };
    }

    function enter(next) {
      phase = next;
      phaseAt = clock;
      if (next === "world") mix.fill(1);
      if (next === "ribbon") mix.fill(0);
      syncHint();
    }

    /* Break the world apart, or gather the ribbon back; ignored in flight. */
    function toggle() {
      if (reduceMotion.matches) return;
      if (phase === "world") enter("breaking");
      else if (phase === "ribbon") enter("gathering");
      start();
    }

    /* The globe's outline on screen: perspective makes it a little larger
       than its nominal radius. */
    function silhouette() {
      const r = model.radius;
      return (view.scale * r * DISTANCE) / Math.sqrt(DISTANCE * DISTANCE - r * r);
    }

    /* The hint is the same action as a button: in a clear band under the
       world, named for what the next click does, and quiet while the nodes
       travel. It stays focusable in flight so a keyboard press never drops
       focus. */
    function syncHint() {
      if (!hint) return;
      hint.hidden = reduceMotion.matches;
      hint.setAttribute("aria-disabled", String(phase !== "world" && phase !== "ribbon"));
      hint.textContent = phase === "world" || phase === "gathering" ? "Break the world apart" : "Gather it back";
      if (view) {
        hint.style.left = `${view.cx}px`;
        hint.style.top = `${view.cy + silhouette() + 30}px`;
      }
    }

    function settle() {
      const T = clock - phaseAt;
      if (phase === "breaking") {
        for (let k = 0; k < n; k++) {
          const local = Math.min(1, Math.max(0, (T - delay[k] * BREAK * 0.45) / (BREAK * 0.55)));
          mix[k] = 1 - easeInOut(local);
        }
        if (T >= BREAK) enter("ribbon");
      } else if (phase === "ribbon") {
        if (T >= HOLD) enter("gathering");
      } else if (phase === "gathering") {
        for (let k = 0; k < n; k++) {
          const lag = (1 - model.along(k)) * 0.45 + delay[k] * 0.15;
          const local = Math.min(1, Math.max(0, (T - lag * GATHER) / (GATHER * 0.4)));
          mix[k] = easeInOut(local);
        }
        if (T >= GATHER) enter("world");
      }
    }

    function overWorld(x, y) {
      return Math.hypot(x - view.cx, y - view.cy) < silhouette() * 1.04;
    }

    function project() {
      model.place(pos, clock, mix);
      let sum = 0;
      for (let k = 0; k < n; k++) sum += mix[k];
      whole = sum / n;
      const reach = 1.8 - 0.8 * whole;
      for (let k = 0; k < n; k++) {
        const z = pos[3 * k + 2];
        const f = DISTANCE / (DISTANCE - z);
        sx[k] = view.cx + pos[3 * k] * f * view.scale;
        sy[k] = view.cy + pos[3 * k + 1] * f * view.scale;
        near[k] = Math.min(1, Math.max(0, (z + reach) / (2 * reach)));
        lit[k] = view.fade(sx[k], sy[k]);
        const level = (0.35 + 0.65 * near[k]) * lit[k];
        nodeLevel[k] = level < 0.04 ? 255 : Math.min(LEVELS - 1, Math.floor(level * LEVELS));
      }
      /* The ribbon's edges hold only between nodes on the ribbon, the world's
         only between nodes on the globe; in flight, nodes travel loose. */
      for (let e = 0; e < m; e++) {
        const a = edges[2 * e];
        const b = edges[2 * e + 1];
        const hold = smooth(0.8, 1, kinds[e] ? mix[a] * mix[b] : (1 - mix[a]) * (1 - mix[b]));
        const level =
          (0.06 + 0.75 * Math.pow((near[a] + near[b]) * 0.5, 1.4)) * Math.min(lit[a], lit[b]) * hold;
        edgeLevel[e] = level < 0.04 ? 255 : Math.min(LEVELS - 1, Math.floor(level * LEVELS));
      }
    }

    function nodeRadius(k) {
      return 0.5 + 2.2 * near[k] * near[k];
    }

    /* Edges and nodes, batched by brightness: one stroke and one fill per level. */
    function trace(c, lineWidth, radius, style) {
      c.lineWidth = lineWidth;
      for (let b = 0; b < LEVELS; b++) {
        style(c, b);
        c.beginPath();
        for (let e = 0; e < m; e++) {
          if (edgeLevel[e] !== b) continue;
          const a = edges[2 * e];
          const d = edges[2 * e + 1];
          c.moveTo(sx[a], sy[a]);
          c.lineTo(sx[d], sy[d]);
        }
        c.stroke();
        c.beginPath();
        for (let k = 0; k < n; k++) {
          if (nodeLevel[k] !== b) continue;
          const r = radius(k);
          c.moveTo(sx[k] + r, sy[k]);
          c.arc(sx[k], sy[k], r, 0, Math.PI * 2);
        }
        c.fill();
      }
    }

    /* The globe's body: a faint light falling from the upper left. */
    function body(c, strength, colour) {
      const r = silhouette();
      const shade = c.createRadialGradient(
        view.cx - r * 0.35, view.cy - r * 0.4, r * 0.1,
        view.cx, view.cy, r,
      );
      shade.addColorStop(0, colour(strength * whole * whole));
      shade.addColorStop(1, colour(0));
      c.fillStyle = shade;
      c.beginPath();
      c.arc(view.cx, view.cy, r, 0, Math.PI * 2);
      c.fill();
    }

    /* Low-resolution pass: light on black, then ordered dither to the ink's tones. */
    function paintField() {
      lg.setTransform(1, 0, 0, 1, 0, 0);
      lg.globalCompositeOperation = "source-over";
      lg.fillStyle = "#000";
      lg.fillRect(0, 0, GW, GH);
      lg.setTransform(1 / CELL, 0, 0, 1 / CELL, 0, 0);
      lg.globalCompositeOperation = "lighter";
      if (whole > 0.01) body(lg, 0.3, (a) => `rgba(255, 255, 255, ${a})`);
      trace(lg, CELL * 0.75 * view.ink, (k) => CELL * (0.3 + 0.45 * nodeRadius(k)) * view.ink, (c, b) => {
        c.strokeStyle = white[b];
        c.fillStyle = white[b];
      });
      const light = lg.getImageData(0, 0, GW, GH).data;
      const out = new Uint32Array(image.data.buffer);
      for (let y = 0; y < GH; y++) {
        const row = (y & 3) * 4;
        for (let x = 0; x < GW; x++) {
          const i = y * GW + x;
          const q = Math.floor((light[4 * i] / 255) * TONES + BAYER[row + (x & 3)]);
          out[i] = tones[q > TONES ? TONES : q];
        }
      }
      pg.putImageData(image, 0, 0);
      g.imageSmoothingEnabled = false;
      g.drawImage(px, 0, 0, GW * CELL, GH * CELL);
    }

    /* The lens: the same network at full resolution, magnified, with arrows. */
    function paintLens() {
      g.save();
      g.shadowColor = P.ink === NAVY ? "rgba(12, 31, 64, 0.22)" : "rgba(0, 0, 0, 0.45)";
      g.shadowBlur = 28;
      g.shadowOffsetY = 10;
      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.fillStyle = P.ground;
      g.fill();
      g.restore();

      g.save();
      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.clip();
      g.translate(lens.x, lens.y);
      g.scale(ZOOM, ZOOM);
      g.translate(-lens.x, -lens.y);
      if (whole > 0.01) {
        const [r0, g0, b0] = rgb(P.ink);
        body(g, 0.1, (a) => `rgba(${r0}, ${g0}, ${b0}, ${a})`);
      }
      g.strokeStyle = P.ink;
      g.fillStyle = P.ink;
      trace(g, 1.1 / ZOOM, nodeRadius, (c, b) => {
        c.globalAlpha = (b + 1) / LEVELS;
      });
      const reach = LENS_R / ZOOM;
      const head = 5 / ZOOM;
      g.lineWidth = 1.1 / ZOOM;
      for (let e = 0; e < m; e++) {
        if (edgeLevel[e] === 255) continue;
        const a = edges[2 * e];
        const d = edges[2 * e + 1];
        const tx = sx[a] + (sx[d] - sx[a]) * 0.6;
        const ty = sy[a] + (sy[d] - sy[a]) * 0.6;
        if (Math.abs(tx - lens.x) > reach || Math.abs(ty - lens.y) > reach) continue;
        const len = Math.hypot(sx[d] - sx[a], sy[d] - sy[a]);
        if (len < head * 3) continue;
        const ux = (sx[d] - sx[a]) / len;
        const uy = (sy[d] - sy[a]) / len;
        g.globalAlpha = (edgeLevel[e] + 1) / LEVELS;
        g.beginPath();
        g.moveTo(tx - ux * head - uy * head * 0.6, ty - uy * head + ux * head * 0.6);
        g.lineTo(tx, ty);
        g.lineTo(tx - ux * head + uy * head * 0.6, ty - uy * head - ux * head * 0.6);
        g.stroke();
      }
      g.restore();

      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.strokeStyle = P.ink;
      g.lineWidth = 1;
      g.stroke();
      const tag = `×${ZOOM.toFixed(1)}`;
      g.font = `500 10px ${UI_FONT}`;
      const w = g.measureText(tag).width + 12;
      g.fillStyle = P.ink;
      g.beginPath();
      g.roundRect(lens.x - w / 2, lens.y + LENS_R - 22, w, 15, 7.5);
      g.fill();
      g.fillStyle = P.ground;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(tag, lens.x, lens.y + LENS_R - 14.2);
    }

    /* A node the lens can rest on: lit, mid-depth, and well inside the frame. */
    function fits(k) {
      const margin = LENS_R + 12;
      return (
        lit[k] > 0.95 && near[k] > 0.35 && near[k] < 0.85 &&
        sx[k] > view.left + margin && sx[k] < W - margin && sy[k] > margin && sy[k] < H - margin
      );
    }

    /* Idle, the lens travels from node to node; under the pointer it follows. */
    function moveLens(dt) {
      const idle = !pointer || performance.now() - lastPointer > 2600;
      if (idle && (focus < 0 || clock - focusSince > 6 || !fits(focus))) {
        focus = -1;
        for (let tries = 0; tries < 200 && focus < 0; tries++) {
          const k = Math.floor(pick() * n);
          if (fits(k)) focus = k;
        }
        focusSince = clock;
      }
      const target = !idle ? pointer : focus >= 0 ? { x: sx[focus], y: sy[focus] } : { x: view.cx, y: view.cy };
      if (!lens || reduceMotion.matches) {
        lens = { x: target.x, y: target.y };
        return;
      }
      const follow = Math.min(1, dt * (idle ? 1.2 : 18));
      lens.x += (target.x - lens.x) * follow;
      lens.y += (target.y - lens.y) * follow;
    }

    function draw(dt) {
      if (!reduceMotion.matches) clock += dt;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      settle();
      project();
      paintField();
      if (W >= 600) {
        moveLens(dt);
        paintLens();
      }
    }

    function resize() {
      P = palette();
      const box = hero.getBoundingClientRect();
      W = box.width;
      H = box.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      GW = Math.ceil(W / CELL);
      GH = Math.ceil(H / CELL);
      lo.width = px.width = GW;
      lo.height = px.height = GH;
      image = pg.createImageData(GW, GH);
      const ground = rgb(P.ground);
      const ink = rgb(P.ink);
      tones = new Uint32Array(TONES + 1);
      for (let q = 1; q <= TONES; q++) {
        const [r, gr, b] = ground.map((c, i) => Math.round(c + (ink[i] - c) * (q / TONES)));
        tones[q] = (255 << 24) | (b << 16) | (gr << 8) | r;
      }
      view = frameView();
      syncHint();
      draw(0);
    }

    function tick(now) {
      frame = 0;
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      draw(dt);
      if (visible && !document.hidden && !reduceMotion.matches) frame = requestAnimationFrame(tick);
    }

    function start() {
      if (frame) return;
      lastTime = performance.now();
      frame = requestAnimationFrame(tick);
    }

    const observer = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) start();
    });
    observer.observe(hero);
    const sizer = new ResizeObserver(() => resize());
    sizer.observe(hero);
    /* The scheme toggle swaps the body attribute without a navigation. */
    const schemes = new MutationObserver(() => resize());
    schemes.observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });

    on(hero, "pointermove", (event) => {
      const box = hero.getBoundingClientRect();
      pointer = { x: event.clientX - box.left, y: event.clientY - box.top };
      lastPointer = performance.now();
      const clickable = (phase === "world" && overWorld(pointer.x, pointer.y)) || phase === "ribbon";
      canvas.style.cursor = clickable && !reduceMotion.matches ? "pointer" : "";
      start();
    });
    on(hero, "pointerleave", () => {
      pointer = null;
      start();
    });
    on(canvas, "click", (event) => {
      const box = hero.getBoundingClientRect();
      if (phase === "ribbon" || overWorld(event.clientX - box.left, event.clientY - box.top)) toggle();
    });
    if (hint) on(hint, "click", toggle);
    on(document, "visibilitychange", () => {
      if (!document.hidden && visible) start();
    });

    resize();
    start();
    cleanup(() => {
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      sizer.disconnect();
      schemes.disconnect();
    });
  }

  /* =========================================================================
     The demo window. Every element is already in its final state in the page;
     `render(time)` derives the whole window from one clock, so pausing,
     resuming, replaying, and jumping to a chapter are the same operation.
     Without the script the final state stays; with reduced motion nothing
     plays, and the chapters step through still frames instead. */
  const TYPE_MS = 16;

  function mountDemo(section, { on, cleanup, reduceMotion }) {
    const win = section.querySelector("[data-pg-window]");
    const button = section.querySelector("[data-pg-play]");
    const chapters = [...section.querySelectorAll("[data-pg-chapter]")];
    if (!win || !button) return;
    const one = (selector) => win.querySelector(selector);
    const all = (selector) => [...win.querySelectorAll(selector)];
    const body = one(".pg-demo__body");
    const notebook = one('[data-view="notebook"]');
    const views = all("[data-view]");
    const tabs = all("[data-tab]");
    const cells = all(".pg-demo__cell");
    const lines = cells.map((c) => [...c.querySelectorAll(".pg-demo__line")]);
    const widths = lines.map((ls) => ls.map((l) => Number(l.style.getPropertyValue("--chars"))));
    const typed = widths.map((w) => w.reduce((a, b) => a + b, 0));
    const counts = cells.map((c) => c.querySelector(".pg-demo__count"));
    const outputs = cells.map((c) => c.querySelector(".pg-demo__output"));
    const bundle = all('[data-made="bundle"]');
    const corpus = all('[data-made="corpus"]');
    const rowsOf = (view) => [...one(`[data-view="${view}"]`).querySelectorAll("tbody tr")];
    const sheetRows = rowsOf("dataset");
    const members = rowsOf("archive");
    const worldRows = rowsOf("outcome");
    const formula = one(".pg-demo__formula");
    const kernel = one("[data-pg-kernel]");
    const cursor = one(".pg-demo__cursor");
    const progress = one(".pg-demo__progress");
    const dataset = one('[data-file="dataset.csv"]');
    const archive = one('[data-file="worlds.npz"]');
    const member = one('[data-member="outcome_raw.npy"]');
    const notebookTab = one('[data-tab="notebook"]');
    if (cells.length !== 4 || !dataset || !archive || !member || !notebookTab) return;

    /* The script: each beat starts where the last one ends. */
    const beats = {};
    let end = 300;
    const beat = (name, ms) => {
      beats[name] = [end, end + ms];
      end += ms;
    };
    beat("type1", typed[0] * TYPE_MS);
    beat("run1", 800);
    beat("cell2", 450);
    beat("type2", typed[1] * TYPE_MS);
    beat("run2", 400 + bundle.length * 130 + 300);
    beat("rest1", 300);
    beat("move1", 900);
    beat("click1", 300);
    beat("open1", 300 + sheetRows.length * 30 + 1800);
    beat("move2", 900);
    beat("click2", 300);
    beat("cell3", 450);
    beat("type3", typed[2] * TYPE_MS);
    beat("run3", 1600);
    beat("cell4", 450);
    beat("type4", typed[3] * TYPE_MS);
    beat("run4", 700);
    beat("rest2", 300);
    beat("move3", 900);
    beat("click3", 300);
    beat("open3", 300 + members.length * 20 + 1400);
    beat("move4", 900);
    beat("click4", 300);
    beat("open4", 300 + worldRows.length * 40 + 300);
    const END = end + 200;
    const phase = (name, time) => {
      const [a, b] = beats[name];
      return Math.min(1, Math.max(0, (time - a) / (b - a)));
    };

    /* What the pane shows, and when each tab first opens. */
    const scenes = [
      [0, "notebook"],
      [beats.open1[0], "dataset"],
      [beats.cell3[0], "notebook"],
      [beats.open3[0], "archive"],
      [beats.open4[0], "outcome"],
    ];
    const opens = { notebook: 0, dataset: beats.open1[0], archive: beats.open3[0], outcome: beats.open4[0] };
    const path = [
      ["move1", "click1", dataset],
      ["move2", "click2", notebookTab],
      ["move3", "click3", archive],
      ["move4", "click4", member],
    ];
    const starts = [0, beats.cell2[0], beats.move1[0], beats.move2[0], beats.cell4[0], beats.move3[0]];

    function type(i, p, active) {
      let left = Math.round(p * typed[i]);
      let caret = active;
      lines[i].forEach((line, j) => {
        const n = Math.max(0, Math.min(widths[i][j], left));
        left -= widths[i][j];
        line.style.setProperty("--typed", n);
        const here = caret && (n < widths[i][j] || j === lines[i].length - 1);
        line.classList.toggle("is-caret", here);
        if (here) caret = false;
      });
    }

    function run(i, time) {
      const [a, b] = beats[`run${i + 1}`];
      counts[i].textContent = time < a ? "[ ]" : time < b ? "[*]" : `[${i + 1}]`;
      if (outputs[i]) outputs[i].classList.toggle("is-absent", time < b);
      return time >= a && time < b;
    }

    /* Where a target sits in the window; remembered, because a target the
       cursor left may be hidden by the time the next move starts from it. */
    const spots = new Map();
    function spot(el) {
      const r = el.getBoundingClientRect();
      if (r.width) {
        const box = win.getBoundingClientRect();
        spots.set(el, [r.left - box.left + Math.min(46, r.width * 0.4), r.top - box.top + r.height * 0.55]);
      }
      return spots.get(el);
    }

    function point(time) {
      const box = win.getBoundingClientRect();
      let from = [box.width * 0.82, box.height * 0.86];
      let at = from;
      let shown = false;
      let press = 1;
      for (const [move, click, el] of path) {
        const to = spot(el) || from;
        if (time >= beats[move][0]) {
          const p = easeInOut(phase(move, time));
          at = [from[0] + (to[0] - from[0]) * p, from[1] + (to[1] - from[1]) * p];
          shown = time < beats[click][1] + 500;
          press = time >= beats[click][0] && time < beats[click][0] + 140 ? 0.86 : 1;
        }
        from = to;
      }
      cursor.style.transform = `translate(${at[0]}px, ${at[1]}px) scale(${press})`;
      cursor.classList.toggle("is-on", shown);
    }

    function render(time) {
      const busy = cells.map((c, i) => {
        const ready = i ? beats[`cell${i + 1}`][1] : 0;
        if (i) c.classList.toggle("is-absent", time < ready);
        type(i, phase(`type${i + 1}`, time), time >= ready && time < beats[`run${i + 1}`][0]);
        return run(i, time);
      });
      kernel.classList.toggle("is-busy", busy.some(Boolean));

      const made = time - beats.run2[0];
      bundle.forEach((f, i) => f.classList.toggle("is-absent", made < 300 + i * 130));
      corpus.forEach((f) => f.classList.toggle("is-absent", time < beats.run3[1]));

      let scene = "notebook";
      for (const [at, name] of scenes) if (time >= at) scene = name;
      views.forEach((v) => v.classList.toggle("is-open", v.dataset.view === scene));
      tabs.forEach((t) => {
        t.classList.toggle("is-absent", time < opens[t.dataset.tab]);
        t.classList.toggle("is-active", t.dataset.tab === scene);
      });
      /* A sheet takes the whole window: the explorer folds away. */
      body.classList.toggle("is-wide", scene !== "notebook");
      /* Like a notebook, keep the latest cell in view when the pane is short. */
      notebook.scrollTop = notebook.scrollHeight;

      dataset.classList.toggle("is-selected", time >= beats.click1[0] && time < beats.click3[0]);
      archive.classList.toggle("is-selected", time >= beats.click3[0]);
      member.classList.toggle("is-selected", time >= beats.click4[0]);
      const reveal = (rows, since, step) => rows.forEach((r, i) => r.classList.toggle("is-off", since < i * step));
      reveal(sheetRows, time - beats.open1[0] - 300, 30);
      reveal(members, time - beats.open3[0] - 300, 20);
      reveal(worldRows, time - beats.open4[0] - 300, 40);
      formula.classList.toggle("is-off", time < beats.open1[0] + 300 + sheetRows.length * 30);

      point(time);
      progress.style.setProperty("--progress", Math.min(1, time / END));
      let current = -1;
      starts.forEach((s, i) => {
        if (time > 0 && time >= s) current = i;
      });
      chapters.forEach((c, i) => {
        c.classList.toggle("is-current", i === current);
        c.classList.toggle("is-before", i < current);
        if (i === current) c.setAttribute("aria-current", "step");
        else c.removeAttribute("aria-current");
      });
    }

    /* Reduced motion: no playback. The page loads on the last frame, marked as
       the last chapter; each chapter shows its own last frame. */
    if (reduceMotion.matches) {
      render(END);
      chapters.forEach((c, i) => on(c, "click", () => render((starts[i + 1] ?? END) - 1)));
      return;
    }

    let time = 0;
    let frame = 0;
    let last = 0;
    let playing = false;
    let started = false;
    let held = false;
    let inView = false;
    let away = false;

    function sync() {
      button.hidden = false;
      button.textContent = time >= END ? "Replay" : playing ? "Pause" : "Play";
    }

    function tick(now) {
      frame = 0;
      time = Math.min(END, time + (now - last));
      last = now;
      render(time);
      if (time >= END) playing = false;
      if (playing) frame = requestAnimationFrame(tick);
      sync();
    }

    function play() {
      if (playing) return;
      if (time >= END) time = 0;
      playing = true;
      last = performance.now();
      frame = requestAnimationFrame(tick);
      sync();
    }

    function pause() {
      playing = false;
      if (frame) cancelAnimationFrame(frame);
      frame = 0;
      sync();
    }

    win.classList.add("is-live");
    render(0);
    sync();

    /* Plays the first time it is seen; leaving the view pauses it, and
       coming back resumes it unless the reader paused it. */
    const observer = new IntersectionObserver(
      ([entry]) => {
        inView = entry.isIntersecting;
        if (inView) {
          if (!started || (!held && time < END)) play();
          started = true;
        } else if (playing) {
          pause();
        }
      },
      { threshold: 0.45 },
    );
    observer.observe(win);
    /* A hidden tab stops the clock, so no part of the demo plays unseen. */
    on(document, "visibilitychange", () => {
      if (document.hidden) {
        away = playing;
        if (playing) pause();
      } else if (away && inView && !held) {
        away = false;
        play();
      }
    });
    on(button, "click", () => {
      held = playing;
      if (playing) pause();
      else play();
    });
    chapters.forEach((c, i) =>
      on(c, "click", () => {
        held = false;
        started = true;
        time = starts[i];
        render(time);
        play();
      }),
    );
    cleanup(() => {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
    });
  }

  function mount(context) {
    const field = document.querySelector("[data-pg-field]");
    if (field && field.getContext) mountField(field, context);
    const demo = document.querySelector("[data-pg-demo]");
    if (demo) mountDemo(demo, context);
  }

  if (window.pymcGenerator) window.pymcGenerator.register(mount);
})();
