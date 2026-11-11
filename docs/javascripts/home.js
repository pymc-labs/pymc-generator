/* Home page: the cinematic field (the world drawn during this build, focused
   with a loupe that follows the cursor) and the story that turns one call into
   data, a graph, a decomposition, and its proof. Both read the page's
   #pg-world payload; neither invents data. */

(() => {
  const NAVY = "#0C1F40";
  const AQUA = "#B4E7DD";
  const PERIWINKLE = "#9FAAE2";
  const PEACH = "#F6AE72";
  const MONO = '"Fragment Mono", Menlo, monospace';
  /* Edge colours follow the package figures: the hue names the source family. */
  const EDGE = {
    cy: "#3D57B0", cc: "#8792D6",
    zc: "#1E8C7A", zy: "#4FB3A0", zz: "#2E6F66",
    dc: "#B5532A", dy: "#D9783A", dz: "#E39A63",
    by: "#9AA3B5",
  };
  const ROLE = { D: "latent demand", Z: "control", C: "channel", B: "baseline", Y: "sales" };

  function readWorld() {
    const el = document.getElementById("pg-world");
    if (!el || !el.dataset.world) return null;
    try {
      return JSON.parse(el.dataset.world);
    } catch {
      return null;
    }
  }

  function unit(values) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo;
    return values.map((v) => (span > 0 ? (v - lo) / span : 0.5));
  }

  function prng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const family = (name) => name[0];
  const easeOut = (t) => 1 - Math.pow(1 - t, 4);

  /* Series per node: what the field pulses with at the current week. */
  function nodeSeries(world) {
    const out = {};
    for (const name of world.nodes) {
      const raw = name === "B" ? world.decomposition.baseline : world.series[name];
      out[name] = raw ? unit(raw) : null;
    }
    return out;
  }

  /* The field's own palette for the light and dark schemes. */
  function theme() {
    const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
    return dark
      ? {
          dark,
          ground: NAVY,
          node: "#11284d",
          ink: "#F2F4F8",
          ink2: "#C5CCDA",
          ghost: (a) => `rgba(242, 244, 248, ${a})`,
        }
      : {
          dark,
          ground: "#ffffff",
          node: "#ffffff",
          ink: NAVY,
          ink2: "#3b4a66",
          ghost: (a) => `rgba(12, 31, 64, ${a})`,
        };
  }

  /* =========================================================================
     The field. */
  function mountField(canvas, world, { on, cleanup, reduceMotion }) {
    const hero = canvas.closest(".pg-hero");
    const copy = hero.querySelector(".pg-hero__copy");
    const g = canvas.getContext("2d");
    const pulses = nodeSeries(world);
    const sales = unit(world.series.Y);
    const weeks = world.weeks;
    const maxCoef = Math.max(...world.edges.map((e) => Math.abs(e.coef || 0)), 1e-9);
    const random = prng(20260712);
    const particles = world.edges.map(() =>
      Array.from({ length: 3 }, () => random()),
    );
    let W = 0;
    let H = 0;
    let dpr = 1;
    let anchors = {};
    let ghosts = [];
    let bloom = null;
    let pointer = null;
    let lastPointer = 0;
    let week = 0;
    let frame = 0;
    let visible = true;
    let lastTime = performance.now();
    let clock = 0;
    let lens = { x: 0, y: 0 };
    const LENS_R = 96;
    const ZOOM = 2.2;
    let T = theme();

    const specimen = hero.querySelector("[data-pg-specimen]");
    if (specimen) {
      specimen.textContent =
        `Drawn while this page was built: ${world.name} · seed ${world.seed} · ` +
        `${weeks} weeks · identity error ${world.identity_error.toExponential(2)}`;
    }

    function region() {
      const narrow = W < 960;
      if (narrow) {
        const copyBottom = copy ? copy.getBoundingClientRect().bottom - hero.getBoundingClientRect().top : H * 0.5;
        const band = Math.min(140, H * 0.16);
        return { x0: W * 0.1, x1: W * 0.9, y0: copyBottom + 40, y1: H - band - 90 };
      }
      const textRight = copy ? copy.getBoundingClientRect().right - hero.getBoundingClientRect().left : W * 0.45;
      return { x0: Math.max(W * 0.5, textRight + 70), x1: W * 0.94, y0: H * 0.13, y1: H * 0.72 };
    }

    function layout() {
      const r = region();
      const columns = { D: 0, Z: 0.32, C: 0.64, B: 1, Y: 1 };
      const byFamily = {};
      for (const name of world.nodes) (byFamily[family(name)] ||= []).push(name);
      anchors = {};
      for (const [fam, names] of Object.entries(byFamily)) {
        names.forEach((name, i) => {
          let fy = (i + 1) / (names.length + 1);
          if (fam === "B") fy = 0.18;
          if (fam === "Y") fy = 0.62;
          anchors[name] = {
            x: r.x0 + columns[fam] * (r.x1 - r.x0),
            y: r.y0 + fy * (r.y1 - r.y0),
            phase: random() * Math.PI * 2,
          };
        });
      }
      lens = { x: anchors.Y.x - 80, y: anchors.Y.y - 40 };
    }

    /* Depth: three layers of other worlds, out of focus, drawn once per size. */
    function paintGhosts() {
      const rand = prng(42);
      ghosts = [0.22, 0.5, 0.82].map((depth, layerIndex) => {
        const layer = document.createElement("canvas");
        layer.width = Math.ceil(W * dpr * 1.2);
        layer.height = Math.ceil(H * dpr * 1.2);
        const c = layer.getContext("2d");
        c.scale(dpr, dpr);
        c.filter = `blur(${[9, 5, 2.4][layerIndex]}px)`;
        const count = Math.round((W / 1440) * [26, 18, 12][layerIndex]) + 6;
        const pts = Array.from({ length: count }, () => ({
          x: rand() * W * 1.2,
          y: rand() * H * 1.2,
          r: (4 + rand() * 7) * (0.6 + depth),
          fill: [null, AQUA, PERIWINKLE, null, PEACH][Math.floor(rand() * 5)],
        }));
        c.lineWidth = 1;
        c.strokeStyle = T.ghost(0.08 + depth * 0.1);
        for (const p of pts) {
          const targets = pts
            .filter((q) => q.x > p.x && q.x - p.x < W * 0.22 && Math.abs(q.y - p.y) < H * 0.3)
            .slice(0, 2);
          for (const q of targets) {
            c.beginPath();
            c.moveTo(p.x, p.y);
            c.quadraticCurveTo((p.x + q.x) / 2, (p.y + q.y) / 2 - 18, q.x, q.y);
            c.stroke();
          }
        }
        for (const p of pts) {
          c.beginPath();
          c.arc(p.x, p.y, p.r, 0, Math.PI * 2);
          c.fillStyle = p.fill ? p.fill : T.node;
          c.globalAlpha = p.fill ? 0.55 : 0.9;
          c.fill();
          c.globalAlpha = 1;
          c.stroke();
        }
        return { layer, depth };
      });

      bloom = document.createElement("canvas");
      bloom.width = Math.ceil(W * dpr);
      bloom.height = Math.ceil(H * dpr);
      const b = bloom.getContext("2d");
      b.scale(dpr, dpr);
      const glow = (x, y, radius, color, alpha) => {
        const grad = b.createRadialGradient(x, y, 0, x, y, radius);
        grad.addColorStop(0, color);
        grad.addColorStop(1, "rgba(255,255,255,0)");
        b.globalAlpha = alpha;
        b.fillStyle = grad;
        b.fillRect(0, 0, W, H);
      };
      const dim = T.dark ? 0.35 : 1;
      glow(W * 0.8, H * 0.28, W * 0.5, AQUA, 0.75 * dim);
      glow(W * 0.12, H * 0.92, W * 0.42, PERIWINKLE, 0.42 * dim);
      glow(W * 0.55, H * 0.02, W * 0.3, AQUA, 0.35 * dim);
      b.globalAlpha = 1;
    }

    function resize() {
      T = theme();
      const box = hero.getBoundingClientRect();
      W = box.width;
      H = box.height;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
      canvas.style.width = `${W}px`;
      canvas.style.height = `${H}px`;
      layout();
      paintGhosts();
      draw(0);
    }

    function position(name) {
      const a = anchors[name];
      if (reduceMotion.matches) return a;
      const drift = 5;
      const px = pointer ? (pointer.x / W - 0.5) * 10 : 0;
      const py = pointer ? (pointer.y / H - 0.5) * 6 : 0;
      return {
        x: a.x + Math.sin(clock * 0.45 + a.phase) * drift + px,
        y: a.y + Math.cos(clock * 0.37 + a.phase * 1.3) * drift + py,
      };
    }

    function curve(src, dst, index) {
      const mx = (src.x + dst.x) / 2;
      const my = (src.y + dst.y) / 2;
      const dx = dst.x - src.x;
      const dy = dst.y - src.y;
      const bend = (index % 2 ? 1 : -1) * 0.12;
      return { x0: src.x, y0: src.y, cx: mx - dy * bend, cy: my + dx * bend, x1: dst.x, y1: dst.y };
    }

    function at(c, u) {
      const v = 1 - u;
      return {
        x: v * v * c.x0 + 2 * v * u * c.cx + u * u * c.x1,
        y: v * v * c.y0 + 2 * v * u * c.cy + u * u * c.y1,
      };
    }

    function radius() {
      return Math.max(14, Math.min(22, W / 66));
    }

    /* The sharp world. `detail` adds the labels only the loupe can read. */
    function drawWorld(detail) {
      const t = Math.min(weeks - 1, Math.max(0, Math.round(week)));
      const pos = {};
      for (const name of world.nodes) pos[name] = position(name);
      const r = radius();

      world.edges.forEach((edge, i) => {
        const c = curve(pos[edge.src], pos[edge.dst], i);
        const weight = edge.coef == null ? 0.4 : Math.abs(edge.coef) / maxCoef;
        g.beginPath();
        g.moveTo(c.x0, c.y0);
        g.quadraticCurveTo(c.cx, c.cy, c.x1, c.y1);
        g.strokeStyle = EDGE[edge.type] || T.ink;
        g.globalAlpha = 0.62;
        g.lineWidth = 0.9 + weight * 1.6;
        g.setLineDash(edge.type === "by" ? [4, 4] : []);
        g.stroke();
        g.setLineDash([]);
        g.globalAlpha = 1;
        if (!reduceMotion.matches && edge.type !== "by") {
          for (const phase of particles[i]) {
            const u = (phase + clock * (0.06 + 0.12 * weight)) % 1;
            if (u < 0.12 || u > 0.88) continue;
            const p = at(c, u);
            g.beginPath();
            g.arc(p.x, p.y, 1.9, 0, Math.PI * 2);
            g.fillStyle = EDGE[edge.type];
            g.fill();
          }
        }
        if (detail && edge.coef != null) {
          const m = at(c, 0.5);
          g.font = `4.6px ${MONO}`;
          g.textAlign = "center";
          g.textBaseline = "middle";
          const label = `β ${edge.coef.toFixed(2)}`;
          const w = g.measureText(label).width + 3;
          g.fillStyle = T.node;
          g.fillRect(m.x - w / 2, m.y - 3.2, w, 6.4);
          g.fillStyle = T.ink;
          g.fillText(label, m.x, m.y + 0.2);
        }
      });

      for (const name of world.nodes) {
        const p = pos[name];
        const series = pulses[name];
        const swell = series ? 0.86 + 0.32 * series[t] : 1;
        const rr = r * swell;
        g.beginPath();
        g.arc(p.x, p.y, rr, 0, Math.PI * 2);
        g.fillStyle = name === "Y" ? PEACH : T.node;
        g.fill();
        g.lineWidth = 1.4;
        g.strokeStyle = T.ink;
        g.setLineDash(family(name) === "D" ? [3, 2.4] : []);
        g.stroke();
        g.setLineDash([]);
        g.fillStyle = name === "Y" ? NAVY : T.ink;
        g.font = `${Math.round(r * 0.62)}px ${MONO}`;
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(name, p.x, p.y + 0.5);
        if (detail) {
          const fam = family(name);
          const role = fam === "C" && world.mechanisms[name] ? world.mechanisms[name] : ROLE[fam];
          g.font = `4.8px ${MONO}`;
          g.fillStyle = T.ink2;
          g.fillText(role, p.x, p.y + rr + 5.5);
        }
      }
    }

    function drawRidge() {
      const band = Math.min(140, H * 0.16);
      const base = H - 10;
      const x = (i) => (i / (weeks - 1)) * W;
      const y = (i) => base - 12 - sales[i] * (band - 18);
      g.beginPath();
      g.moveTo(0, base);
      for (let i = 0; i < weeks; i++) g.lineTo(x(i), y(i));
      g.lineTo(W, base);
      g.closePath();
      const fill = g.createLinearGradient(0, base - band, 0, base);
      fill.addColorStop(0, "rgba(180,231,221,0.55)");
      fill.addColorStop(1, "rgba(180,231,221,0)");
      g.fillStyle = fill;
      g.fill();
      g.beginPath();
      for (let i = 0; i < weeks; i++) (i ? g.lineTo : g.moveTo).call(g, x(i), y(i));
      g.strokeStyle = T.ink;
      g.globalAlpha = 0.7;
      g.lineWidth = 1.2;
      g.stroke();
      g.globalAlpha = 1;

      const t = Math.min(weeks - 1, Math.max(0, Math.round(week)));
      const cx = x(t);
      g.beginPath();
      g.moveTo(cx, base - band - 6);
      g.lineTo(cx, base);
      g.strokeStyle = T.ghost(0.45);
      g.lineWidth = 1;
      g.stroke();
      g.beginPath();
      g.arc(cx, y(t), 3.6, 0, Math.PI * 2);
      g.fillStyle = PEACH;
      g.fill();
      g.strokeStyle = T.ink;
      g.stroke();
      g.font = `11px ${MONO}`;
      g.fillStyle = T.ink;
      g.textBaseline = "alphabetic";
      const label = `week ${t + 1} · sales ${world.series.Y[t].toFixed(2)}`;
      const flip = cx > W - 190;
      g.textAlign = flip ? "right" : "left";
      g.fillText(label, cx + (flip ? -8 : 8), base - band - 2);
    }

    function drawLens() {
      g.save();
      g.shadowColor = T.dark ? "rgba(0, 0, 0, 0.45)" : "rgba(12, 31, 64, 0.22)";
      g.shadowBlur = 28;
      g.shadowOffsetY = 10;
      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.fillStyle = T.ground;
      g.fill();
      g.restore();

      g.save();
      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.clip();
      g.translate(lens.x, lens.y);
      g.scale(ZOOM, ZOOM);
      g.translate(-lens.x, -lens.y);
      drawWorld(true);
      g.restore();

      g.beginPath();
      g.arc(lens.x, lens.y, LENS_R, 0, Math.PI * 2);
      g.strokeStyle = T.ink;
      g.lineWidth = 1;
      g.stroke();
      const tag = `×${ZOOM.toFixed(1)}`;
      g.font = `10px ${MONO}`;
      const w = g.measureText(tag).width + 12;
      g.fillStyle = T.ink;
      g.beginPath();
      g.roundRect(lens.x - w / 2, lens.y + LENS_R - 22, w, 15, 7.5);
      g.fill();
      g.fillStyle = T.ground;
      g.textAlign = "center";
      g.textBaseline = "middle";
      g.fillText(tag, lens.x, lens.y + LENS_R - 14.2);
    }

    function draw(dt) {
      clock += dt;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      g.fillStyle = T.ground;
      g.fillRect(0, 0, W, H);

      const drift = reduceMotion.matches ? 0 : 1;
      if (bloom) {
        g.drawImage(
          bloom,
          Math.sin(clock * 0.05) * 30 * drift,
          Math.cos(clock * 0.04) * 18 * drift,
          W,
          H,
        );
      }
      const px = pointer ? pointer.x / W - 0.5 : 0;
      const py = pointer ? pointer.y / H - 0.5 : 0;
      for (const { layer, depth } of ghosts) {
        const ox = -W * 0.1 + Math.sin(clock * 0.03 * (1 + depth)) * 24 * drift - px * 46 * depth;
        const oy = -H * 0.1 + Math.cos(clock * 0.025 * (1 + depth)) * 14 * drift - py * 28 * depth;
        g.drawImage(layer, ox, oy, W * 1.2, H * 1.2);
      }

      const idle = !pointer || performance.now() - lastPointer > 2600;
      if (idle) {
        week = reduceMotion.matches ? weeks - 1 : (week + dt * 7) % weeks;
      } else {
        const target = (pointer.x / W) * (weeks - 1);
        week += (target - week) * Math.min(1, dt * 10);
      }

      const r = region();
      const lensTarget =
        !idle && pointer.x > r.x0 - LENS_R && pointer.y < H * 0.82
          ? { x: pointer.x, y: pointer.y }
          : {
              x: (r.x0 + r.x1) / 2 + Math.sin(clock * 0.21) * (r.x1 - r.x0) * 0.36 * drift,
              y: (r.y0 + r.y1) / 2 + Math.sin(clock * 0.29 + 1.2) * (r.y1 - r.y0) * 0.32 * drift,
            };
      const follow = idle ? Math.min(1, dt * 2.5) : Math.min(1, dt * 18);
      lens.x += (lensTarget.x - lens.x) * (reduceMotion.matches && idle ? 1 : follow);
      lens.y += (lensTarget.y - lens.y) * (reduceMotion.matches && idle ? 1 : follow);

      drawRidge();
      drawWorld(false);
      if (W >= 600) drawLens();
    }

    function tick(now) {
      frame = 0;
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      draw(dt);
      if (visible && !document.hidden) frame = requestAnimationFrame(tick);
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
    cleanup(() => schemes.disconnect());

    on(hero, "pointermove", (event) => {
      const box = hero.getBoundingClientRect();
      pointer = { x: event.clientX - box.left, y: event.clientY - box.top };
      lastPointer = performance.now();
    });
    on(hero, "pointerleave", () => {
      pointer = null;
    });
    on(document, "visibilitychange", () => {
      if (!document.hidden && visible) start();
    });

    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => draw(0));
    resize();
    start();
    cleanup(() => {
      if (frame) cancelAnimationFrame(frame);
      observer.disconnect();
      sizer.disconnect();
    });
  }

  /* =========================================================================
     The story: code → data → graph → causes → proof, one sticky stage. */
  const SVG = "http://www.w3.org/2000/svg";
  const CODE = [
    "import pymc_generator as pg",
    "scenario = pg.SCENARIOS[1]",
    "prior = scenario.prior(n_time_steps=104, seed=0)",
    "world = pg.sample_scm(prior, seed=0,",
    "                      connect_all=scenario.connect_all)",
    "world.data                # spend, controls, sales",
    "world.identity_error()    # the decomposition check",
  ];

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG, name);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  }

  function mountStory(section, world, { on, cleanup, reduceMotion }) {
    const svg = section.querySelector("[data-pg-stage]");
    const progress = section.querySelectorAll(".pg-story__progress li");
    if (!svg) return;
    svg.replaceChildren();
    const N = world.weeks;
    const X0 = 150;
    const X1 = 770;
    const xs = Array.from({ length: N }, (_, i) => X0 + (i / (N - 1)) * (X1 - X0));

    const rows = ["C1", "C2", "C3", "C4", "Z1", "Z2", "Y"].filter((k) => world.series[k]);
    const lanes = [
      ["Y", "sales", world.series.Y],
      ["baseline", "baseline", world.decomposition.baseline],
      ...Object.keys(world.decomposition)
        .filter((k) => /^C\d+$/.test(k))
        .map((k) => [k, k, world.decomposition[k]]),
      ["indirect", "indirect", world.decomposition.indirect],
    ];
    const laneTop = (i) => 34 + i * (470 / lanes.length);
    const laneH = Math.min(48, 470 / lanes.length - 16);
    const rowTop = (i) => 40 + i * (490 / rows.length);
    const rowH = Math.min(48, 490 / rows.length - 14);

    const dag = {};
    {
      const byFamily = {};
      for (const name of world.nodes) (byFamily[family(name)] ||= []).push(name);
      const cols = { D: 130, Z: 310, C: 490, B: 680, Y: 680 };
      for (const [fam, names] of Object.entries(byFamily)) {
        names.forEach((name, i) => {
          let fy = (i + 1) / (names.length + 1);
          if (fam === "B") fy = 0.2;
          if (fam === "Y") fy = 0.62;
          dag[name] = { x: cols[fam], y: 50 + fy * 460 };
        });
      }
    }

    const flat = (y, x0, x1) => xs.map((_, i) => [x0 + (i / (N - 1)) * (x1 - x0), y]);
    const shaped = (values, top, h) => {
      const u = unit(values);
      return xs.map((x, i) => [x, top + (1 - u[i]) * h]);
    };
    const point = (p) => xs.map(() => [p.x, p.y]);

    const layer = (name) => el("g", { class: `pg-stage__${name}` }, svg);
    const codeLayer = layer("code");
    const edgeLayer = layer("edges");
    const laneLayer = layer("lanes");
    const strokeLayer = layer("strokes");
    const nodeLayer = layer("nodes");
    const proofLayer = layer("proof");

    /* Every animated thing is an item with per-state targets. */
    const items = [];
    const add = (item) => {
      item.cur = { ...item.states.code };
      if (item.states.code.pts) item.cur.pts = item.states.code.pts.map((p) => [...p]);
      items.push(item);
      return item;
    };
    const states = (fn) => Object.fromEntries(
      ["code", "series", "graph", "decomposition", "identity"].map((s) => [s, fn(s)]),
    );

    el("rect", { x: 40, y: 92, width: 720, height: CODE.length * 34 + 36, rx: 6, class: "pg-stage__plate" }, codeLayer);
    CODE.forEach((line, i) => {
      const text = el("text", { x: 64, y: 130 + i * 34, class: "pg-stage__code" }, codeLayer);
      text.textContent = line;
    });
    add({
      kind: "group",
      node: codeLayer,
      states: states((s) => ({ op: s === "code" ? 1 : 0 })),
    });

    rows.forEach((name, i) => {
      const length = CODE[i] ? CODE[i].replace(/^\s+/, "").length : 20;
      const indent = CODE[i] ? CODE[i].length - CODE[i].replace(/^\s+/, "").length : 0;
      const codePts = flat(136 + i * 34, 64 + indent * 8.4, 64 + (indent + length) * 8.4);
      const seriesPts = shaped(world.series[name], rowTop(i), rowH);
      const node = dag[name];
      const path = el("path", { class: `pg-stage__series pg-stage__series--${family(name)}` }, strokeLayer);
      add({
        kind: "path",
        node: path,
        states: states((s) => {
          if (s === "code") return { pts: codePts, op: 0 };
          if (s === "series") return { pts: seriesPts, op: 1 };
          return { pts: point(node), op: 0 };
        }),
      });
      const label = el("text", { class: "pg-stage__rowlabel", x: 0, y: 0 }, strokeLayer);
      label.textContent = { C: "spend", Z: "control", Y: "sales" }[family(name)] || "";
      add({
        kind: "text",
        node: label,
        states: states((s) => ({
          x: 112,
          y: rowTop(i) + rowH / 2 + 4,
          op: s === "series" ? 1 : 0,
        })),
      });
    });

    world.edges.forEach((edge, i) => {
      const a = dag[edge.src];
      const b = dag[edge.dst];
      const bend = (i % 2 ? 1 : -1) * 0.1;
      const d = `M${a.x},${a.y} Q${(a.x + b.x) / 2 - (b.y - a.y) * bend},${(a.y + b.y) / 2 + (b.x - a.x) * bend} ${b.x},${b.y}`;
      const path = el("path", { d, class: "pg-stage__edge", stroke: EDGE[edge.type] || NAVY }, edgeLayer);
      if (edge.type === "by") path.setAttribute("stroke-dasharray", "4 4");
      add({
        kind: edge.type === "by" ? "group" : "edge",
        node: path,
        states: states((s) => ({ op: s === "graph" ? 1 : 0, draw: s === "graph" ? 1 : 0 })),
      });
    });

    for (const name of world.nodes) {
      const group = el("g", { class: `pg-stage__node pg-stage__node--${family(name)}` }, nodeLayer);
      el("circle", { r: 17 }, group);
      const text = el("text", { "text-anchor": "middle", dy: "0.35em" }, group);
      text.textContent = name;
      const rowIndex = rows.indexOf(name);
      add({
        kind: "node",
        node: group,
        states: states((s) => {
          if (s === "series" && rowIndex >= 0) {
            return { x: 44, y: rowTop(rowIndex) + rowH / 2, scale: 0.6, op: 1 };
          }
          if (s === "graph") return { x: dag[name].x, y: dag[name].y, scale: 1, op: 1 };
          if (s === "code" && rowIndex >= 0) return { x: 40, y: 136 + rowIndex * 34, scale: 0.3, op: 0 };
          return { x: dag[name].x, y: dag[name].y, scale: 0.6, op: 0 };
        }),
      });
    }

    const axis = el("line", { class: "pg-stage__axis", x1: 0, x2: 0, y1: 20, y2: 540 }, laneLayer);
    const axisWeek = Math.round(N * 0.62);
    add({
      kind: "line",
      node: axis,
      states: states((s) => ({ x: xs[axisWeek], op: s === "decomposition" ? 1 : 0 })),
    });
    const salesPts = shaped(world.series.Y, laneTop(0), laneH);
    lanes.forEach(([key, label, values], i) => {
      const own = shaped(values, laneTop(i), laneH);
      const kind = i === 0 ? "sales" : /^C\d+$/.test(key) ? "C" : key;
      const path = el("path", { class: `pg-stage__lane pg-stage__lane--${kind}` }, laneLayer);
      add({
        kind: "path",
        node: path,
        states: states((s) => {
          if (s === "decomposition") return { pts: own, op: 1 };
          if (s === "identity") return { pts: salesPts, op: i === 0 ? 1 : 0.35 };
          return { pts: salesPts, op: 0 };
        }),
      });
      const text = el("text", { class: "pg-stage__lanelabel", x: 0, y: 0 }, laneLayer);
      text.textContent = label;
      add({
        kind: "text",
        node: text,
        states: states((s) => ({
          x: 132,
          y: laneTop(i) + laneH / 2 + 4,
          op: s === "decomposition" || (s === "identity" && i === 0) ? 1 : 0,
        })),
      });
    });

    const residual = el("path", { class: "pg-stage__residual" }, proofLayer);
    add({
      kind: "path",
      node: residual,
      states: states((s) => ({ pts: flat(s === "identity" ? 300 : laneTop(0) + laneH, X0, X1), op: s === "identity" ? 1 : 0 })),
    });
    const residualLabel = el("text", { class: "pg-stage__lanelabel", x: 132, y: 304 }, proofLayer);
    residualLabel.textContent = "sales − Σ parts";
    const number = el("text", { class: "pg-stage__number", x: X0, y: 420 }, proofLayer);
    const [mantissa, exponent] = world.identity_error.toExponential(2).split("e");
    number.textContent = `${mantissa} × 10`;
    const sup = el("tspan", { "baseline-shift": "super", class: "pg-stage__exp" }, number);
    sup.textContent = String(Number(exponent));
    const caption = el("text", { class: "pg-stage__caption", x: X0, y: 456 }, proofLayer);
    caption.textContent = `largest absolute error over all ${N} weeks, computed while this page was built`;
    add({
      kind: "group",
      node: proofLayer,
      states: states((s) => ({ op: s === "identity" ? 1 : 0 })),
    });

    /* Render the current values of every item. */
    const pathD = (pts) => {
      let d = "";
      for (let i = 0; i < pts.length; i++) d += `${i ? "L" : "M"}${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)}`;
      return d;
    };
    function render() {
      for (const item of items) {
        const c = item.cur;
        item.node.style.opacity = c.op.toFixed(3);
        item.node.style.visibility = c.op < 0.01 ? "hidden" : "visible";
        if (item.kind === "path") item.node.setAttribute("d", pathD(c.pts));
        if (item.kind === "text") {
          item.node.setAttribute("x", c.x.toFixed(1));
          item.node.setAttribute("y", c.y.toFixed(1));
        }
        if (item.kind === "line") {
          item.node.setAttribute("x1", c.x.toFixed(1));
          item.node.setAttribute("x2", c.x.toFixed(1));
        }
        if (item.kind === "node") {
          item.node.setAttribute("transform", `translate(${c.x.toFixed(1)} ${c.y.toFixed(1)}) scale(${c.scale.toFixed(3)})`);
        }
        if (item.kind === "edge") {
          const length = item.length || (item.length = item.node.getTotalLength());
          item.node.style.strokeDasharray = `${length}`;
          item.node.style.strokeDashoffset = `${(1 - c.draw) * length}`;
        }
      }
    }

    let current = "code";
    let frame = 0;
    function go(state) {
      if (state === current) return;
      current = state;
      progress.forEach((li) => {
        const order = [...progress].map((x) => x.dataset.step);
        const here = order.indexOf(state);
        const mine = order.indexOf(li.dataset.step);
        li.classList.toggle("is-current", mine === here);
        li.classList.toggle("is-before", mine < here);
      });
      section.dataset.state = state;
      const from = items.map((item) => ({
        ...item.cur,
        pts: item.cur.pts ? item.cur.pts.map((p) => [...p]) : undefined,
      }));
      const to = items.map((item) => item.states[state]);
      const duration = reduceMotion.matches ? 0 : 1100;
      const begin = performance.now();
      if (frame) cancelAnimationFrame(frame);
      const step = (now) => {
        const raw = duration ? Math.min(1, (now - begin) / duration) : 1;
        items.forEach((item, k) => {
          const a = from[k];
          const b = to[k];
          /* Stagger points left to right so lines appear to be drawn across. */
          for (const key of ["op", "x", "y", "scale", "draw"]) {
            if (b[key] !== undefined) item.cur[key] = a[key] + (b[key] - a[key]) * easeOut(raw);
          }
          if (b.pts) {
            const n = b.pts.length;
            for (let i = 0; i < n; i++) {
              const lag = (i / n) * 0.35;
              const local = Math.min(1, Math.max(0, (raw - lag) / 0.65));
              const e = easeOut(local);
              item.cur.pts[i][0] = a.pts[i][0] + (b.pts[i][0] - a.pts[i][0]) * e;
              item.cur.pts[i][1] = a.pts[i][1] + (b.pts[i][1] - a.pts[i][1]) * e;
            }
          }
        });
        render();
        frame = raw < 1 ? requestAnimationFrame(step) : 0;
      };
      frame = requestAnimationFrame(step);
    }

    render();
    section.dataset.state = "code";
    progress.forEach((li, i) => li.classList.toggle("is-current", i === 0));
    const steps = section.querySelectorAll(".pg-story__step");
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) if (entry.isIntersecting) go(entry.target.dataset.step);
      },
      /* On narrow screens the stage sits on top, so steps trigger lower down. */
      {
        rootMargin: window.matchMedia("(max-width: 59.984375em)").matches
          ? "-68% 0px -31% 0px"
          : "-45% 0px -45% 0px",
      },
    );
    steps.forEach((s) => observer.observe(s));
    cleanup(() => {
      observer.disconnect();
      if (frame) cancelAnimationFrame(frame);
    });
  }

  function mount(context) {
    const world = readWorld();
    if (!world) return;
    const field = document.querySelector("[data-pg-field]");
    if (field && field.getContext) mountField(field, world, context);
    const story = document.querySelector("[data-pg-story]");
    if (story) mountStory(story, world, context);
  }

  if (window.pymcGenerator) window.pymcGenerator.register(mount);
})();
