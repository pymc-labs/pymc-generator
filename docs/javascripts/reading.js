/* Reading instruments shared by every page: the plate loupe, the source viewer,
   the reading rail (contents progress plus a resume mark), and the header read
   line.
   Material's instant navigation swaps the page body without reloading, so every
   instrument is mounted per page through `document$` and torn down before the
   next one. Nothing here is required to read the page. */

(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  let teardown = [];

  const on = (target, type, handler, options) => {
    target.addEventListener(type, handler, options);
    teardown.push(() => target.removeEventListener(type, handler, options));
  };

  /* ---------------------------------------------------------------------------
     Loupe: a circular magnifier over figures. It clones the figure's own
     vectors (or image) at a higher scale, so what you see under the glass is
     the figure itself, not a blurred enlargement of its pixels. */
  const LOUPE_SIZE = 184;
  const LOUPE_ZOOM = 2.4;

  function mediaOf(host) {
    return host.querySelector(":scope > svg, :scope > img, :scope img, :scope svg");
  }

  function mountLoupe(host) {
    const media = mediaOf(host);
    if (!media) return;
    host.classList.add("pg-loupe-host");
    let loupe = null;
    let clone = null;
    let frame = 0;
    let point = null;

    const build = () => {
      const box = media.getBoundingClientRect();
      if (box.width < 240) return false;
      loupe = document.createElement("div");
      loupe.className = "pg-loupe";
      loupe.setAttribute("aria-hidden", "true");
      clone = media.cloneNode(true);
      clone.removeAttribute("id");
      clone.removeAttribute("role");
      clone.removeAttribute("aria-label");
      clone.style.width = `${box.width * LOUPE_ZOOM}px`;
      clone.style.height = `${box.height * LOUPE_ZOOM}px`;
      clone.style.maxWidth = "none";
      const reading = document.createElement("span");
      reading.className = "pg-loupe__reading";
      reading.textContent = `×${LOUPE_ZOOM.toFixed(1)}`;
      loupe.append(clone, reading);
      host.append(loupe);
      return true;
    };

    const paint = () => {
      frame = 0;
      if (!loupe || !point) return;
      const box = media.getBoundingClientRect();
      const hostBox = host.getBoundingClientRect();
      const x = point.x - box.left;
      const y = point.y - box.top;
      const inside = x >= 0 && y >= 0 && x <= box.width && y <= box.height;
      loupe.classList.toggle("is-visible", inside);
      if (!inside) return;
      const left = point.x - hostBox.left - LOUPE_SIZE / 2;
      const top = point.y - hostBox.top - LOUPE_SIZE / 2;
      loupe.style.transform = `translate(${left}px, ${top}px)`;
      clone.style.transform =
        `translate(${LOUPE_SIZE / 2 - x * LOUPE_ZOOM}px, ${LOUPE_SIZE / 2 - y * LOUPE_ZOOM}px)`;
    };

    on(host, "pointerenter", (event) => {
      if (event.pointerType !== "mouse" || !finePointer.matches) return;
      if (!loupe && !build()) return;
      point = { x: event.clientX, y: event.clientY };
      paint();
    });
    on(host, "pointermove", (event) => {
      if (!loupe || event.pointerType !== "mouse") return;
      point = { x: event.clientX, y: event.clientY };
      if (!frame) frame = requestAnimationFrame(paint);
    });
    on(host, "pointerleave", () => {
      point = null;
      if (loupe) loupe.classList.remove("is-visible");
    });
    teardown.push(() => {
      if (frame) cancelAnimationFrame(frame);
      if (loupe) loupe.remove();
    });
  }

  function mountLoupes(root) {
    root
      .querySelectorAll("figure.scm-plot, figure.pg-figure, .jp-RenderedImage, [data-pg-loupe]")
      .forEach(mountLoupe);
  }

  /* ---------------------------------------------------------------------------
     Source viewer: an API page's "Source code in …" rows open their code in a
     modal dialog over the blurred page instead of expanding in place. The code
     block itself moves into the dialog and back on close, so its line anchors
     and copy button keep working; the page behind stays put while the code
     scrolls. Without the script the rows expand as plain details elements. */
  const CLOSE_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>';

  function mountSources(root) {
    const rows = [...root.querySelectorAll("details.mkdocstrings-source")];
    const article = root.querySelector(".md-content__inner");
    if (!rows.length || !article || typeof HTMLDialogElement !== "function") return;

    const dialog = document.createElement("dialog");
    dialog.className = "pg-source";
    dialog.setAttribute("aria-labelledby", "pg-source-title");
    dialog.innerHTML =
      '<div class="pg-source__frame">' +
      '<div class="pg-source__bar">' +
      '<p class="pg-source__title" id="pg-source-title"></p>' +
      `<button type="button" class="pg-source__close" aria-label="Close source code">${CLOSE_ICON}</button>` +
      "</div>" +
      '<div class="pg-source__body" tabindex="0" aria-label="Source code"></div>' +
      "</div>";
    article.append(dialog);
    const title = dialog.querySelector(".pg-source__title");
    const body = dialog.querySelector(".pg-source__body");
    let shown = null;

    /* The object the code belongs to, from its anchor id with the module path
       of the source file stripped: `sample_scm`, `SCM.equations`. */
    function objectName(row, file) {
      const object = row.closest(".doc-object");
      const anchor = object?.querySelector(":scope > a[id], :scope > :is(h2, h3, h4, h5, h6)[id]");
      if (!anchor) return "";
      const module = file
        ? file.textContent.trim().replace(/(\/__init__)?\.py$/, "").replaceAll("/", ".")
        : "";
      return module && anchor.id.startsWith(`${module}.`)
        ? anchor.id.slice(module.length + 1)
        : anchor.id.split(".").pop();
    }

    function open(row) {
      const code = row.querySelector(":scope > .highlight");
      if (!code || shown) return;
      row.open = false;
      const file = row.querySelector(":scope > summary code");
      const name = objectName(row, file);
      title.replaceChildren();
      if (name) {
        const span = document.createElement("span");
        span.className = "pg-source__name";
        span.textContent = name;
        title.append(span);
      }
      if (file) title.append(file.cloneNode(true));
      body.append(code);
      shown = { row, code };
      dialog.showModal();
      body.scrollTop = 0;
      body.focus({ preventScroll: true });
    }

    on(dialog, "close", () => {
      if (!shown) return;
      const { row, code } = shown;
      shown = null;
      row.append(code);
      row.querySelector(":scope > summary").focus({ preventScroll: true });
    });
    on(dialog.querySelector(".pg-source__close"), "click", () => dialog.close());
    // The frame fills the dialog, so a click landing on the dialog itself is
    // a click on the blurred page around it.
    on(dialog, "click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    // Line-number links scroll the code, not the locked page behind it. The
    // listener runs at the window, in capture, ahead of the site's instant
    // navigation, which would otherwise scroll the page to the line.
    on(
      window,
      "click",
      (event) => {
        // Material rewrites in-page links to absolute URLs, so match on the hash.
        const link = event.target instanceof Element && event.target.closest("a[href]");
        if (!link || !link.hash || !body.contains(link)) return;
        const target = body.querySelector(`[id="${CSS.escape(decodeURIComponent(link.hash.slice(1)))}"]`);
        if (!target) return;
        event.preventDefault();
        event.stopPropagation();
        const offset = target.getBoundingClientRect().top - body.getBoundingClientRect().top;
        body.scrollTo({ top: body.scrollTop + offset - body.clientHeight / 3, behavior: "auto" });
      },
      true,
    );

    rows.forEach((row) => {
      const summary = row.querySelector(":scope > summary");
      if (!summary || !row.querySelector(":scope > .highlight")) return;
      row.classList.add("pg-source-row");
      summary.setAttribute("aria-haspopup", "dialog");
      on(summary, "click", (event) => {
        event.preventDefault();
        open(row);
      });
    });

    teardown.push(() => {
      if (dialog.open) dialog.close();
      dialog.remove();
    });
  }

  /* ---------------------------------------------------------------------------
     Reading position: the header read line, the rail's fill, and a resume mark
     that remembers where you stopped on this page. */
  const STORE = "pymc-generator:reading:";

  function readFraction() {
    const doc = document.documentElement;
    const max = doc.scrollHeight - window.innerHeight;
    return max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 1;
  }

  function currentHeading(article) {
    let found = null;
    for (const heading of article.querySelectorAll("h2[id], h3[id]")) {
      if (heading.getBoundingClientRect().top - 120 > 0) break;
      found = heading;
    }
    return found;
  }

  function mountReading() {
    const article = document.querySelector(".md-content__inner");
    const root = document.documentElement;
    if (!article) return;
    const key = STORE + location.pathname;
    /* The home page is a showcase, not a document to resume. */
    const resumable = !document.querySelector(".pg-hero");
    let frame = 0;
    let saveTimer = 0;

    const paint = () => {
      frame = 0;
      root.style.setProperty("--pg-read", readFraction().toFixed(4));
    };
    const save = () => {
      saveTimer = 0;
      const fraction = readFraction();
      try {
        if (fraction > 0.97) {
          localStorage.removeItem(key);
        } else if (window.scrollY > 400) {
          const heading = currentHeading(article);
          localStorage.setItem(
            key,
            JSON.stringify({
              y: Math.round(window.scrollY),
              id: heading ? heading.id : null,
              title: heading ? heading.textContent.replace("¶", "").trim() : null,
            }),
          );
        }
      } catch {
        /* storage can be unavailable (private mode); the mark is optional */
      }
    };

    on(window, "scroll", () => {
      if (!frame) frame = requestAnimationFrame(paint);
      if (resumable && !saveTimer) saveTimer = window.setTimeout(save, 600);
    }, { passive: true });
    on(window, "resize", paint, { passive: true });
    paint();
    teardown.push(() => {
      if (frame) cancelAnimationFrame(frame);
      if (saveTimer) window.clearTimeout(saveTimer);
    });

    let saved = null;
    try {
      saved = JSON.parse(localStorage.getItem(key) || "null");
    } catch {
      saved = null;
    }
    if (!resumable || !saved || location.hash || window.scrollY > 200) return;
    const mark = document.createElement("button");
    mark.type = "button";
    mark.className = "pg-resume";
    const where = saved.title ? `Resume at “${saved.title}”` : "Resume where you stopped";
    mark.innerHTML = '<span class="pg-resume__flag" aria-hidden="true"></span><span></span>';
    mark.lastChild.textContent = where;
    const rail = document.querySelector(".md-sidebar--secondary .md-nav--secondary");
    (rail && rail.offsetParent ? rail : document.body).prepend(mark);
    if (!(rail && rail.offsetParent)) mark.classList.add("pg-resume--floating");
    const dismiss = () => mark.remove();
    on(mark, "click", () => {
      const target = saved.id ? document.getElementById(saved.id) : null;
      const behavior = reduceMotion.matches ? "auto" : "smooth";
      if (target) target.scrollIntoView({ behavior, block: "start" });
      else window.scrollTo({ top: saved.y, behavior });
      dismiss();
    });
    const leave = () => {
      if (window.scrollY > 600) dismiss();
    };
    on(window, "scroll", leave, { passive: true });
    teardown.push(dismiss);
  }

  /* ---------------------------------------------------------------------------
     Mount per page. */
  const modules = [];
  let context = null;
  /* Page modules (the home page's field and story) register here; they run on
     every mount, and immediately if they arrive after the first one. */
  window.pymcGenerator = {
    register(fn) {
      modules.push(fn);
      if (context) fn(context);
    },
  };

  function mount() {
    teardown.forEach((undo) => undo());
    teardown = [];
    mountLoupes(document);
    mountSources(document);
    mountReading();
    context = { on, cleanup: (undo) => teardown.push(undo), reduceMotion };
    modules.forEach((fn) => fn(context));
  }

  if (window.document$) window.document$.subscribe(mount);
  else document.addEventListener("DOMContentLoaded", mount);
})();
