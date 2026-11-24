/* Industrial parts QC demo UI — vanilla JS, no framework, no build step
 * Notes: the detect API handles one image per call; the line mode takt is simulated by a front-end timer.
 * Counters live on the server: a page reload restores them from /api/state (cumulative).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = {
    lineState: $("lineState"), category: $("category"),
    thresholdVal: $("thresholdVal"), deviceVal: $("deviceVal"), modelMeta: $("modelMeta"),
    modeLine: $("modeLine"), modeCamera: $("modeCamera"),
    btnStart: $("btnStart"), btnPause: $("btnPause"), btnStep: $("btnStep"),
    btnShoot: $("btnShoot"), btnReset: $("btnReset"),
    interval: $("interval"), intervalLabel: $("intervalLabel"),
    autoWrap: $("autoWrap"), autoCapture: $("autoCapture"), loopBadge: $("loopBadge"),
    origImg: $("origImg"), overlayImg: $("overlayImg"), video: $("video"),
    imgPlaceholder: $("imgPlaceholder"), overlayPlaceholder: $("overlayPlaceholder"),
    origMeta: $("origMeta"),
    verdict: $("verdict"), lamp: $("lamp"), verdictText: $("verdictText"),
    scoreBig: $("scoreBig"), scoreFill: $("scoreFill"), thrLine: $("thrLine"),
    latencyVal: $("latencyVal"), sourceVal: $("sourceVal"), posVal: $("posVal"), hint: $("hint"),
    statTotal: $("statTotal"), statOk: $("statOk"), statNg: $("statNg"),
    statRate: $("statRate"), statLatency: $("statLatency"),
    trendCanvas: $("trendCanvas"), trendMeta: $("trendMeta"),
    thumbs: $("thumbs"), histBody: $("histBody"), histMeta: $("histMeta"),
    toast: $("toast"), captureCanvas: $("captureCanvas"),
  };

  const state = {
    mode: "line",
    running: false,
    busy: false,
    intervalS: 2.0,
    category: null,
    threshold: 0,
    device: "",
    demoTotal: 0,
    timer: null,
    cameraStream: null,
    history: [],
    points: [],
    counters: { total: 0, ok: 0, ng: 0, defect_rate: 0, avg_latency_ms: 0 },
  };

  // ---------------------------------------------------------------- Utilities
  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.detail || msg; } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    return res.json();
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    el.toast.textContent = message;
    el.toast.className = "toast" + (isError ? " error" : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.toast.classList.add("hidden"), 4500);
  }

  function fmtScore(v) { return Number(v).toFixed(3); }

  function setRunningUI() {
    el.lineState.textContent = state.running ? "● Running" : "● Paused";
    el.lineState.className = "state " + (state.running ? "running" : "paused");
    el.btnStart.disabled = state.running;
    el.btnPause.disabled = !state.running;
    el.hint.textContent = state.running
      ? (state.mode === "line" ? "Feeding parts at takt time…" : "Auto-capturing from camera…")
      : "Press \"Start\" to feed parts at takt time";
  }

  // ---------------------------------------------------------------- State / boot
  async function boot() {
    bindEvents();
    try {
      const st = await api("/api/state");
      applyState(st);
      restoreFromServer(st);
      state.intervalS = st.default_interval_s || 2.0;
      el.interval.value = String(state.intervalS);
      el.intervalLabel.textContent = state.intervalS.toFixed(1) + " s";
      el.hint.textContent = "Press \"Start\" to feed parts at takt time";
      drawTrend();
    } catch (err) {
      toast("Initialization failed: " + err.message, true);
    }
  }

  function applyState(st) {
    state.category = st.default_category;
    state.threshold = Number(st.threshold);
    state.device = st.device;
    state.demoTotal = st.demo_total || 0;

    el.category.innerHTML = "";
    st.categories.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c; opt.textContent = c;
      if (c === st.default_category) opt.selected = true;
      el.category.appendChild(opt);
    });

    el.thresholdVal.textContent = state.threshold.toFixed(4);
    el.deviceVal.textContent = st.device;
    el.modelMeta.textContent = `${st.categories.length} models · ${st.demo_source}`;
    const thrPct = (1 / 1.8) * 100;
    el.thrLine.style.left = Math.min(100, thrPct).toFixed(1) + "%";
    el.trendMeta.textContent = `Threshold ${state.threshold.toFixed(3)} · ${state.demoTotal} demo images`;
  }

  function restoreFromServer(st) {
    const c = st.counters || {};
    state.counters = {
      total: c.total || 0, ok: c.ok || 0, ng: c.ng || 0,
      defect_rate: c.defect_rate || 0, avg_latency_ms: c.avg_latency_ms || 0,
    };
    state.history = (c.history || []).slice();
    state.points = (c.scores || []).slice(-50).map((s) => ({ score: s, ng: s >= state.threshold }));
    renderCounters();
    renderHistory();
    if (state.history.length) renderLast(state.history[0]);
  }

  /** After a reload, restore the last result into the main stage (counters keep accumulating) */
  function renderLast(h) {
    if (!h || !h.image_url) return;
    el.origImg.src = h.image_url;
    el.origImg.classList.remove("hidden");
    el.imgPlaceholder.classList.add("hidden");
    el.overlayImg.src = h.overlay_url;
    el.overlayImg.classList.remove("hidden");
    el.overlayPlaceholder.classList.add("hidden");
    const ng = h.decision === "NG";
    el.lamp.className = "lamp " + (ng ? "ng" : "ok");
    el.verdictText.className = "verdict-text " + (ng ? "ng" : "ok");
    el.verdictText.textContent = ng ? "NG · Failed" : "OK · Passed";
    el.scoreBig.textContent = fmtScore(h.score);
    renderScoreBar(Number(h.score));
    el.latencyVal.textContent = `${Number(h.latency_ms).toFixed(1)} ms`;
    el.sourceVal.textContent = `${h.source} / ${h.file}`;
    el.origMeta.textContent = `${h.category} · ${h.time}`;
    el.posVal.textContent = "—";
    el.hint.textContent = "Last inspection result restored from server";
  }

  // ---------------------------------------------------------------- Rendering
  function renderCounters() {
    const c = state.counters;
    el.statTotal.textContent = c.total;
    el.statOk.textContent = c.ok;
    el.statNg.textContent = c.ng;
    el.statRate.textContent = (c.defect_rate || 0).toFixed(1) + "%";
    el.statLatency.textContent = c.total ? `${(c.avg_latency_ms || 0).toFixed(0)} ms` : "--";
  }

  function renderScoreBar(score) {
    const max = Math.max(state.threshold * 1.8, score * 1.08, 1);
    const pct = Math.min(100, (score / max) * 100);
    el.scoreFill.style.width = pct.toFixed(1) + "%";
    const thrPct = Math.min(100, (state.threshold / max) * 100);
    el.thrLine.style.left = thrPct.toFixed(1) + "%";
  }

  function ngFlash() {
    el.verdict.classList.remove("ng-flash");
    void el.verdict.offsetWidth; // force reflow so the animation can retrigger
    el.verdict.classList.add("ng-flash");
  }

  function renderResult(r) {
    state.threshold = Number(r.threshold || state.threshold);
    const ng = r.decision === "NG";

    el.origImg.src = r.image_url;
    el.overlayImg.src = r.overlay_url;
    if (state.mode !== "camera") {
      el.origImg.classList.remove("hidden");
      el.imgPlaceholder.classList.add("hidden");
    }
    el.overlayImg.classList.remove("hidden");
    el.overlayPlaceholder.classList.add("hidden");

    el.lamp.className = "lamp " + (ng ? "ng" : "ok");
    el.verdictText.className = "verdict-text " + (ng ? "ng" : "ok");
    el.verdictText.textContent = ng ? "NG · Failed" : "OK · Passed";
    el.verdict.classList.toggle("ng-active", ng);
    if (ng) ngFlash();

    el.scoreBig.textContent = fmtScore(r.score);
    renderScoreBar(Number(r.score));
    el.latencyVal.textContent = `${Number(r.latency_ms).toFixed(1)} ms`;
    el.sourceVal.textContent = `${r.source} / ${r.file}`;
    el.origMeta.textContent = `${r.category} · ${r.time}`;
    if (typeof r.position === "number") {
      el.posVal.textContent = `${r.position} / ${r.demo_total}`;
    } else {
      el.posVal.textContent = state.mode === "camera" ? "Camera capture" : "—";
    }

    // loop badge
    if (r.looped) {
      el.loopBadge.classList.remove("hidden");
      setTimeout(() => el.loopBadge.classList.add("hidden"), 3500);
    }

    // counters and history are authoritative on the server
    if (r.counters) {
      state.counters = r.counters;
      state.points = (r.counters.scores || []).slice(-50).map((s) => ({ score: s, ng: s >= state.threshold }));
      state.history = (r.counters.history || []).slice();
    } else {
      state.counters.total += 1;
      state.counters[ng ? "ng" : "ok"] += 1;
      state.counters.defect_rate = state.counters.total ? (state.counters.ng / state.counters.total) * 100 : 0;
      state.points.push({ score: Number(r.score), ng });
      if (state.points.length > 50) state.points.shift();
      state.history.unshift({ ...r });
    }
    renderCounters();
    renderHistory();
    drawTrend();
    el.hint.textContent = ng ? "Defect detected — please review this part" : "Part passed, waiting for the next one…";
  }

  function renderHistory() {
    // thumbnails (latest 12)
    const thumbs = state.history.slice(0, 12);
    el.thumbs.innerHTML = "";
    if (!thumbs.length) {
      const d = document.createElement("div");
      d.className = "empty";
      d.textContent = "No inspections yet";
      el.thumbs.appendChild(d);
    } else {
      thumbs.forEach((h) => {
        if (!h.overlay_url) return;
        const img = document.createElement("img");
        img.src = h.overlay_url;
        img.className = h.decision === "NG" ? "ng" : "ok";
        img.title = `${h.file} | ${fmtScore(h.score)} | ${h.decision}`;
        img.addEventListener("click", () => {
          if (h.image_url) el.origImg.src = h.image_url;
          el.overlayImg.src = h.overlay_url;
          el.origMeta.textContent = `${h.category} · ${h.time}`;
          el.overlayPlaceholder.classList.add("hidden");
          el.imgPlaceholder.classList.add("hidden");
        });
        el.thumbs.appendChild(img);
      });
    }

    // table (latest 30)
    el.histBody.innerHTML = "";
    state.history.slice(0, 30).forEach((h) => {
      const tr = document.createElement("tr");
      tr.className = h.decision === "NG" ? "ng" : "ok";
      const cells = [h.time, h.category, h.source, h.file, fmtScore(h.score), h.decision,
                     `${Number(h.latency_ms).toFixed(1)} ms`];
      cells.forEach((text, i) => {
        const td = document.createElement("td");
        td.textContent = text;
        if (i === 4 || i === 6) td.className = "num";
        tr.appendChild(td);
      });
      el.histBody.appendChild(tr);
    });
    el.histMeta.textContent = `Last ${Math.min(30, state.history.length)} · ${state.counters.total} total`;
  }

  function drawTrend() {
    const canvas = el.trendCanvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, rect.width), h = Math.max(120, rect.height);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pts = state.points;
    const M = { l: 40, r: 12, t: 12, b: 20 };
    const plotW = w - M.l - M.r, plotH = h - M.t - M.b;
    const thr = state.threshold;
    let lo = thr * 0.7, hi = thr * 1.25;
    if (pts.length) {
      const vals = pts.map((p) => p.score);
      lo = Math.min(lo, Math.min(...vals) * 0.97);
      hi = Math.max(hi, Math.max(...vals) * 1.04);
    }
    const y = (v) => M.t + plotH - ((v - lo) / (hi - lo || 1)) * plotH;

    // grid + y axis
    ctx.font = "10px Segoe UI, system-ui, sans-serif";
    ctx.fillStyle = "#5f7699";
    ctx.strokeStyle = "rgba(31,47,71,.75)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const v = lo + ((hi - lo) * i) / 4;
      const yy = y(v);
      ctx.beginPath(); ctx.moveTo(M.l, yy); ctx.lineTo(w - M.r, yy); ctx.stroke();
      ctx.fillText(v.toFixed(1), 6, yy + 3);
    }

    // threshold line
    const yt = y(thr);
    ctx.strokeStyle = "#f59e0b";
    ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(M.l, yt); ctx.lineTo(w - M.r, yt); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f59e0b";
    ctx.fillText("Threshold " + thr.toFixed(2), M.l + 4, yt - 4);

    if (!pts.length) {
      ctx.fillStyle = "#405a7a";
      ctx.font = "12px Segoe UI, system-ui, sans-serif";
      ctx.fillText("No data yet — start inspecting to plot the last 50 scores", M.l + 10, M.t + plotH / 2);
      return;
    }

    // polyline
    const x = (i) => (pts.length === 1 ? M.l + plotW / 2 : M.l + (i / (pts.length - 1)) * plotW);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    pts.forEach((p, i) => { i ? ctx.lineTo(x(i), y(p.score)) : ctx.moveTo(x(i), y(p.score)); });
    ctx.stroke();

    // data points
    pts.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(x(i), y(p.score), 3, 0, Math.PI * 2);
      ctx.fillStyle = p.ng ? "#ef4444" : "#22c55e";
      ctx.fill();
    });
  }

  // ---------------------------------------------------------------- Inspection flow
  async function tick() {
    if (state.busy) return;
    state.busy = true;
    try {
      const r = state.mode === "camera" ? await captureAndDetect() : await nextDemo();
      if (r) renderResult(r);
    } catch (err) {
      toast("Inspection failed: " + err.message, true);
      pause();
    } finally {
      state.busy = false;
      if (state.running) schedule();
    }
  }

  async function nextDemo() {
    return api("/api/demo/next", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category: state.category }),
    });
  }

  async function captureAndDetect() {
    if (!state.cameraStream) throw new Error("Camera not ready");
    const video = el.video;
    const canvas = el.captureCanvas;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    if (!blob) throw new Error("Capture failed");
    const fd = new FormData();
    fd.append("file", blob, `camera_${Date.now()}.jpg`);
    fd.append("category", state.category);
    fd.append("source", "camera");
    return api("/api/detect", { method: "POST", body: fd });
  }

  function schedule() {
    clearTimeout(state.timer);
    state.timer = setTimeout(tick, state.intervalS * 1000);
  }

  function start() {
    if (state.running) return;
    el.toast.classList.add("hidden");
    state.running = true;
    setRunningUI();
    if (state.mode === "camera" && !state.cameraStream) initCamera().catch(() => {});
    tick();
  }

  function pause() {
    state.running = false;
    clearTimeout(state.timer);
    setRunningUI();
  }

  async function step() {
    if (state.busy) return;
    if (state.running) pause();
    await tick();
  }

  async function reset() {
    try {
      const c = await api("/api/reset", { method: "POST" });
      state.counters = { total: c.total, ok: c.ok, ng: c.ng, defect_rate: c.defect_rate, avg_latency_ms: c.avg_latency_ms };
      state.history = [];
      state.points = [];
      el.loopBadge.classList.add("hidden");
      el.lamp.className = "lamp";
      el.verdictText.className = "verdict-text";
      el.verdictText.textContent = "Waiting";
      el.verdict.classList.remove("ng-active", "ng-flash");
      el.scoreBig.textContent = "--";
      el.scoreFill.style.width = "0%";
      el.latencyVal.textContent = "--";
      el.sourceVal.textContent = "--";
      el.posVal.textContent = "--";
      el.origImg.removeAttribute("src");
      el.overlayImg.removeAttribute("src");
      el.origImg.classList.add("hidden");
      el.overlayImg.classList.add("hidden");
      el.imgPlaceholder.classList.remove("hidden");
      el.overlayPlaceholder.classList.remove("hidden");
      renderCounters();
      renderHistory();
      drawTrend();
      toast("Statistics reset — previous log archived to outputs/live/");
    } catch (err) {
      toast("Reset failed: " + err.message, true);
    }
  }

  // ---------------------------------------------------------------- Camera
  async function initCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API unavailable in this browser (requires HTTPS or localhost)");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    state.cameraStream = stream;
    el.video.srcObject = stream;
    el.video.hidden = false;
    el.origImg.classList.add("hidden");
    el.imgPlaceholder.classList.add("hidden");
    el.overlayPlaceholder.classList.remove("hidden");
    el.btnShoot.classList.remove("hidden");
    el.autoWrap.classList.remove("hidden");
  }

  function stopCamera() {
    if (state.cameraStream) {
      state.cameraStream.getTracks().forEach((t) => t.stop());
      state.cameraStream = null;
    }
    el.video.srcObject = null;
    el.video.hidden = true;
    el.origImg.classList.remove("hidden");
    el.btnShoot.classList.add("hidden");
    el.autoWrap.classList.add("hidden");
    el.autoCapture.checked = false;
  }

  async function setMode(mode) {
    if (state.mode === mode) return;
    pause();
    state.mode = mode;
    el.modeLine.classList.toggle("active", mode === "line");
    el.modeCamera.classList.toggle("active", mode === "camera");
    if (mode === "camera") {
      try {
        await initCamera();
        el.hint.textContent = "Camera ready: click \"Capture & Inspect\", or enable auto capture";
      } catch (err) {
        toast(`Cannot access camera: ${err.message}. Switched back to Line Sim mode.`, true);
        state.mode = "line";
        el.modeLine.classList.add("active");
        el.modeCamera.classList.remove("active");
        stopCamera();
        el.hint.textContent = "Camera unavailable — back to Line Sim mode";
      }
    } else {
      stopCamera();
      el.hint.textContent = "Switched back to Line Sim mode";
    }
    setRunningUI();
  }

  // ---------------------------------------------------------------- Events
  function bindEvents() {
    el.btnStart.addEventListener("click", start);
    el.btnPause.addEventListener("click", pause);
    el.btnStep.addEventListener("click", step);
    el.btnReset.addEventListener("click", reset);
    el.btnShoot.addEventListener("click", async () => {
      if (state.busy) return;
      if (state.running) pause();
      await tick();
    });

    el.modeLine.addEventListener("click", () => setMode("line"));
    el.modeCamera.addEventListener("click", () => setMode("camera"));

    el.interval.addEventListener("input", () => {
      state.intervalS = Number(el.interval.value);
      el.intervalLabel.textContent = state.intervalS.toFixed(1) + " s";
      if (state.running) schedule();
    });

    el.category.addEventListener("change", async () => {
      try {
        const st = await api(`/api/state?category=${encodeURIComponent(el.category.value)}`);
        state.category = st.default_category;
        state.threshold = Number(st.threshold);
        state.demoTotal = st.demo_total;
        el.thresholdVal.textContent = state.threshold.toFixed(4);
        el.deviceVal.textContent = st.device;
        el.trendMeta.textContent = `Threshold ${state.threshold.toFixed(3)} · ${state.demoTotal} demo images`;
        state.points = state.points.map((p) => ({ ...p, ng: p.score >= state.threshold }));
        drawTrend();
        toast(`Category switched to ${state.category} (threshold ${state.threshold.toFixed(3)})`);
      } catch (err) {
        toast("Category switch failed: " + err.message, true);
      }
    });

    el.autoCapture.addEventListener("change", () => {
      if (el.autoCapture.checked && !state.running) {
        toast("Auto capture enabled — press \"Start\" to capture at takt time");
      }
    });

    document.addEventListener("keydown", (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "select" || tag === "input" || tag === "textarea") return;
      if (e.code === "Space") { e.preventDefault(); state.running ? pause() : start(); }
      if (e.code === "ArrowRight") { e.preventDefault(); step(); }
    });

    window.addEventListener("resize", drawTrend);
  }

  boot();
})();
