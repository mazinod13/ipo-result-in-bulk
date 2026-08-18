// MeroShare IPO Bulk Checker — content script (isolated world).
// Drives the result form: fills BOID + captcha (captcha solved locally by the
// offscreen ONNX model), submits, reads the /result/check API response
// (captured by inject.js), and compiles a report over an uploaded BOID list.
(function () {
  "use strict";

  const MAX_TRIES = 8;
  const PACE_MIN = 4000, PACE_MAX = 9000;      // human pace between BOIDs
  const THINK_MIN = 700, THINK_MAX = 1800;     // think time before submit

  let autoFill = true, autoRunning = false, lastFilledSrc = null;
  let boids = [];                               // [{name, boid}]
  let results = [];                             // [{name, boid, company, status, detail}]
  let lastApi = null;                           // {success, message, at}

  const $ = (s) => document.querySelector(s);
  const captchaImg = () => $('img[alt="captcha"]');
  const captchaInput = () => $("#userCaptcha");
  const boidInput = () => $("#boid");
  const reloadBtn = () => $('button[tooltip="Reload Captcha"]');
  const submitBtn = () => $('button[type="submit"]');
  const companyName = () => { const e = $(".ng-value-label"); return e ? e.textContent.trim() : ""; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const rand = (lo, hi) => lo + Math.random() * (hi - lo);

  // API result comes from the page's main world via inject.js
  window.addEventListener("message", (e) => {
    if (e.source === window && e.data && e.data.__meroResult) {
      lastApi = { success: e.data.success, message: e.data.message, at: Date.now() };
    }
  });

  function setValue(el, val) {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(el, val);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    el.dispatchEvent(new Event("blur", { bubbles: true }));
  }

  async function solve(dataURL) {
    try {
      const res = await chrome.runtime.sendMessage({ type: "solve", dataURL });
      if (!res || res.error) { log("model error: " + (res && res.error)); return null; }
      return res.pred;
    } catch (e) {
      log("model unreachable: " + e.message);
      return null;
    }
  }

  async function fillOnce() {
    const im = captchaImg();
    if (!im || !im.src.startsWith("data:")) return null;
    lastFilledSrc = im.src;
    const pred = await solve(im.src);
    if (pred && captchaInput()) setValue(captchaInput(), pred);
    return pred;
  }

  function classify(text) {
    if (/captcha/i.test(text)) return "captcha";
    if (/not\s*allot/i.test(text)) return "Not Alloted";
    if (/allot/i.test(text)) return "Alloted";
    if (/invalid\s*boid|boid.*invalid/i.test(text)) return "Invalid BOID";
    return "?";
  }

  async function loadNewCaptcha() {
    const prev = captchaImg() ? captchaImg().src : null;
    if (reloadBtn()) reloadBtn().click();
    for (let t = 0; t < 25; t++) {
      await sleep(150);
      const im = captchaImg();
      if (im && im.src.startsWith("data:") && im.src !== prev) return im.src;
    }
    return captchaImg() ? captchaImg().src : null;
  }

  async function waitFor(fn, ms) {
    for (let t = 0; t < ms; t += 120) { if (fn()) return true; await sleep(120); }
    return fn();
  }

  async function checkOne(boid) {
    if (!boidInput()) return { status: "Error", detail: "no BOID field" };
    setValue(boidInput(), boid);
    await sleep(300);

    for (let i = 1; i <= MAX_TRIES; i++) {
      const pred = await fillOnce();
      await sleep(300);
      const cv = captchaInput() ? captchaInput().value : "";
      if (cv !== pred && captchaInput()) { setValue(captchaInput(), pred); await sleep(250); }

      const enabled = await waitFor(() => submitBtn() && !submitBtn().disabled, 1500);
      if (!enabled) { await loadNewCaptcha(); continue; }

      await sleep(rand(THINK_MIN, THINK_MAX));         // human think time
      const t0 = Date.now();
      const form = document.querySelector("form");
      if (form && form.requestSubmit) form.requestSubmit(submitBtn());
      else submitBtn().click();

      let msg = "";
      for (let t = 0; t < 6000; t += 200) {
        await sleep(200);
        if (lastApi && lastApi.at >= t0) { msg = lastApi.message; break; }
      }
      if (!msg) { await loadNewCaptcha(); continue; }

      const status = classify(msg);
      if (status === "captcha") { await loadNewCaptcha(); continue; }
      return { status, detail: msg };
    }
    return { status: "Error", detail: "captcha failed after " + MAX_TRIES + " tries" };
  }

  async function runBatch() {
    if (autoRunning) return;
    if (!boids.length) { log("upload a BOID file first"); return; }
    const company = companyName();
    if (!company) { log("select a company first"); return; }
    autoRunning = true;
    results = [];
    try {
      log(`batch: ${boids.length} BOIDs for "${company}"`);
      for (let idx = 0; idx < boids.length; idx++) {
        const { name, boid } = boids[idx];
        const r = await checkOne(boid);
        results.push({ name, boid, company, status: r.status, detail: r.detail });
        log(`(${idx + 1}/${boids.length}) ${boid} ${name} -> ${r.status}`);
        if (idx < boids.length - 1) {
          const w = Math.round(rand(PACE_MIN, PACE_MAX));
          log(`  …pausing ${(w / 1000).toFixed(1)}s`);
          await sleep(w);
        }
      }
      log("batch done. Click 'Download report'.");
    } catch (e) {
      log("batch error: " + e.message);
    } finally {
      autoRunning = false;
    }
  }

  // ---- BOID file (xlsx/csv) parsing ----
  function parseWorkbook(arrayBuf) {
    const wb = XLSX.read(arrayBuf, { type: "array" });
    const ws = wb.Sheets[wb.SheetNames[0]];
    const rows = XLSX.utils.sheet_to_json(ws, { header: 1 });
    const out = [];
    for (const row of rows) {
      let name = null, boid = null;
      for (const cell of row) {
        const s = String(cell == null ? "" : cell).trim();
        if (/^\d{16}$/.test(s)) boid = s;
        else if (s && !name && s.length > 2 && !/^\d+$/.test(s.replace(/\s/g, ""))) name = s;
      }
      if (boid) out.push({ name: name || "", boid });
    }
    return out;
  }

  function onFile(file) {
    const r = new FileReader();
    r.onload = (e) => {
      try {
        boids = parseWorkbook(e.target.result);
        log(`loaded ${boids.length} BOIDs from ${file.name}`);
      } catch (err) { log("could not read file: " + err.message); }
    };
    r.readAsArrayBuffer(file);
  }

  function downloadReport() {
    if (!results.length) { log("no results yet"); return; }
    const rows = [["name", "boid", "company", "status", "detail"],
      ...results.map((r) => [r.name, r.boid, r.company, r.status, r.detail])];
    const csv = rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = "ipo_report.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 3000);
  }

  // Auto-fill whenever the captcha image changes.
  setInterval(() => {
    if (!autoFill || autoRunning) return;
    const im = captchaImg();
    if (im && im.src.startsWith("data:") && im.src !== lastFilledSrc) {
      fillOnce().then((p) => p && log("auto-filled: " + p));
    }
  }, 600);

  // ---- panel ----
  const panel = document.createElement("div");
  panel.style.cssText =
    "position:fixed;bottom:12px;right:12px;z-index:99999;background:#222;color:#fff;" +
    "padding:10px;border-radius:8px;font:12px sans-serif;width:250px;box-shadow:0 2px 8px rgba(0,0,0,.4)";
  panel.innerHTML =
    "<b>IPO Bulk Checker</b>" +
    '<label style="display:block;margin:4px 0"><input type="checkbox" id="mAuto" checked> auto-fill captcha</label>' +
    '<div style="margin:6px 0"><input type="file" id="mFile" accept=".xlsx,.xls,.csv" style="width:100%"></div>' +
    '<div style="margin:6px 0"><button id="mRun">Run batch</button> <button id="mDl">Download report</button></div>' +
    '<div id="mLog" style="max-height:120px;overflow:auto;font:11px monospace;opacity:.9"></div>';
  document.body.appendChild(panel);

  function log(m) {
    const d = document.getElementById("mLog");
    d.innerHTML += m + "<br>"; d.scrollTop = d.scrollHeight;
    console.log("[ipo]", m);
  }
  document.getElementById("mFile").onchange = (e) => e.target.files[0] && onFile(e.target.files[0]);
  document.getElementById("mRun").onclick = runBatch;
  document.getElementById("mDl").onclick = downloadReport;
  document.getElementById("mAuto").onchange = (e) => { autoFill = e.target.checked; };
  log("ready. Select company, upload BOID file, Run batch.");
})();
