# MeroShare IPO Bulk Checker (Chrome extension)

Self-contained Chrome/Edge extension: solves the CDSC IPO-result captcha with a
**local ONNX model running in the browser** (no Python, no server, no Tampermonkey),
then bulk-checks allotment for a list of BOIDs and exports a report.

## Install (unpacked)
1. Open `chrome://extensions` (or `edge://extensions`).
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select this `meroshare-extension` folder.
4. Open `https://iporesult.cdsc.com.np/` — a panel appears bottom-right.

## Use
1. **Select the IPO company** in the page's dropdown.
2. In the panel, **upload your BOID file** (`.xlsx`/`.csv` — first sheet, any column holding 16-digit BOIDs; the first text cell in a row is used as the name).
3. Click **Run batch**. It fills each BOID, solves the captcha locally, submits, reads the result, and paces itself between people.
4. Click **Download report** → `ipo_report.csv` (`name, boid, company, status, detail`; `detail` keeps the alloted quantity).

Leave **auto-fill captcha** on to have single captchas filled as you refresh them.

## How it works
- `inject.js` (page main world) captures the `POST /result/check` JSON response.
- `content.js` (isolated world) drives the form + panel and compiles the report.
- The captcha is sent to an **offscreen document** (`offscreen.js`) that runs the
  segmentation (`src/segment.js`) + the ONNX model (`src/digit_cnn.onnx`) via
  `onnxruntime-web` — WASM runs in the extension context, so the page's CSP can't block it.

## Accuracy
The model is ~50% per captcha on unseen images, so the batch **retries** (reloads +
re-solves, up to 8×) until it gets a valid result. Expect a few seconds per BOID plus
the human-pace pauses. Tune `MAX_TRIES`, `PACE_MIN/MAX`, `THINK_MIN/MAX` at the top of `content.js`.

## Troubleshooting
- **Nothing happens / "model unreachable":** open `chrome://extensions` → this extension →
  **service worker** (and **Inspect** the offscreen document if listed) and check the console
  for WASM load errors. The offscreen doc must load `src/ort/*` — those files must be present.
- **WASM fails to load:** confirm `src/ort/ort.wasm.min.js`, `ort-wasm-simd-threaded.wasm`,
  and `ort-wasm-simd-threaded.mjs` exist. `manifest.json` sets `wasm-unsafe-eval` in the CSP.
- **Result never read:** the site may have changed the API path; `inject.js` matches `/result/check`.

## Rebuilding the model (dev)
Everything under `verify/` is dev-only (do **not** ship it — delete before packing):
```
python verify/export_onnx.py     # digit_cnn.pth -> src/digit_cnn.onnx
python verify/py_baseline.py     # regenerate src/border-mask.js + baseline
cd _verify && npm install && node verify.js   # JS vs Python parity (should match accuracy)
```

## Note
This automates a captcha the site uses to prevent automation. It's fine for checking
your own/family results; **publishing or selling it is another matter** — Chrome Web Store
prohibits captcha-circumvention extensions and CDSC's terms likely forbid automated access.
