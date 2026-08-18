# MeroShare IPO Bulk Checker

A Chrome/Edge extension that checks **CDSC IPO allotment results** for a whole list
of BOIDs at once. It solves the site's anti-bot captcha **locally, in the browser**
(a tiny neural network compiled to WebAssembly — no server, no external API), fills the
form for each BOID, reads the result, and exports a report.

- **Extension:** [`meroshare-extension/`](meroshare-extension/)
- **BOID list format:** [`BOID.template.xlsx`](BOID.template.xlsx)

---

## 1. How the captcha solver was built

The CDSC captcha is deliberately hostile to OCR: five serif digits over a **matrix
grid**, crossed by a thick **wavy strike-through**, wrapped in a thin **ellipse**, with
the digits **overlapping** and sitting on a **staggered baseline**. Off-the-shelf OCR
fails because the grid destroys white-space segmentation and the strike-through fuses
all five digits into one blob.

Rather than read the whole thing end-to-end (which needs a large dataset), the problem
is split into **five single-digit classifications** — far more data-efficient.

### Pipeline (per captcha)
```
150×40 grayscale
      │  threshold (digits → white)
      │  remove the static border  (from the blank-captcha frame)
      │  remove long horizontal runs  → strips the grid + strike-through,
      │                                  leaving mostly vertical digit strokes
      ▼
 column ink profile
      │  slide a fixed-width window to the densest region  → the 5-digit band
      │  refine the band edges to the real ink extent
      │  place 4 cuts at the low-ink valleys between digits
      ▼
 5 cells → each resized to 28×28, as 2 channels:
      channel 0 = raw pixels     (keeps serif/þickness cues)
      channel 1 = denoised strokes (grid/strike removed)
      ▼
 small CNN  (2×28×28 → 10 classes, ~433K params)  → digit 0–9
```

### Data & training
- **Real captchas were collected from the site and labelled by their answer** (filename
  = the digits). The training tooling lives outside this repo; a companion collector
  captured each refreshed captcha so the set could grow to ~1,000+ images.
- Trained with a **captcha-level train/val split**, light **augmentation** (small
  shift/rotate/scale on each digit crop), dropout + weight decay, keeping the
  best model by validation score.
- Honest accuracy: **~80% per digit**. Because all five must be right for one captcha,
  whole-captcha success on unseen images is lower — so the extension simply **retries
  with a fresh captcha until it gets a valid result** (a few attempts per BOID).

### From PyTorch to the browser
- The trained model is exported to **ONNX** and runs via **onnxruntime-web** (WASM).
- The OpenCV preprocessing was re-implemented in plain JavaScript (`src/segment.js`) and
  **verified against the original Python pipeline**: the JS port reproduces the Python
  model's accuracy exactly (identical whole-captcha accuracy on the same sample), so
  the in-browser solver is equivalent to the trained model.

The verification harness (`meroshare-extension/verify/`) exports the ONNX model,
produces a Python baseline, and diffs the JS+ONNX predictions against it.

---

## 2. How the extension works

Manifest V3, fully self-contained. Four moving parts:

| File | Runs in | Job |
|------|---------|-----|
| `inject.js` | page (main world) | hooks `POST /result/check` and forwards the JSON result |
| `content.js` | page (isolated world) | drives the form, the on-page panel, the batch loop, the report |
| `background.js` | service worker | owns the offscreen document |
| `offscreen.js` | offscreen document | decodes the captcha, runs segmentation + the ONNX model |

```
 captcha <img> (data URI)
      │  content.js → background.js → offscreen.js
      ▼
 segment.js  →  digit_cnn.onnx (onnxruntime-web / WASM)  →  "48213"
      │  content.js types it, submits the form
      ▼
 inject.js captures /result/check JSON  →  content.js records status
      ▼
 report → ipo_report.csv
```

Running the model in the **offscreen document** (an extension-owned page) means the WASM
runtime isn't subject to the target site's Content-Security-Policy — the usual reason
in-page WASM fails.

The captcha result is read straight from the site's **API response** (`{ success, message }`)
rather than scraped from the DOM, so classification is exact:
`"Congratulation Alloted !!! …"` → **Alloted**, `"Sorry, not alloted…"` → **Not Alloted**,
`"Invalid Captcha…"` → **retry**.

---

## 3. Install it on your browser

No build step — it loads as-is.

1. Open `chrome://extensions` (or `edge://extensions`).
2. Turn on **Developer mode** (top-right).
3. Click **Load unpacked** and select the **`meroshare-extension/`** folder.
4. Open `https://iporesult.cdsc.com.np/` — a panel appears in the bottom-right corner.

> The `meroshare-extension/verify/` folder is dev-only (model export + parity tests).
> It is not loaded by the extension; you can delete it before zipping/sharing.

### Using it
1. **Select the IPO company** in the page's dropdown.
2. In the panel, **upload your BOID file** — same shape as
   [`BOID.template.xlsx`](BOID.template.xlsx) (first sheet; any column holding a
   16-digit BOID; the first text cell in a row is used as the name).
3. Click **Run batch** — it works through every BOID at a human-like pace, solving the
   captcha locally and retrying as needed.
4. Click **Download report** → `ipo_report.csv` (`name, boid, company, status, detail`;
   `detail` keeps the alloted quantity).

Leave **auto-fill captcha** on to have single captchas filled as you refresh them.

Tunables at the top of `content.js`: `MAX_TRIES`, `PACE_MIN/MAX`, `THINK_MIN/MAX`.

---

## 4. BOID file format

`BOID.template.xlsx` — first sheet, one row per person:

| SN | Name | BOID |
|----|------|------|
| 1 | John Doe | 1301230000000001 |
| 2 | Jane Smith | 1301230000000002 |

Only the 16-digit BOID and a name are needed; extra columns are ignored. Keep BOIDs as
text so leading digits are preserved. **Never commit a real BOID list** — `.gitignore`
keeps `BOID.xlsx` and generated reports out of the repo.

---

## Note on responsible use

This automates a captcha the site uses to deter automation. It's intended for checking
your own / your family's results at a reasonable pace. **Publishing or selling it is a
different matter** — the Chrome Web Store prohibits captcha-circumvention extensions, and
CDSC's terms of use likely forbid automated access. Use accordingly.
