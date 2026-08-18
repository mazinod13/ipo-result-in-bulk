// JS port of the Python (OpenCV) captcha segmentation pipeline in train.py.
// Pure functions over a grayscale Uint8Array (150x40) -> five 2-channel 28x28
// tensors, ready for the ONNX digit model. Verified against Python in _verify/.
(function (g) {
  "use strict";

  const BORDER = (typeof module !== "undefined" && module.exports)
    ? require("./border-mask.js") : g.MERO_BORDER;

  const IMG_W = 150, IMG_H = 40, NUM = 5, BAND = 96, CROP = 28;

  function b64ToBytes(b64) {
    if (typeof atob === "function") {
      const s = atob(b64), a = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
      return a;
    }
    return new Uint8Array(Buffer.from(b64, "base64"));
  }
  const MASK = b64ToBytes(BORDER.B64);            // 150*40, 255 = border

  // th = 255 where gray <= 100 (cv2 THRESH_BINARY_INV), border zeroed out.
  function thresholdInv(gray) {
    const n = gray.length, th = new Uint8Array(n);
    for (let i = 0; i < n; i++) th[i] = (gray[i] <= 100 && MASK[i] === 0) ? 255 : 0;
    return th;
  }

  // Horizontal erosion/dilation with a 1x13 kernel (out-of-bounds ignored),
  // matching cv2.morphologyEx OPEN used to strip the strike-through + grid.
  function erodeH(src, k) {
    const r = (k - 1) >> 1, out = new Uint8Array(src.length);
    for (let y = 0; y < IMG_H; y++) {
      const row = y * IMG_W;
      for (let x = 0; x < IMG_W; x++) {
        let on = 255;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= IMG_W) continue;      // border const = max -> ignore
          if (src[row + xx] === 0) { on = 0; break; }
        }
        out[row + x] = on;
      }
    }
    return out;
  }
  function dilateH(src, k) {
    const r = (k - 1) >> 1, out = new Uint8Array(src.length);
    for (let y = 0; y < IMG_H; y++) {
      const row = y * IMG_W;
      for (let x = 0; x < IMG_W; x++) {
        let on = 0;
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= IMG_W) continue;      // border const = min -> ignore
          if (src[row + xx] === 255) { on = 255; break; }
        }
        out[row + x] = on;
      }
    }
    return out;
  }

  function subtract(a, b) {                          // th AND NOT horiz
    const out = new Uint8Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = (a[i] === 255 && b[i] === 0) ? 255 : 0;
    return out;
  }

  // 3x3 median (majority for binary) with replicated borders (cv2 medianBlur).
  function median3(src) {
    const out = new Uint8Array(src.length);
    const clamp = (v, hi) => v < 0 ? 0 : (v >= hi ? hi - 1 : v);
    for (let y = 0; y < IMG_H; y++) {
      for (let x = 0; x < IMG_W; x++) {
        let cnt = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            if (src[clamp(y + dy, IMG_H) * IMG_W + clamp(x + dx, IMG_W)] === 255) cnt++;
        out[y * IMG_W + x] = cnt >= 5 ? 255 : 0;
      }
    }
    return out;
  }

  function cleanImage(gray) {
    const th = thresholdInv(gray);
    const horiz = dilateH(erodeH(th, 13), 13);       // MORPH_OPEN
    return median3(subtract(th, horiz));
  }

  function colSums(clean) {
    const col = new Float64Array(IMG_W);
    for (let y = 0; y < IMG_H; y++)
      for (let x = 0; x < IMG_W; x++)
        if (clean[y * IMG_W + x] === 255) col[x]++;
    return col;
  }

  // Port of _cut_positions(): densest fixed-width window, edge-refine, valley cuts.
  function cutPositions(col) {
    const W = col.length, win = Math.min(BAND, W);
    let total = 0; for (let i = 0; i < W; i++) total += col[i];
    let left, right;
    if (total === 0) { left = Math.max(0, (W - win) >> 1); right = left + win; }
    else {
      const cs = new Float64Array(W + 1);
      for (let i = 0; i < W; i++) cs[i + 1] = cs[i] + col[i];
      let best = -1; left = 0;
      for (let s = 0; s + win <= W; s++) { const v = cs[s + win] - cs[s]; if (v > best) { best = v; left = s; } }
      right = left + win;
      const lo = Math.max(0, left - 12), hi = Math.min(W, right + 12);
      let peak = 0; for (let i = lo; i < hi; i++) if (col[i] > peak) peak = col[i];
      const thr = Math.max(2.0, peak * 0.12);
      let first = -1, last = -1;
      for (let i = lo; i < hi; i++) if (col[i] >= thr) { if (first < 0) first = i; last = i; }
      if (first >= 0) { left = first; right = last + 1; }
    }
    const seg = (right - left) / NUM, cuts = [left];
    for (let k = 1; k < NUM; k++) {
      const ideal = left + k * seg;
      const s = Math.max(cuts[cuts.length - 1] + 3, Math.trunc(ideal - 4));
      const e = Math.min(right - 3, Math.trunc(ideal + 5));
      let cut;
      if (s >= e) cut = Math.trunc(ideal);
      else { let m = Infinity, mi = s; for (let i = s; i < e; i++) if (col[i] < m) { m = col[i]; mi = i; } cut = mi; }
      cuts.push(cut);
    }
    cuts.push(right);
    return cuts;
  }

  // Area-average resample (approximates cv2.INTER_AREA) from a sub-crop to CROPxCROP.
  function resizeArea(src, sw, sh, dw, dh) {
    const dst = new Float32Array(dw * dh), sxr = sw / dw, syr = sh / dh;
    for (let dy = 0; dy < dh; dy++) {
      const y0 = dy * syr, y1 = (dy + 1) * syr, iy0 = Math.floor(y0), iy1 = Math.ceil(y1);
      for (let dx = 0; dx < dw; dx++) {
        const x0 = dx * sxr, x1 = (dx + 1) * sxr, ix0 = Math.floor(x0), ix1 = Math.ceil(x1);
        let sum = 0, area = 0;
        for (let yy = iy0; yy < iy1; yy++) {
          const wy = Math.min(y1, yy + 1) - Math.max(y0, yy); if (wy <= 0) continue;
          for (let xx = ix0; xx < ix1; xx++) {
            const wx = Math.min(x1, xx + 1) - Math.max(x0, xx); if (wx <= 0) continue;
            sum += src[yy * sw + xx] * (wy * wx); area += wy * wx;
          }
        }
        dst[dy * dw + dx] = area > 0 ? sum / area : 0;
      }
    }
    return dst;
  }

  function subCrop(full, x0, x1) {                   // full-height column slice
    const cw = x1 - x0, out = new Float32Array(cw * IMG_H);
    for (let y = 0; y < IMG_H; y++)
      for (let x = 0; x < cw; x++) out[y * cw + x] = full[y * IMG_W + (x0 + x)];
    return out;
  }

  // Build the ONNX batch [NUM, 2, CROP, CROP] as a flat Float32Array.
  function buildInputs(gray) {
    const clean = cleanImage(gray);
    const cuts = cutPositions(colSums(clean));
    const per = 2 * CROP * CROP, data = new Float32Array(NUM * per);
    for (let i = 0; i < NUM; i++) {
      const x0 = cuts[i], x1 = cuts[i + 1], cw = Math.max(1, x1 - x0);
      const rawSq = resizeArea(subCrop(gray, x0, x0 + cw), cw, IMG_H, CROP, CROP);
      const clnSq = resizeArea(subCrop(clean, x0, x0 + cw), cw, IMG_H, CROP, CROP);
      const base = i * per;
      for (let j = 0; j < CROP * CROP; j++) {
        data[base + j] = rawSq[j] / 127.5 - 1;                 // channel 0: raw
        data[base + CROP * CROP + j] = clnSq[j] / 127.5 - 1;   // channel 1: clean
      }
    }
    return { data: data, n: NUM, dims: [NUM, 2, CROP, CROP], cuts: cuts };
  }

  const API = { buildInputs, cleanImage, cutPositions, colSums, IMG_W, IMG_H, NUM, CROP };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else g.MERO_SEG = API;
})(typeof self !== "undefined" ? self : this);
