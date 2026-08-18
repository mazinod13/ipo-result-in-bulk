// Verify the JS segmentation + ONNX model reproduce the Python pipeline.
const fs = require("fs");
const path = require("path");
const { PNG } = require("pngjs");
const ort = require("onnxruntime-node");
const SEG = require("../src/segment.js");

const DATA = "D:/meroshare/assets/captcha_images";
const ONNX = path.join(__dirname, "../src/digit_cnn.onnx");
const baseline = JSON.parse(fs.readFileSync(path.join(__dirname, "baseline.json"), "utf8"));

function loadGray(file) {
  const png = PNG.sync.read(fs.readFileSync(path.join(DATA, file)));
  if (png.width !== SEG.IMG_W || png.height !== SEG.IMG_H) return null;
  const gray = new Uint8Array(SEG.IMG_W * SEG.IMG_H);
  for (let i = 0; i < gray.length; i++) gray[i] = png.data[i * 4]; // R (image is grayscale)
  return gray;
}

function argmaxRows(logits, n, cls) {
  let out = "";
  for (let i = 0; i < n; i++) {
    let bi = 0, bv = -Infinity;
    for (let c = 0; c < cls; c++) { const v = logits[i * cls + c]; if (v > bv) { bv = v; bi = c; } }
    out += bi;
  }
  return out;
}

(async () => {
  const session = await ort.InferenceSession.create(ONNX);
  let matchPy = 0, jsCorrect = 0, pyCorrect = 0, tested = 0, skipped = 0;
  const mismatches = [];

  for (const row of baseline) {
    const gray = loadGray(row.file);
    if (!gray) { skipped++; continue; }
    const { data, dims } = SEG.buildInputs(gray);
    const t = new ort.Tensor("float32", data, dims);
    const res = await session.run({ input: t });
    const js = argmaxRows(res.logits.data, SEG.NUM, 10);

    tested++;
    if (js === row.py) matchPy++;
    if (js === row.label) jsCorrect++;
    if (row.py === row.label) pyCorrect++;
    if (js !== row.py && mismatches.length < 15) mismatches.push({ f: row.file, py: row.py, js, label: row.label });
  }

  console.log(`tested ${tested}, skipped ${skipped}`);
  console.log(`JS == Python  : ${(matchPy / tested * 100).toFixed(1)}%  (fidelity of the port)`);
  console.log(`JS  accuracy  : ${(jsCorrect / tested * 100).toFixed(1)}%`);
  console.log(`Py  accuracy  : ${(pyCorrect / tested * 100).toFixed(1)}%`);
  if (mismatches.length) { console.log("sample mismatches (py vs js vs label):"); mismatches.forEach((m) => console.log(" ", m.f, m.py, m.js, m.label)); }
})();
