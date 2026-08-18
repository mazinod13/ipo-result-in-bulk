// Runs in the offscreen document (extension context -> WASM allowed).
// Decodes a captcha data URI, segments it, runs the ONNX digit model, and
// returns the 5-digit prediction.

ort.env.wasm.wasmPaths = chrome.runtime.getURL("src/ort/");
ort.env.wasm.numThreads = 1;                       // no SharedArrayBuffer needed
ort.env.wasm.simd = true;

let sessionPromise = null;
function getSession() {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(chrome.runtime.getURL("src/digit_cnn.onnx"));
  }
  return sessionPromise;
}

async function toGray(dataURL) {
  const blob = await (await fetch(dataURL)).blob();
  const bmp = await createImageBitmap(blob);
  const c = new OffscreenCanvas(MERO_SEG.IMG_W, MERO_SEG.IMG_H);
  const ctx = c.getContext("2d");
  ctx.drawImage(bmp, 0, 0, MERO_SEG.IMG_W, MERO_SEG.IMG_H);
  const d = ctx.getImageData(0, 0, MERO_SEG.IMG_W, MERO_SEG.IMG_H).data;
  const gray = new Uint8Array(MERO_SEG.IMG_W * MERO_SEG.IMG_H);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = Math.round(0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]);
  }
  return gray;
}

async function solve(dataURL) {
  const gray = await toGray(dataURL);
  const { data, dims } = MERO_SEG.buildInputs(gray);
  const session = await getSession();
  const out = await session.run({ input: new ort.Tensor("float32", data, dims) });
  const logits = out.logits.data;                 // [5*10]
  let pred = "";
  for (let i = 0; i < MERO_SEG.NUM; i++) {
    let bi = 0, bv = -Infinity;
    for (let c = 0; c < 10; c++) { const v = logits[i * 10 + c]; if (v > bv) { bv = v; bi = c; } }
    pred += bi;
  }
  return pred;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.target !== "offscreen") return;
  if (msg.type === "solve") {
    solve(msg.dataURL).then((pred) => sendResponse({ pred }))
      .catch((e) => sendResponse({ error: String(e) }));
    return true;   // async
  }
});
