// Service worker: owns the offscreen document (which runs the ONNX/WASM model)
// and relays solve requests from the content script to it.

let creating = null;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  if (!creating) {
    creating = chrome.offscreen.createDocument({
      url: "offscreen.html",
      reasons: ["WORKERS"],
      justification: "Run the captcha OCR model (ONNX Runtime / WASM).",
    });
  }
  await creating;
  creating = null;
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.target === "offscreen") return;   // not for us (avoid relay loop)
  if (msg && msg.type === "solve") {
    (async () => {
      try {
        await ensureOffscreen();
        const res = await chrome.runtime.sendMessage({ target: "offscreen", type: "solve", dataURL: msg.dataURL });
        sendResponse(res);
      } catch (e) {
        sendResponse({ error: String(e) });
      }
    })();
    return true;   // async response
  }
});
