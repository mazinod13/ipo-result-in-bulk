// ==UserScript==
// @name         CDSC Captcha Collector
// @namespace    meroshare.captcha
// @version      1.1
// @description  Silently sends each refreshed captcha to the local collect.py server (python collect.py serve).
// @match        *://*.cdsc.com.np/*
// @match        *://iporesult.cdsc.com.np/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// @connect      localhost
// @run-at       document-idle
// @all-frames   true
// ==/UserScript==

(function () {
    "use strict";

    const ENDPOINT = "http://127.0.0.1:8756/save";   // must match `python collect.py serve --port`
    const POLL_MS = 600;                              // how often to check for a new captcha
    const DEBUG = true;                              // set false once it's working
    let lastSent = null;
    let heartbeat = 0;

    // The captcha is ~150x40. Match images/canvases in that size range.
    function sizeOk(w, h) {
        return w >= 90 && w <= 300 && h >= 20 && h <= 70;
    }

    function fromImg(img) {
        if (img.src && img.src.startsWith("data:")) return img.src;   // exact original bytes
        try {
            const c = document.createElement("canvas");
            c.width = img.naturalWidth; c.height = img.naturalHeight;
            c.getContext("2d").drawImage(img, 0, 0);
            return c.toDataURL("image/png");
        } catch (e) {
            console.warn("[captcha] image is cross-origin, can't read pixels:", img.src.slice(0, 60));
            return null;
        }
    }

    function fromCanvas(cv) {
        try { return cv.toDataURL("image/png"); }
        catch (e) { console.warn("[captcha] canvas is tainted, can't read pixels"); return null; }
    }

    function send(dataURL, where) {
        if (!dataURL || dataURL === lastSent) return;   // dedupe: only new captchas
        lastSent = dataURL;
        if (DEBUG) console.log("[captcha] posting from", where, "len", dataURL.length);
        GM_xmlhttpRequest({
            method: "POST", url: ENDPOINT, data: dataURL,
            headers: { "Content-Type": "text/plain" },
            onload: (r) => console.log("[captcha] sent ->", r.status),
            onerror: () => console.warn("[captcha] server not reachable - is `python collect.py serve` running?"),
        });
    }

    function scan() {
        // 1) <img> candidates
        for (const img of document.images) {
            if (img.complete && img.naturalWidth && sizeOk(img.naturalWidth, img.naturalHeight)) {
                send(fromImg(img), "img");
                return;
            }
        }
        // 2) <canvas> candidates
        for (const cv of document.querySelectorAll("canvas")) {
            if (sizeOk(cv.width, cv.height)) {
                send(fromCanvas(cv), "canvas");
                return;
            }
        }
        // 3) nothing matched: periodically dump what IS on the page so we can tune the filter
        if (DEBUG && (heartbeat++ % 8 === 0)) {
            const imgs = [...document.images].map(i => `${i.naturalWidth}x${i.naturalHeight} ${i.src.slice(0, 24)}`);
            const cvs = [...document.querySelectorAll("canvas")].map(c => `${c.width}x${c.height}`);
            console.log("[captcha] no match. imgs:", imgs, "| canvases:", cvs, "| frame:", location.href);
        }
    }

    setInterval(scan, POLL_MS);
    console.log("[captcha] collector active in frame:", location.href);
})();
