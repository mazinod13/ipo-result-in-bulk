// Runs in the PAGE's main world (document_start) to intercept the allotment
// API response and forward it to the content script via window.postMessage.
(function () {
  "use strict";
  const isCheck = (u) => u && /result\/check/i.test(u);
  const post = (j) => {
    try { window.postMessage({ __meroResult: true, success: !!j.success, message: (j.message || "").trim() }, "*"); }
    catch (e) {}
  };

  try {
    const P = XMLHttpRequest.prototype, open = P.open, send = P.send;
    P.open = function (m, u) { this.__u = u; return open.apply(this, arguments); };
    P.send = function () {
      this.addEventListener("load", function () {
        if (isCheck(this.__u)) { try { post(JSON.parse(this.responseText)); } catch (e) {} }
      });
      return send.apply(this, arguments);
    };
  } catch (e) {}

  try {
    const of = window.fetch;
    window.fetch = function (...a) {
      return of.apply(this, a).then((resp) => {
        const u = typeof a[0] === "string" ? a[0] : (a[0] && a[0].url);
        if (isCheck(u)) resp.clone().json().then(post).catch(() => {});
        return resp;
      });
    };
  } catch (e) {}
})();
