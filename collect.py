"""Collect + auto-guess captcha images from the CDSC IPO-result site.

The site's WAF blocks scripted downloads, so we capture from the browser. Two
capture modes:

  watch  (fastest) — right-click the captcha in the browser -> "Copy image".
                     This script watches the clipboard and auto-saves each new
                     image. No typing at all.
  paste            — paste the image's ``data:image/png;base64,...`` value.

Every captured image is run through the trained digit model and saved named by
its GUESS (e.g. ``87935.png``). You only have to rename the ones it got wrong —
no labelling during collection. Then continue training on the corrected set.

Usage
-----
    python collect.py watch          # clipboard auto-capture (default)
    python collect.py paste          # paste base64 data URIs
    python collect.py once "<uri>"   # save a single data URI
    python collect.py train          # CONTINUE training on assets/captcha_images
    python collect.py train --fresh --epochs 80
"""

import os
import io
import time
import base64
import hashlib
import binascii
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

import train  # reuse DigitCNN, segmentation, predict_img, paths

OUT_DIR = train.DATA_DIR                 # ./assets/captcha_images
CANVAS = (150, 40)                       # (width, height) of a real captcha


# ----------------------------------------------------------------
# Model helpers
# ----------------------------------------------------------------
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train.DigitCNN().to(device)
    if os.path.exists(train.MODEL_PATH):
        model.load_state_dict(torch.load(train.MODEL_PATH, map_location=device))
        model.eval()
        print(f"Loaded model {train.MODEL_PATH} - captures will be auto-guessed.")
        return model, device
    print(f"No model at {train.MODEL_PATH} yet - captures saved as 'captcha'. "
          f"Train once, then guesses will name the files.")
    return None, device


def guess_name(model, device, gray):
    if model is None:
        return "captcha"
    try:
        return train.predict_img(model, gray, device)
    except Exception as e:
        print(f"  ! guess failed: {e}")
        return "captcha"


# ----------------------------------------------------------------
# Saving
# ----------------------------------------------------------------
def _to_canvas_gray(pil_img):
    """Convert any captured image to a 150x40 grayscale array (real-captcha shape)."""
    gray = np.array(pil_img.convert("L"))
    if (gray.shape[1], gray.shape[0]) != CANVAS:
        gray = cv2.resize(gray, CANVAS, interpolation=cv2.INTER_AREA)
    return gray


def save_gray(gray, model, device, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    name = guess_name(model, device, gray)
    path = os.path.join(out_dir, f"{name}.png")
    i = 1
    while os.path.exists(path):                 # never overwrite a corrected file
        path = os.path.join(out_dir, f"{name}_{i}.png")
        i += 1
    cv2.imwrite(path, gray)
    print(f"  saved -> {os.path.basename(path)}   (guess: {name})")
    return path


def _decode_data_uri(data):
    data = data.strip().strip('"').strip("'")
    if data.startswith("data:"):
        comma = data.find(",")
        if comma != -1:
            data = data[comma + 1:]
    b64 = "".join(data.split())
    if not b64:
        return None
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        print(f"  ! invalid base64: {e}")
        return None


# ----------------------------------------------------------------
# Modes
# ----------------------------------------------------------------
def mode_once(data_uri):
    model, device = load_model()
    raw = _decode_data_uri(data_uri)
    if raw:
        save_gray(_to_canvas_gray(Image.open(io.BytesIO(raw))), model, device)


def mode_paste():
    model, device = load_model()
    print("\nPaste a captcha data URI and press Enter. 'q' or Ctrl+C to quit.\n")
    n = 0
    try:
        while True:
            data = input("data URI > ").strip()
            if not data:
                continue
            if data.lower() in ("q", "quit", "exit"):
                break
            raw = _decode_data_uri(data)
            if raw:
                save_gray(_to_canvas_gray(Image.open(io.BytesIO(raw))), model, device)
                n += 1
    except (KeyboardInterrupt, EOFError):
        pass
    print(f"\nCollected {n} captcha(s) into {OUT_DIR}")


def mode_watch(poll=0.7):
    """Watch the Windows clipboard; auto-save each NEW image copied from the browser."""
    from PIL import ImageGrab
    model, device = load_model()
    print("\nWatching clipboard. In the browser: right-click the captcha -> 'Copy image'.")
    print("Each new image is auto-saved & guessed. Ctrl+C to stop.\n")
    last_hash = None
    n = 0
    try:
        while True:
            obj = ImageGrab.grabclipboard()
            if isinstance(obj, Image.Image):
                h = hashlib.md5(obj.tobytes()).hexdigest()
                if h != last_hash:                 # dedupe: same clipboard = skip
                    last_hash = h
                    save_gray(_to_canvas_gray(obj), model, device)
                    n += 1
            time.sleep(poll)
    except (KeyboardInterrupt, EOFError):
        pass
    print(f"\nCollected {n} captcha(s) into {OUT_DIR}")


def mode_serve(port=8756):
    """Run a tiny local server. The Tampermonkey userscript (capture.user.js)
    POSTs each refreshed captcha here and it's saved silently. You just click
    'refresh' on the captcha in the browser."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    model, device = load_model()
    seen = set()

    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.end_headers()

        def do_GET(self):
            msg = f"collect.py server running. Saved {len(seen)} captcha(s) so far.".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self._cors(); self.end_headers()
            self.wfile.write(msg)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode("utf-8", "ignore")
            raw = _decode_data_uri(body)
            status = 400
            if raw:
                digest = hashlib.md5(raw).hexdigest()
                if digest in seen:                 # same captcha posted twice -> skip
                    status = 208
                else:
                    seen.add(digest)
                    try:
                        save_gray(_to_canvas_gray(Image.open(io.BytesIO(raw))), model, device)
                        status = 200
                    except Exception as e:
                        print(f"  ! save failed: {e}"); status = 500
            self.send_response(status); self._cors(); self.end_headers()

        def log_message(self, *a):
            pass                                   # silence default request logging

    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(f"\nListening on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print("Install capture.user.js in Tampermonkey, open the IPO-result page,")
    print("and just click the captcha refresh button — each one saves here.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\nStopped. Images in {OUT_DIR}")


def mode_train(args):
    train.train(resume=not args.fresh, epochs=args.epochs)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect + guess captchas, and continue training.")
    sub = parser.add_subparsers(dest="cmd")

    p_serve = sub.add_parser("serve", help="local server; userscript posts each refreshed captcha (default)")
    p_serve.add_argument("--port", type=int, default=8756)
    sub.add_parser("watch", help="auto-capture images from the clipboard")
    sub.add_parser("paste", help="paste base64 data URIs")
    p_once = sub.add_parser("once", help="save a single data URI")
    p_once.add_argument("data_uri")
    p_train = sub.add_parser("train", help="continue training on collected images")
    p_train.add_argument("--epochs", type=int, default=train.EPOCHS)
    p_train.add_argument("--fresh", action="store_true", help="train from scratch, ignore saved weights")

    args = parser.parse_args()
    cmd = args.cmd or "serve"          # default to the silent browser-capture server

    if cmd == "serve":
        mode_serve(args.port)
    elif cmd == "watch":
        mode_watch()
    elif cmd == "paste":
        mode_paste()
    elif cmd == "once":
        mode_once(args.data_uri)
    elif cmd == "train":
        mode_train(args)


if __name__ == "__main__":
    main()
