"""Emit (1) the border mask as a JS module and (2) a baseline of Python
predictions so the JS/ONNX port can be verified against it."""
import sys, os, json, base64, random
sys.path.insert(0, "D:/meroshare")
import cv2, numpy as np, torch
import train

EXT = "D:/meroshare/meroshare-extension"
DATA = "D:/meroshare/assets/captcha_images"

# --- 1. border mask -> JS ---
shape = (40, 150)
bmask = train._border_mask(shape)                     # uint8 (40,150), 255 at border
raw = (bmask > 0).astype(np.uint8) * 255
b64 = base64.b64encode(raw.tobytes()).decode()
with open(EXT + "/src/border-mask.js", "w") as f:
    f.write("// Auto-generated from blankCaptcha.png (threshold+dilate). 150x40, 255=border.\n")
    f.write('export const BORDER_MASK_B64 = "%s";\n' % b64)
    f.write("export const IMG_W = 150, IMG_H = 40;\n")
print("wrote border-mask.js", len(b64), "b64 chars")

# --- 2. python baseline predictions ---
device = "cpu"
model = train.DigitCNN().to(device)
model.load_state_dict(torch.load("D:/meroshare/digit_cnn.pth", map_location=device))
model.eval()
bm = train._border_mask(shape)

files = [f for f in os.listdir(DATA) if f.endswith(".png")]
files = [f for f in files if len(os.path.splitext(f)[0].split("_")[0]) == 5
         and os.path.splitext(f)[0].split("_")[0].isdigit()]
random.Random(0).shuffle(files)
files = files[:200]

out, correct = [], 0
for fn in files:
    label = os.path.splitext(fn)[0].split("_")[0]
    img = cv2.imread(os.path.join(DATA, fn), cv2.IMREAD_GRAYSCALE)
    pred = train.predict_img(model, img, device, bm)
    out.append({"file": fn, "label": label, "py": pred})
    correct += int(pred == label)

with open(EXT + "/_verify/baseline.json", "w") as f:
    json.dump(out, f)
print(f"baseline: {len(out)} images, python whole-captcha acc = {correct/len(out):.1%}")
