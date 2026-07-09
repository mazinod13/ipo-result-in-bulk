"""Captcha reader — segmentation + per-digit CNN classifier.

Rather than reading the whole 5-digit captcha end-to-end (which needs a lot of
data), we reuse the segmentation logic to split each captcha into 5 single-digit
crops and train a small CNN to classify ONE digit (10 classes). This is far more
data-efficient: 34 captchas -> 170 labeled digit crops.

Pipeline per image:
  1. Binarize (digits white) and drop the static border.
  2. Remove long horizontal runs (grid + strike-through) to expose vertical strokes.
  3. Slide a fixed-width window to the densest ink -> the 5-digit band.
  4. Split the band into 5 equal cells; each cell's label is the digit at that
     position in the filename ("87935" -> 8,7,9,3,5).

Train:      python train.py
Predict:    python train.py <path-to-captcha.png>
"""

import os
import sys
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
DATA_DIR = "./assets/captcha_images_raw"
BLANK_PATH = "./assets/blankCaptcha.png"
MODEL_PATH = "digit_cnn.pth"

NUM_DIGITS = 5
BAND_WIDTH = 96            # width of the 5-digit block (measured from real samples)
CROP_SIZE = 28            # each digit crop is resized to CROP_SIZE x CROP_SIZE
BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15          # fraction of captchas held out for validation
AUGMENT = True            # random shift/rotate/scale on training crops


# ----------------------------------------------------------------
# Segmentation (the "logic": grid/strike removal + band split)
# ----------------------------------------------------------------
def _border_mask(shape):
    blank = cv2.imread(BLANK_PATH, cv2.IMREAD_GRAYSCALE)
    if blank is None or blank.shape != shape:
        return None
    _, m = cv2.threshold(blank, 150, 255, cv2.THRESH_BINARY_INV)
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def _clean_image(img, border_mask):
    """Denoised captcha: border, grid, and strike-through (long horizontal runs)
    removed, leaving mostly vertical digit strokes. uint8, digits white on black."""
    _, th = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    if border_mask is not None:
        th[border_mask > 0] = 0
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 1)))
    return cv2.medianBlur(cv2.subtract(th, horiz), 3)


def _stroke_columns(img, border_mask):
    """Per-column ink profile of the denoised image (used to locate the band)."""
    return np.sum(_clean_image(img, border_mask) > 0, axis=0).astype(np.float64)


def _cut_positions(col):
    """Return NUM_DIGITS+1 x-coordinates splitting the 5-digit band.

    1. Anchor a fixed-width window at the densest ink region.
    2. Refine its left/right edges to the actual digit-ink extent (stops the
       strike's left hook / a narrow window from clipping the first/last digit).
    3. Place the 4 internal cuts at the lowest-ink columns (valleys between
       digits) near each equal-division boundary, since digits aren't equal width.
    """
    W = len(col)
    win = min(BAND_WIDTH, W)
    if col.sum() == 0:
        left = max(0, (W - win) // 2); right = left + win
    else:
        csum = np.concatenate(([0.0], np.cumsum(col)))
        left = int(np.argmax(csum[win:] - csum[:-win]))
        right = left + win
        # Refine edges to the real ink extent within a slightly expanded window.
        lo, hi = max(0, left - 12), min(W, right + 12)
        peak = col[lo:hi].max()
        thr = max(2.0, peak * 0.12)
        on = np.where(col[lo:hi] >= thr)[0]
        if on.size:
            left, right = lo + int(on[0]), lo + int(on[-1]) + 1

    seg = (right - left) / NUM_DIGITS
    cuts = [left]
    for k in range(1, NUM_DIGITS):
        ideal = left + k * seg
        s = max(cuts[-1] + 3, int(ideal - 4))
        e = min(right - 3, int(ideal + 5))
        cut = int(ideal) if s >= e else s + int(np.argmin(col[s:e]))
        cuts.append(cut)
    cuts.append(right)
    return cuts


def segment(img, border_mask=None):
    """Split a grayscale captcha into NUM_DIGITS raw crops (for visualization)."""
    if border_mask is None:
        border_mask = _border_mask(img.shape)
    cuts = _cut_positions(_stroke_columns(img, border_mask))
    return [img[:, cuts[i]:cuts[i + 1]] for i in range(NUM_DIGITS)]


def segment_pairs(img, border_mask=None):
    """Split into NUM_DIGITS (raw_crop, clean_crop) pairs at the same cut lines.
    raw = original pixels (with noise); clean = grid/strike removed."""
    if border_mask is None:
        border_mask = _border_mask(img.shape)
    clean = _clean_image(img, border_mask)
    cuts = _cut_positions(np.sum(clean > 0, axis=0).astype(np.float64))
    return [(img[:, cuts[i]:cuts[i + 1]], clean[:, cuts[i]:cuts[i + 1]])
            for i in range(NUM_DIGITS)]


def _sq(crop):
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def _to_squares(raw_crop, clean_crop):
    """Return the (raw_square, clean_square) uint8 pair for a digit crop."""
    return _sq(raw_crop), _sq(clean_crop)


def _normalize_pair(raw_sq, clean_sq):
    """Two uint8 squares -> normalized 2-channel tensor [2, H, W]."""
    r = torch.from_numpy(raw_sq).float() / 255.0
    c = torch.from_numpy(clean_sq).float() / 255.0
    return torch.stack([(r - 0.5) / 0.5, (c - 0.5) / 0.5], 0)


def _prep_pair(raw_crop, clean_crop):
    return _normalize_pair(*_to_squares(raw_crop, clean_crop))


def _augment_pair(raw_sq, clean_sq):
    """Apply ONE random shift/rotation/scale to both channels (kept in sync), so
    the classifier tolerates segmentation jitter."""
    h, w = raw_sq.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), np.random.uniform(-7, 7),
                                np.random.uniform(0.88, 1.12))
    M[0, 2] += np.random.uniform(-2.5, 2.5)
    M[1, 2] += np.random.uniform(-2.5, 2.5)
    raw = cv2.warpAffine(raw_sq, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
    clean = cv2.warpAffine(clean_sq, M, (w, h), borderValue=0, flags=cv2.INTER_LINEAR)
    return raw, clean


def list_captchas(image_dir):
    """List (path, label) for every validly-named captcha in a directory."""
    items = []
    for fname in os.listdir(image_dir):
        if not fname.endswith((".png", ".jpg")):
            continue
        label = os.path.splitext(fname)[0].split("_")[0]
        if len(label) == NUM_DIGITS and label.isdigit():
            items.append((os.path.join(image_dir, fname), label))
    return items


# ----------------------------------------------------------------
# Dataset: one sample = one digit crop + its digit label
# ----------------------------------------------------------------
class DigitDataset(Dataset):
    """Built from a list of captchas so train/val never share an image."""
    def __init__(self, items, bmask, augment=False):
        self.augment = augment
        self.samples = []            # (raw_sq, clean_sq, label int 0-9)
        for path, label in items:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            for (raw, clean), digit in zip(segment_pairs(img, bmask), label):
                if raw.size == 0:
                    continue
                self.samples.append((*_to_squares(raw, clean), int(digit)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sq, clean_sq, y = self.samples[idx]
        if self.augment:
            raw_sq, clean_sq = _augment_pair(raw_sq, clean_sq)
        return _normalize_pair(raw_sq, clean_sq), y


# ----------------------------------------------------------------
# Model: small CNN, single-digit classifier
# ----------------------------------------------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14 (raw+clean channels)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------
@torch.no_grad()
def captcha_accuracy(model, items, bmask, device):
    """Evaluate on whole captchas. Returns (whole_captcha_acc, per_digit_acc)."""
    model.eval()
    cap_ok = cap_tot = dig_ok = dig_tot = 0
    for path, label in items:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        pred = predict_img(model, img, device, bmask)
        cap_ok += int(pred == label); cap_tot += 1
        for p, t in zip(pred.ljust(NUM_DIGITS), label):
            dig_ok += int(p == t); dig_tot += 1
    return cap_ok / max(cap_tot, 1), dig_ok / max(dig_tot, 1)


def train(resume=True, epochs=EPOCHS):
    """Train the digit classifier with a captcha-level train/val split, crop
    augmentation, and best-by-validation checkpointing. If resume=True and a
    saved model exists, its weights are loaded first to CONTINUE training."""
    if not (os.path.exists(DATA_DIR) and os.listdir(DATA_DIR)):
        print(f"No data in '{DATA_DIR}'."); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split at the CAPTCHA level so crops from one image never leak across sets.
    items = list_captchas(DATA_DIR)
    random.Random(42).shuffle(items)
    n_val = max(1, int(len(items) * VAL_SPLIT))
    val_items, train_items = items[:n_val], items[n_val:]

    bmask = None
    for path, _ in items:                       # derive border mask once
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            bmask = _border_mask(im.shape); break

    train_ds = DigitDataset(train_items, bmask, augment=AUGMENT)
    if len(train_ds) == 0:
        print("No digit crops produced - check the images/labels."); return
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = DigitCNN().to(device)
    if resume and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Resumed from {MODEL_PATH}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"Train: {len(train_items)} captchas ({len(train_ds)} crops) | "
          f"Val: {len(val_items)} captchas | {device}")
    best_val = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            val_cap, val_dig = captcha_accuracy(model, val_items, bmask, device)
            flag = ""
            if val_cap > best_val:              # keep the best model on validation
                best_val = val_cap
                torch.save(model.state_dict(), MODEL_PATH)
                flag = "  <- saved (best)"
            print(f"Epoch {epoch:3d}/{epochs} | loss {loss_sum/total:.4f} | "
                  f"train digit {correct/total:5.1%} | VAL digit {val_dig:5.1%} | "
                  f"VAL captcha {val_cap:5.1%}{flag}")
        else:
            print(f"Epoch {epoch:3d}/{epochs} | loss {loss_sum/total:.4f} | train digit {correct/total:5.1%}")

    print(f"Best validation captcha accuracy: {best_val:.1%}. Model saved to {MODEL_PATH}")


# ----------------------------------------------------------------
# Inference
# ----------------------------------------------------------------
@torch.no_grad()
def predict_img(model, img, device, border_mask=None):
    model.eval()
    pairs = segment_pairs(img, border_mask)
    batch = torch.stack([_prep_pair(r, c) for r, c in pairs if r.size > 0]).to(device)
    preds = model(batch).argmax(1).tolist()
    return "".join(str(p) for p in preds)


def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitCNN().to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"No trained model at {MODEL_PATH}. Run 'python train.py' first."); return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}."); return
    print(predict_img(model, img, device))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        train()
