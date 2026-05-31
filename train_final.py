"""
train_final.py — PlantSage AI  (Final, Stable Version)
======================================================
Fixes the BatchNormalization catastrophic forgetting bug.

ROOT CAUSE of previous low accuracy:
  ❌ Phase 2 set base_model.trainable = True 
  ❌ This switches BN from inference mode → training mode
  ❌ BN uses batch stats instead of running stats → features destroyed
  ❌ Loss jumps from 2.8 to 41+ → catastrophic forgetting

THE FIX:
  ✅ BN layers are ALWAYS kept in inference mode (trainable=False)
  ✅ Only Conv/Dense layers in the base are unfrozen for Phase 2
  ✅ Loss stays stable during fine-tuning

Expected accuracy: 60–75%

Datasets used: MIXED (archive + Indian Leaf datasets) = 78 classes, ~21k images
  - More data is better for generalization
  - Having both complex + clean backgrounds makes the model more robust

Usage:
    python train_final.py            ← train / resume
    python train_final.py --reset    ← start fresh
"""

import os, json, random, argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from class_mapping import (
    ARCHIVE_MAP, LEAF_DATASET_MAP, PLANT_DATASET_MAP,
    DISPLAY_NAMES, TOXIC_PLANTS, CAUTION_PLANTS, AYUSH_PLANTS
)

# ─── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 25       # head-only, all base frozen
PHASE2_EPOCHS = 35       # fine-tune CONV layers only, BN stays frozen
UNFREEZE_LAST = 15       # unfreeze last N non-BN layers of base
LR            = 1e-4
LR_FINETUNE   = 2e-6    # very low — prevents forgetting
MIN_IMAGES    = 30

OUTPUT_DIR    = Path("output")
RESUME_DIR    = OUTPUT_DIR / "resume_final"
BEST_CKPT     = OUTPUT_DIR / "best_final.weights.h5"
RESUME_CKPT   = RESUME_DIR / "last_epoch.weights.h5"
STATE_FILE    = RESUME_DIR / "state.json"
TFLITE_PATH   = OUTPUT_DIR / "plant_model.tflite"
LABEL_MAP_PATH= OUTPUT_DIR / "label_map.json"

BASE          = Path(__file__).parent.parent.parent
ARCHIVE_ROOT  = BASE / "archive"
LEAF_DS_ROOT  = BASE / "Indian Medicinal Leaves Image Datasets" / "Indian Medicinal Leaves Image Datasets" / "Medicinal Leaf dataset"
PLANT_DS_ROOT = BASE / "Indian Medicinal Leaves Image Datasets" / "Indian Medicinal Leaves Image Datasets" / "Medicinal plant dataset"

OUTPUT_DIR.mkdir(exist_ok=True)
RESUME_DIR.mkdir(exist_ok=True)

if args.reset:
    import shutil
    shutil.rmtree(RESUME_DIR, ignore_errors=True)
    RESUME_DIR.mkdir()
    if BEST_CKPT.exists(): BEST_CKPT.unlink()
    print("🔄 Reset complete.\n")

# ─── Resume helpers ───────────────────────────────────────────────
def save_state(phase, epoch_done, history, best_val=0.0):
    with open(STATE_FILE, "w") as f:
        json.dump({"phase": phase, "epoch_done": epoch_done,
                   "history": history, "best_val": best_val}, f, indent=2)

def load_state():
    if STATE_FILE.exists() and RESUME_CKPT.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None

class ResumeCheckpoint(tf.keras.callbacks.Callback):
    """Saves resume checkpoint + state after every epoch."""
    def __init__(self, phase, history_so_far, best_val=0.0):
        super().__init__()
        self.phase    = phase
        self.history  = {k: list(v) for k, v in history_so_far.items()}
        self.best_val = best_val

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        self.model.save_weights(str(RESUME_CKPT))
        va = logs.get("val_accuracy", 0)
        if va > self.best_val:
            self.best_val = va
        save_state(self.phase, epoch + 1, self.history, self.best_val)
        flag = "📈 NEW BEST!" if va >= self.best_val else ""
        print(f"  💾  Phase {self.phase} | Ep {epoch+1} | val={va:.4f} {flag}")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Collect all images (mixed datasets)
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Step 1: Collecting images from all datasets…\n")
class_to_paths: dict[str, list[Path]] = defaultdict(list)

for root, mapping in [
    (ARCHIVE_ROOT,  ARCHIVE_MAP),
    (LEAF_DS_ROOT,  LEAF_DATASET_MAP),
    (PLANT_DS_ROOT, PLANT_DATASET_MAP)
]:
    if not root.exists():
        print(f"  ⚠  Not found: {root}"); continue
    for folder in root.iterdir():
        if not folder.is_dir(): continue
        cls = mapping.get(folder.name, "SKIP")
        if cls == "SKIP": continue
        imgs = (list(folder.glob("**/*.jpg")) +
                list(folder.glob("**/*.jpeg")) +
                list(folder.glob("**/*.png")))
        class_to_paths[cls].extend(imgs)

valid       = {k: v for k, v in class_to_paths.items() if len(v) >= MIN_IMAGES}
class_names = sorted(valid.keys())
num_classes = len(class_names)
idx_of      = {n: i for i, n in enumerate(class_names)}
total       = sum(len(v) for v in valid.values())

print(f"  ✅ {num_classes} classes | {total} images | avg {total//num_classes}/class\n")
for i, n in enumerate(class_names):
    tag = "⚠" if n in TOXIC_PLANTS else "🌿" if n in AYUSH_PLANTS else " "
    print(f"  [{i:2d}] {tag} {n:24s} {len(valid[n]):4d} imgs")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Splits
# ─────────────────────────────────────────────────────────────────
all_p, all_l = [], []
for n in class_names:
    for p in valid[n]:
        all_p.append(str(p)); all_l.append(idx_of[n])

tr_p, tmp_p, tr_l, tmp_l = train_test_split(
    all_p, all_l, test_size=0.2, random_state=42, stratify=all_l)
va_p, te_p, va_l, te_l = train_test_split(
    tmp_p, tmp_l, test_size=0.5, random_state=42, stratify=tmp_l)

print(f"\n  Train: {len(tr_p)} | Val: {len(va_p)} | Test: {len(te_p)}")
cnt = Counter(tr_l)
class_weights = {i: len(tr_l)/(num_classes * cnt.get(i, 1)) for i in range(num_classes)}

# ─────────────────────────────────────────────────────────────────
# STEP 3: Data pipeline
# ─────────────────────────────────────────────────────────────────
def load_img(path, label, aug=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    if aug:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        img = tf.image.random_hue(img, 0.08)
        img = tf.image.random_crop(
            tf.image.resize_with_pad(img, IMG_SIZE+28, IMG_SIZE+28),
            [IMG_SIZE, IMG_SIZE, 3])
    return tf.clip_by_value(img, 0.0, 1.0), label

def make_ds(p, l, aug=False, shuf=False):
    ds = tf.data.Dataset.from_tensor_slices((p, l))
    if shuf: ds = ds.shuffle(len(p), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: load_img(x, y, aug), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(tr_p, tr_l, aug=True,  shuf=True)
val_ds   = make_ds(va_p, va_l, aug=False, shuf=False)
test_ds  = make_ds(te_p, te_l, aug=False, shuf=False)

# ─────────────────────────────────────────────────────────────────
# STEP 4: Build model — one model used throughout
# ─────────────────────────────────────────────────────────────────
print(f"\n🌿 Step 4: Building model ({num_classes} classes)…\n")

base = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, weights="imagenet", pooling="avg")

# ── CRITICAL FIX: freeze everything including BN ─────────────────
def freeze_all(model):
    """Freeze all layers, including BatchNorm (keeps running stats)."""
    for layer in model.layers:
        layer.trainable = False

def unfreeze_conv_only(model, last_n):
    """
    Unfreeze only the last N NON-BatchNorm layers.
    BN layers stay frozen (inference mode) to prevent feature destruction.
    """
    # First freeze everything
    freeze_all(model)
    # Then unfreeze the last N layers that are NOT BN
    non_bn_layers = [l for l in model.layers
                     if not isinstance(l, tf.keras.layers.BatchNormalization)]
    for layer in non_bn_layers[-last_n:]:
        layer.trainable = True
    bn_count   = sum(1 for l in model.layers
                     if isinstance(l, tf.keras.layers.BatchNormalization))
    conv_count = sum(1 for l in model.layers if l.trainable)
    print(f"  Unfrozen : {conv_count} conv layers")
    print(f"  Frozen BN: {bn_count} batch norm layers (prevents catastrophic forgetting)")

freeze_all(base)

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp)
x   = layers.Dense(512, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.45)(x)
x   = layers.Dense(256, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.35)(x)
out = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inp, out)

# ─────────────────────────────────────────────────────────────────
# Load resume state
# ─────────────────────────────────────────────────────────────────
state = load_state()
if state:
    rphase   = state["phase"]
    repoch   = state["epoch_done"]
    history  = state["history"]
    best_val = state.get("best_val", 0.0)
    print(f"\n♻  Resuming — Phase {rphase}, epoch {repoch} done, best val={best_val:.4f}")
    model.load_weights(str(RESUME_CKPT))
    if rphase == 2:
        print("  Re-applying Phase 2 layer config…")
        unfreeze_conv_only(base, UNFREEZE_LAST)
else:
    rphase, repoch, history, best_val = 1, 0, {}, 0.0
    print("\n🆕  Starting fresh.\n")

def make_cbs(phase, hist, bval):
    return [
        ResumeCheckpoint(phase, hist, bval),
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_CKPT), save_weights_only=True,
            save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.4, patience=4, min_lr=1e-9,
                          monitor="val_loss", verbose=1),
        EarlyStopping(patience=12, restore_best_weights=True,
                      monitor="val_accuracy", verbose=1),
    ]

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Head-only training (ALL base layers frozen incl. BN)
# ─────────────────────────────────────────────────────────────────
if rphase == 1 and repoch < PHASE1_EPOCHS:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
    )
    print(f"\n{'='*64}")
    print(f"  PHASE 1 — All base frozen  (epochs {repoch+1}–{PHASE1_EPOCHS})")
    print(f"  LR: {LR}  |  Label smoothing: 0.1  |  Classes: {num_classes}")
    print(f"{'='*64}\n")

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE1_EPOCHS, initial_epoch=repoch,
        class_weight=class_weights,
        callbacks=make_cbs(1, history, best_val),
    )
    for k, v in h1.history.items():
        history.setdefault(k, [])
        history[k] = history[k][:repoch] + [float(x) for x in v]

    best_val = max(history.get("val_accuracy", [0]))
    save_state(2, 0, history, best_val)
    rphase, repoch = 2, 0

elif rphase == 1:
    rphase, repoch = 2, 0
    print("  Phase 1 already complete.")

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Unfreeze Conv only, BN stays frozen
# ─────────────────────────────────────────────────────────────────
if rphase == 2 and repoch < PHASE2_EPOCHS:
    print(f"\n{'='*64}")
    print(f"  PHASE 2 — Conv fine-tune, BN frozen  "
          f"(epochs {PHASE1_EPOCHS+repoch+1}–{PHASE1_EPOCHS+PHASE2_EPOCHS})")
    print(f"  LR: {LR_FINETUNE}  |  Gradient clipping: 1.0")
    unfreeze_conv_only(base, UNFREEZE_LAST)
    print(f"{'='*64}\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
        initial_epoch=PHASE1_EPOCHS + repoch,
        class_weight=class_weights,
        callbacks=make_cbs(2, history, best_val),
    )
    for k, v in h2.history.items():
        history.setdefault(k, [])
        history[k] = history[k][:PHASE1_EPOCHS+repoch] + [float(x) for x in v]

    save_state(2, PHASE2_EPOCHS, history, max(history.get("val_accuracy", [0])))

# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Evaluating on test set…\n")
if BEST_CKPT.exists():
    model.load_weights(str(BEST_CKPT))
    print("  Loaded best checkpoint.\n")

loss, acc, top3 = model.evaluate(test_ds, verbose=1)

# Restore best for test (without any phase 2 arch changes)
freeze_all(base)   # BN back to inference = consistent eval
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")]
)
loss, acc, top3 = model.evaluate(test_ds, verbose=0)  # re-eval in clean state

print(f"\n{'='*64}")
print(f"  ✅  Test Accuracy : {acc*100:.2f}%")
print(f"      Top-3 Accuracy: {top3*100:.2f}%")
print(f"      Loss          : {loss:.4f}")
print(f"{'='*64}")

# ─────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────
label_map = {
    str(i): {
        "class_name":       n,
        "display_name":     DISPLAY_NAMES.get(n, n.replace("_"," ").title()),
        "is_toxic":         n in TOXIC_PLANTS,
        "needs_caution":    n in CAUTION_PLANTS,
        "ayush_recognized": n in AYUSH_PLANTS,
    }
    for i, n in enumerate(class_names)
}
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"\n  Label map → {LABEL_MAP_PATH}  ({num_classes} classes)")

# ─────────────────────────────────────────────────────────────────
# EXPORT TFLITE
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Converting to TFLite…\n")

def rep_ds():
    for p in random.sample(tr_p, min(200, len(tr_p))):
        try:
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            yield [np.array(img, dtype=np.float32)[np.newaxis] / 255.0]
        except Exception:
            continue

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations             = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset    = rep_ds
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type      = tf.float32
conv.inference_output_type     = tf.float32

with open(TFLITE_PATH, "wb") as f:
    f.write(conv.convert())

mb = TFLITE_PATH.stat().st_size / 1024 / 1024
print(f"  ✅ {TFLITE_PATH}  ({mb:.1f} MB)")

# ─────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────
ac = history.get("accuracy", [])
vc = history.get("val_accuracy", [])
if ac:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ac, label="Train Acc", lw=2)
    ax.plot(vc, label="Val Acc",   lw=2)
    if len(ac) > PHASE1_EPOCHS:
        ax.axvline(x=PHASE1_EPOCHS, color="gray", ls="--",
                   label=f"Fine-tune @ ep {PHASE1_EPOCHS}\n(BN stays frozen)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"PlantSage AI — {num_classes} Classes — BN-safe Fine-tuning")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUTPUT_DIR / "training_curve_final.png"
    fig.savefig(p, dpi=120); print(f"\n  Curve → {p}")

print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  TRAINING COMPLETE (BN-safe Fine-tuning)         ║
║                                                      ║
║  Classes   : {num_classes:<38}║
║  Accuracy  : {acc*100:.1f}%                                ║
║  Top-3 Acc : {top3*100:.1f}%                                ║
║  Model     : {str(TFLITE_PATH):<38}║
║              ({mb:.1f} MB, ready for app)               ║
║                                                      ║
║  Next:  python server.py                             ║
╚══════════════════════════════════════════════════════╝
""")
