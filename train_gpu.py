"""
train_gpu.py — PlantSage AI  (GPU Edition — RTX 3050 6GB)
=========================================================
Uses EfficientNetV2S for maximum accuracy on GPU.

RTX 3050 6GB advantages:
  ✅ Much larger model (EfficientNetV2S: 21M params vs MobileNet: 2.5M)
  ✅ Batch size 128 (vs 32 on CPU)
  ✅ Mixed precision float16 → faster, less VRAM
  ✅ Training time: 25–40 min (vs 2+ hrs on CPU)
  ✅ Expected accuracy: 82–90% (vs 25–35% on CPU)

Usage:
    python train_gpu.py            ← train / resume
    python train_gpu.py --reset    ← start fresh
"""

import os, json, random, argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
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

# ─── GPU Setup ────────────────────────────────────────────────────
print("\n🎮 GPU Setup…")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"  ✅ GPU detected: {[g.name for g in gpus]}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)   # prevent OOM
    # Mixed precision: float16 on GPU → 2x faster, half VRAM
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("  ✅ Mixed precision: float16 enabled")
    BATCH_SIZE = 128
else:
    print("  ⚠  No GPU found — falling back to CPU. Training will be slow.")
    print("     Check CUDA installation with: python check_gpu.py")
    BATCH_SIZE = 32

# ─── Config ───────────────────────────────────────────────────────
IMG_SIZE      = 300       # EfficientNetV2S optimal input size
PHASE1_EPOCHS = 20        # head-only, all base frozen
PHASE2_EPOCHS = 40        # fine-tune conv layers (BN stays frozen)
UNFREEZE_LAST = 20        # last N non-BN layers to unfreeze
LR            = 2e-4      # higher LR OK on GPU with larger batches
LR_FINETUNE   = 5e-6      # careful fine-tuning
MIN_IMAGES    = 30

OUTPUT_DIR    = Path("output")
RESUME_DIR    = OUTPUT_DIR / "resume_gpu"
BEST_CKPT     = OUTPUT_DIR / "best_gpu.weights.h5"
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
        va = float(logs.get("val_accuracy", 0))
        self.best_val = max(self.best_val, va)
        save_state(self.phase, epoch + 1, self.history, self.best_val)
        flag = " 📈 NEW BEST!" if va >= self.best_val else ""
        print(f"  💾 Phase {self.phase} | Ep {epoch+1} | val_acc={va:.4f}{flag}")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Collect images (all datasets)
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Step 1: Collecting images…\n")
class_to_paths: dict[str, list[Path]] = defaultdict(list)

for root, mapping in [
    (ARCHIVE_ROOT,  ARCHIVE_MAP),
    (LEAF_DS_ROOT,  LEAF_DATASET_MAP),
    (PLANT_DS_ROOT, PLANT_DATASET_MAP),
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
class_weights = {i: len(tr_l)/(num_classes*cnt.get(i,1)) for i in range(num_classes)}

# ─────────────────────────────────────────────────────────────────
# STEP 3: GPU-optimized data pipeline
# ─────────────────────────────────────────────────────────────────
def load_img(path, label, aug=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    if aug:
        # Standard flips and colour
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        img = tf.image.random_hue(img, 0.08)
        # Random crop (simulates different zoom/translation)
        img = tf.image.random_crop(
            tf.image.resize_with_pad(img, IMG_SIZE+32, IMG_SIZE+32),
            [IMG_SIZE, IMG_SIZE, 3])
        # Random 90° rotation using tf.image.rot90
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)

    return tf.clip_by_value(img, 0.0, 1.0), label

def make_ds(p, l, aug=False, shuf=False):
    ds = tf.data.Dataset.from_tensor_slices((p, l))
    if shuf:
        ds = ds.shuffle(len(p), reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: load_img(x, y, aug),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(tr_p, tr_l, aug=True,  shuf=True)
val_ds   = make_ds(va_p, va_l, aug=False, shuf=False)
test_ds  = make_ds(te_p, te_l, aug=False, shuf=False)

# ─────────────────────────────────────────────────────────────────
# STEP 4: Build EfficientNetV2S model
# ─────────────────────────────────────────────────────────────────
print(f"\n🌿 Step 4: Building EfficientNetV2S ({num_classes} classes)…\n")

base = EfficientNetV2S(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

# ── CRITICAL: freeze EVERYTHING including BN at start ─────────────
def freeze_all(m):
    for layer in m.layers:
        layer.trainable = False

def unfreeze_conv_only(m, last_n):
    """Unfreeze last N non-BN layers. BN ALWAYS stays in inference mode."""
    freeze_all(m)
    non_bn = [l for l in m.layers
              if not isinstance(l, tf.keras.layers.BatchNormalization)]
    for layer in non_bn[-last_n:]:
        layer.trainable = True
    unfrozen = sum(1 for l in m.layers if l.trainable)
    frozen_bn = sum(1 for l in m.layers
                    if isinstance(l, tf.keras.layers.BatchNormalization))
    print(f"  Conv unfrozen: {unfrozen}  |  BN frozen (inference mode): {frozen_bn}")

freeze_all(base)
print(f"  Base model: EfficientNetV2S  |  Layers: {len(base.layers)}")
print(f"  Input size: {IMG_SIZE}×{IMG_SIZE}  |  Batch: {BATCH_SIZE}")

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp)
x   = layers.Dense(1024, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.45)(x)
x   = layers.Dense(512, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.35)(x)
# Use float32 for final output (important with mixed precision)
out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = Model(inp, out)

print(f"  Total params: {model.count_params():,}")

# ─────────────────────────────────────────────────────────────────
# Load resume state
# ─────────────────────────────────────────────────────────────────
state = load_state()
if state:
    rphase, repoch = state["phase"], state["epoch_done"]
    history, bval  = state["history"], state.get("best_val", 0.0)
    print(f"\n♻  Resuming — Phase {rphase} | epoch {repoch} done | best val={bval:.4f}")
    model.load_weights(str(RESUME_CKPT))
    if rphase == 2:
        unfreeze_conv_only(base, UNFREEZE_LAST)
else:
    rphase, repoch, history, bval = 1, 0, {}, 0.0
    print("\n🆕 Starting fresh.\n")

def make_cbs(phase, hist, bv):
    return [
        ResumeCheckpoint(phase, hist, bv),
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_CKPT), save_weights_only=True,
            save_best_only=True, monitor="val_accuracy", verbose=1),
        tf.keras.callbacks.LearningRateScheduler(
            lambda ep, lr: lr * 0.95 if ep > 5 else lr, verbose=0),
        EarlyStopping(patience=10, restore_best_weights=True,
                      monitor="val_accuracy", verbose=1),
    ]

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Head-only (entire base frozen, incl. BN)
# ─────────────────────────────────────────────────────────────────
if rphase == 1 and repoch < PHASE1_EPOCHS:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    print(f"\n{'='*64}")
    print(f"  PHASE 1 — EfficientNetV2S head-only")
    print(f"  Epochs: {repoch+1}–{PHASE1_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
    print(f"{'='*64}\n")

    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE1_EPOCHS, initial_epoch=repoch,
        class_weight=class_weights,
        callbacks=make_cbs(1, history, bval),
    )
    for k, v in h1.history.items():
        history.setdefault(k, [])
        history[k] = history[k][:repoch] + [float(x) for x in v]

    bval = max(history.get("val_accuracy", [0]))
    save_state(2, 0, history, bval)
    rphase, repoch = 2, 0

elif rphase == 1:
    rphase, repoch = 2, 0
    print("  Phase 1 already complete.")

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Fine-tune Conv only, BN stays frozen
# ─────────────────────────────────────────────────────────────────
if rphase == 2 and repoch < PHASE2_EPOCHS:
    print(f"\n{'='*64}")
    print(f"  PHASE 2 — Conv fine-tune (BN stays frozen)")
    print(f"  Epochs: {PHASE1_EPOCHS+repoch+1}–{PHASE1_EPOCHS+PHASE2_EPOCHS}")
    print(f"  LR: {LR_FINETUNE}  |  Gradient clip: 1.0")
    unfreeze_conv_only(base, UNFREEZE_LAST)
    print(f"{'='*64}\n")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    h2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
        initial_epoch=PHASE1_EPOCHS + repoch,
        class_weight=class_weights,
        callbacks=make_cbs(2, history, bval),
    )
    for k, v in h2.history.items():
        history.setdefault(k, [])
        history[k] = (history[k][:PHASE1_EPOCHS+repoch]
                      + [float(x) for x in v])
    save_state(2, PHASE2_EPOCHS, history,
               max(history.get("val_accuracy", [0])))

# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Evaluating on test set…\n")
if BEST_CKPT.exists():
    model.load_weights(str(BEST_CKPT))
    print("  Loaded best checkpoint.\n")
freeze_all(base)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy",
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")])
loss, acc, top3 = model.evaluate(test_ds, verbose=1)

print(f"\n{'='*64}")
print(f"  ✅ Test Accuracy : {acc*100:.2f}%")
print(f"     Top-3 Accuracy: {top3*100:.2f}%")
print(f"     Loss          : {loss:.4f}")
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
print("\n🌿 Converting to TFLite (quantized)…\n")

# Must run inference in float32 for TFLite export
tf.keras.mixed_precision.set_global_policy("float32")
freeze_all(base)

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
# TRAINING CURVE
# ─────────────────────────────────────────────────────────────────
ac = history.get("accuracy", [])
vc = history.get("val_accuracy", [])
if ac:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ac, label="Train", lw=2); ax1.plot(vc, label="Val", lw=2)
    if len(ac) > PHASE1_EPOCHS:
        ax1.axvline(x=PHASE1_EPOCHS, color="gray", ls="--",
                    label=f"Fine-tune @ ep {PHASE1_EPOCHS}")
    ax1.set(title=f"Accuracy — {num_classes} Classes", xlabel="Epoch", ylabel="Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ltr = history.get("loss", []); lva = history.get("val_loss", [])
    ax2.plot(ltr, label="Train", lw=2); ax2.plot(lva, label="Val", lw=2)
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_curve = OUTPUT_DIR / "training_curve_gpu.png"
    fig.savefig(out_curve, dpi=120)
    print(f"\n  Curve → {out_curve}")

print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  GPU TRAINING COMPLETE                           ║
║                                                      ║
║  Model      : EfficientNetV2S (21M params)           ║
║  Classes    : {num_classes:<38}║
║  Accuracy   : {acc*100:.1f}%                                ║
║  Top-3 Acc  : {top3*100:.1f}%                                ║
║  TFLite     : {str(TFLITE_PATH):<38}║
║  Size       : {mb:.1f} MB                                  ║
║                                                      ║
║  Next: python server.py                              ║
╚══════════════════════════════════════════════════════╝
""")
