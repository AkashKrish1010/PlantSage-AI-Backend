"""
train.py  —  PlantSage AI Plant Identification Model  (v2 — fixed)
==================================================================
Trains MobileNetV3Small on the combined archive + Indian Medicinal datasets.

KEY FIX over v1:
  ❌ v1 rebuilt the model for Phase 2 → catastrophic forgetting (0.4% val)
  ✅ v2 keeps ONE model, modifies trainable flags in-place → stable fine-tuning

Expected accuracy: 55–70%  (vs 24% with v1 bug)

✅ RESUME SUPPORT: Stop mid-training (Ctrl+C) and re-run to continue.
   python train.py              ← train / resume
   python train.py --reset      ← clear all checkpoints, start fresh
"""

import os, sys, json, random, argparse
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
    ARCHIVE_MAP, LEAF_DATASET_MAP, PLANT_DATASET_MAP, DISPLAY_NAMES,
    TOXIC_PLANTS, CAUTION_PLANTS, AYUSH_PLANTS
)

# ─── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Start fresh (delete checkpoints)")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 20      # ↑ was 10 — freeze base, train head
PHASE2_EPOCHS = 40      # ↑ was 30 — fine-tune last layers
UNFREEZE_LAST = 10      # ↓ was 20 — fewer unfrozen layers = safer
LR            = 1e-4    # phase 1 learning rate
LR_FINETUNE   = 5e-6    # ↓↓ was 1e-5 — much lower = no catastrophic forgetting
MIN_IMAGES    = 30

OUTPUT_DIR    = Path("output")
RESUME_DIR    = OUTPUT_DIR / "resume"
BEST_CKPT     = OUTPUT_DIR / "best_model.weights.h5"
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

# ─── Reset ────────────────────────────────────────────────────────
if args.reset:
    import shutil
    shutil.rmtree(RESUME_DIR, ignore_errors=True)
    RESUME_DIR.mkdir(exist_ok=True)
    for f in [BEST_CKPT]:
        if f.exists(): f.unlink()
    print("🔄 Reset — starting fresh.\n")

# ─── Resume helpers ───────────────────────────────────────────────
def save_state(phase, epoch_done, history):
    with open(STATE_FILE, "w") as f:
        json.dump({"phase": phase, "epoch_done": epoch_done, "history": history}, f, indent=2)

def load_state():
    if STATE_FILE.exists() and RESUME_CKPT.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return None

class ResumeCheckpoint(tf.keras.callbacks.Callback):
    """Saves weights + state after every epoch for crash-safe resume."""
    def __init__(self, phase, history_so_far):
        super().__init__()
        self.phase   = phase
        self.history = {k: list(v) for k, v in history_so_far.items()}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))
        self.model.save_weights(str(RESUME_CKPT))
        save_state(self.phase, epoch + 1, self.history)
        va = logs.get("val_accuracy", 0)
        print(f"  💾 Saved  [phase {self.phase} | epoch {epoch+1}]  val_acc={va:.4f}")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Collect images
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Step 1: Collecting images…\n")
class_to_paths: dict[str, list[Path]] = defaultdict(list)

for root, mapping in [(ARCHIVE_ROOT, ARCHIVE_MAP),
                      (LEAF_DS_ROOT, LEAF_DATASET_MAP),
                      (PLANT_DS_ROOT, PLANT_DATASET_MAP)]:
    if not root.exists():
        print(f"  ⚠️  Not found: {root}"); continue
    for folder in root.iterdir():
        if not folder.is_dir(): continue
        cls = mapping.get(folder.name, "SKIP")
        if cls == "SKIP": continue
        imgs = (list(folder.glob("**/*.jpg")) +
                list(folder.glob("**/*.jpeg")) +
                list(folder.glob("**/*.png")))
        class_to_paths[cls].extend(imgs)

print(f"  Raw classes: {len(class_to_paths)}  |  "
      f"Total images: {sum(len(v) for v in class_to_paths.values())}")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Filter & finalise class list
# ─────────────────────────────────────────────────────────────────
print(f"\n🌿 Step 2: Dropping classes < {MIN_IMAGES} images…")
valid_classes = {k: v for k, v in class_to_paths.items() if len(v) >= MIN_IMAGES}
skipped       = {k: len(v) for k, v in class_to_paths.items() if len(v) <  MIN_IMAGES}
if skipped: print(f"  Skipped: {skipped}")

class_names  = sorted(valid_classes.keys())
num_classes  = len(class_names)
class_to_idx = {n: i for i, n in enumerate(class_names)}
print(f"  ✅ {num_classes} plant classes → {sum(len(v) for v in valid_classes.values())} images total")

# ─────────────────────────────────────────────────────────────────
# STEP 3: Splits (deterministic seed=42)
# ─────────────────────────────────────────────────────────────────
all_paths, all_labels = [], []
for name in class_names:
    for p in valid_classes[name]:
        all_paths.append(str(p))
        all_labels.append(class_to_idx[name])

tr_p, tmp_p, tr_l, tmp_l = train_test_split(
    all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
va_p, te_p, va_l, te_l = train_test_split(
    tmp_p, tmp_l, test_size=0.5, random_state=42, stratify=tmp_l)

print(f"\n  Train: {len(tr_p)} | Val: {len(va_p)} | Test: {len(te_p)}")

cnt = Counter(tr_l)
class_weights = {i: len(tr_l) / (num_classes * cnt.get(i, 1)) for i in range(num_classes)}

# ─────────────────────────────────────────────────────────────────
# STEP 4: tf.data pipeline
# ─────────────────────────────────────────────────────────────────
def load_img(path, label, augment=False):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_hue(img, 0.05)
        # Random crop for slight translation/zoom effect
        img = tf.image.random_crop(
            tf.image.resize_with_pad(img, IMG_SIZE + 28, IMG_SIZE + 28),
            [IMG_SIZE, IMG_SIZE, 3])
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label

def make_ds(paths, labels, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: load_img(p, l, augment), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(tr_p, tr_l, augment=True,  shuffle=True)
val_ds   = make_ds(va_p, va_l, augment=False, shuffle=False)
test_ds  = make_ds(te_p, te_l, augment=False, shuffle=False)

# ─────────────────────────────────────────────────────────────────
# STEP 5: Build model ONCE (never rebuild — prevents catastrophic forgetting)
# ─────────────────────────────────────────────────────────────────
print(f"\n🌿 Step 5: Building model ({num_classes} classes)…\n")

base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base_model.trainable = False          # start frozen
print(f"  Base layers: {len(base_model.layers)}")

inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x    = base_model(inp)                # ← no training= arg; uses layer's trainable state
x    = layers.Dense(512, activation="relu")(x)
x    = layers.BatchNormalization()(x)
x    = layers.Dropout(0.4)(x)
x    = layers.Dense(256, activation="relu")(x)
x    = layers.Dropout(0.3)(x)
out  = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inp, out)

# ─────────────────────────────────────────────────────────────────
# Load resume state
# ─────────────────────────────────────────────────────────────────
state = load_state()
if state:
    resume_phase   = state["phase"]
    resume_epoch   = state["epoch_done"]
    history_so_far = state["history"]
    print(f"\n♻️  Resuming — Phase {resume_phase}, epoch {resume_epoch} done.")
    model.load_weights(str(RESUME_CKPT))

    # If resuming phase 2, restore fine-tune layer config
    if resume_phase == 2:
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAST]:
            layer.trainable = False
        print(f"  Re-applied: last {UNFREEZE_LAST} base layers unfrozen.")
else:
    resume_phase   = 1
    resume_epoch   = 0
    history_so_far = {}
    print("\n🆕 No checkpoint found — starting fresh.\n")

# ─────────────────────────────────────────────────────────────────
# Common callbacks (used in both phases)
# ─────────────────────────────────────────────────────────────────
def make_callbacks(phase, history_so_far):
    return [
        ResumeCheckpoint(phase=phase, history_so_far=history_so_far),
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_CKPT), save_weights_only=True,
            save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-8,
                          monitor="val_loss", verbose=1),
        EarlyStopping(patience=12, restore_best_weights=True,
                      monitor="val_accuracy", verbose=1),
    ]

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Train head only (base frozen)
# ─────────────────────────────────────────────────────────────────
if resume_phase == 1 and resume_epoch < PHASE1_EPOCHS:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    print(f"\n{'='*60}")
    print(f"  Phase 1 — Head-only training  "
          f"(epochs {resume_epoch+1}–{PHASE1_EPOCHS})")
    print(f"  Base model: FROZEN  |  LR: {LR}")
    print(f"{'='*60}\n")

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        initial_epoch=resume_epoch,
        class_weight=class_weights,
        callbacks=make_callbacks(phase=1, history_so_far=history_so_far),
    )
    for k, v in h1.history.items():
        history_so_far.setdefault(k, [])
        history_so_far[k] = history_so_far[k][:resume_epoch] + [float(x) for x in v]

    # Mark phase 1 as done
    save_state(phase=2, epoch_done=0, history=history_so_far)
    resume_phase = 2
    resume_epoch = 0

elif resume_phase == 1:
    # Phase 1 was fully complete in a previous run
    print("  Phase 1 already complete — skipping to Phase 2.")
    resume_phase = 2
    resume_epoch = 0

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Fine-tune last UNFREEZE_LAST layers
# KEY FIX: We modify the SAME model in-place,
#          NOT rebuild it → no catastrophic forgetting
# ─────────────────────────────────────────────────────────────────
if resume_phase == 2 and resume_epoch < PHASE2_EPOCHS:

    # Unfreeze only the last UNFREEZE_LAST layers of the base
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n{'='*60}")
    print(f"  Phase 2 — Fine-tuning last {UNFREEZE_LAST} base layers  "
          f"(epochs {PHASE1_EPOCHS + resume_epoch + 1}–{PHASE1_EPOCHS + PHASE2_EPOCHS})")
    print(f"  Unfrozen base layers: {trainable_count} / {len(base_model.layers)}")
    print(f"  LR: {LR_FINETUNE}  (gradient clipping: 1.0)")
    print(f"{'='*60}\n")

    # Recompile with much lower LR + gradient clipping
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
        initial_epoch=PHASE1_EPOCHS + resume_epoch,
        class_weight=class_weights,
        callbacks=make_callbacks(phase=2, history_so_far=history_so_far),
    )
    for k, v in h2.history.items():
        history_so_far.setdefault(k, [])
        history_so_far[k] = (history_so_far[k][:PHASE1_EPOCHS + resume_epoch]
                             + [float(x) for x in v])

    save_state(phase=2, epoch_done=PHASE2_EPOCHS, history=history_so_far)

# ─────────────────────────────────────────────────────────────────
# STEP 9: Load best weights & evaluate
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Evaluating on held-out test set…\n")
if BEST_CKPT.exists():
    model.load_weights(str(BEST_CKPT))
    print("  Loaded best checkpoint.")

loss, acc, top3 = model.evaluate(test_ds, verbose=1)
print(f"\n  ✅ Test Accuracy : {acc*100:.2f}%")
print(f"     Top-3 Accuracy: {top3*100:.2f}%")
print(f"     Loss          : {loss:.4f}")

# ─────────────────────────────────────────────────────────────────
# STEP 10: Save label map
# ─────────────────────────────────────────────────────────────────
label_map = {
    str(i): {
        "class_name":       name,
        "display_name":     DISPLAY_NAMES.get(name, name.replace("_", " ").title()),
        "is_toxic":         name in TOXIC_PLANTS,
        "needs_caution":    name in CAUTION_PLANTS,
        "ayush_recognized": name in AYUSH_PLANTS,
    }
    for i, name in enumerate(class_names)
}
with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"\n  Label map → {LABEL_MAP_PATH}  ({num_classes} classes)")

# ─────────────────────────────────────────────────────────────────
# STEP 11: Export TFLite (int8 quantized)
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Converting to TFLite…\n")

def rep_dataset():
    for path in random.sample(tr_p, min(150, len(tr_p))):
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            yield [arr[np.newaxis, ...]]
        except Exception:
            continue

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.float32
converter.inference_output_type     = tf.float32

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = TFLITE_PATH.stat().st_size / 1024 / 1024
print(f"  ✅ {TFLITE_PATH}  ({size_mb:.1f} MB)")

# ─────────────────────────────────────────────────────────────────
# STEP 12: Training curve
# ─────────────────────────────────────────────────────────────────
ac = history_so_far.get("accuracy",     [])
vc = history_so_far.get("val_accuracy", [])
if ac:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ac, label="Train Acc", linewidth=2)
    ax.plot(vc, label="Val Acc",   linewidth=2)
    if len(ac) > PHASE1_EPOCHS:
        ax.axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--",
                   label=f"Fine-tune start (ep {PHASE1_EPOCHS})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"PlantSage AI Training — {num_classes} plant classes")
    fig.tight_layout()
    curve = OUTPUT_DIR / "training_curve.png"
    fig.savefig(curve, dpi=120)
    print(f"\n  Curve → {curve}")

print(f"""
╔══════════════════════════════════════════════════╗
║  ✅  TRAINING COMPLETE                           ║
║                                                  ║
║  Classes  : {num_classes:<36}║
║  Accuracy : {acc*100:.1f}%   Top-3: {top3*100:.1f}%              ║
║  Model    : {str(TFLITE_PATH):<36}║
║  Labels   : {str(LABEL_MAP_PATH):<36}║
║                                                  ║
║  Next:  python server.py                         ║
╚══════════════════════════════════════════════════╝
""")
