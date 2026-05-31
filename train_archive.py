"""
train_archive.py — PlantSage AI  (Archive-Only Model)
=====================================================
Trains on ONLY the archive/ dataset — 21 real-world plant classes.

WHY archive-only is better for the app:
  ✅ Archive = real smartphone field photos (exactly what the camera captures)
  ✅ Consistent visual domain → no domain mismatch
  ✅ ~400 images/class average → sufficient for high accuracy
  ✅ Expected accuracy: 85–92%  (vs 29% with mixed datasets)

Classes: ashoka, bael, barlaria, bamboo, clerodendrum, curry_leaf,
         custard_apple, guava, harsingar, hibiscus, jackfruit, kachnar,
         lantana, lemon, makoy, marigold, mint, nasturtium, papaya,
         rose, scarlet_sage, tulsi

Output: output/plant_model.tflite  +  output/label_map.json

Usage:
    python train_archive.py          ← train / resume
    python train_archive.py --reset  ← start fresh
"""

import os, json, random, argparse
from pathlib import Path
from collections import Counter

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

from class_mapping import ARCHIVE_MAP, DISPLAY_NAMES, TOXIC_PLANTS, CAUTION_PLANTS, AYUSH_PLANTS

# ─── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

# ─── Config ───────────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
PHASE1_EPOCHS = 15       # head-only, base frozen
PHASE2_EPOCHS = 35       # fine-tune top layers
UNFREEZE_LAST = 15       # more layers since smaller class count = easier task
LR            = 1e-4
LR_FINETUNE   = 3e-6     # very conservative: preserve ImageNet features
MIN_IMAGES    = 30

OUTPUT_DIR    = Path("output")
RESUME_DIR    = OUTPUT_DIR / "resume_archive"
BEST_CKPT     = OUTPUT_DIR / "best_archive.weights.h5"
RESUME_CKPT   = RESUME_DIR / "last_epoch.weights.h5"
STATE_FILE    = RESUME_DIR / "state.json"
TFLITE_PATH   = OUTPUT_DIR / "plant_model.tflite"
LABEL_MAP_PATH= OUTPUT_DIR / "label_map.json"

BASE          = Path(__file__).parent.parent.parent   # backup-pdd/
ARCHIVE_ROOT  = BASE / "archive"

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
        print(f"  💾 Saved  [phase {self.phase} | epoch {epoch+1}]  val_acc={va:.4f}  "
              f"({'📈 NEW BEST!' if va >= max(self.history.get('val_accuracy', [0])[:epoch] or [0]) else ''})")

# ─────────────────────────────────────────────────────────────────
# STEP 1: Collect archive images only
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Step 1: Collecting archive/ images only…\n")

from collections import defaultdict
class_to_paths: dict[str, list[Path]] = defaultdict(list)

if not ARCHIVE_ROOT.exists():
    print(f"  ❌ Archive not found: {ARCHIVE_ROOT}")
    exit(1)

for folder in sorted(ARCHIVE_ROOT.iterdir()):
    if not folder.is_dir(): continue
    cls = ARCHIVE_MAP.get(folder.name, "SKIP")
    if cls == "SKIP": continue
    imgs = (list(folder.glob("**/*.jpg")) +
            list(folder.glob("**/*.jpeg")) +
            list(folder.glob("**/*.png")))
    class_to_paths[cls].extend(imgs)
    tag = "⚠️" if cls in TOXIC_PLANTS else "🌿" if cls in AYUSH_PLANTS else "  "
    print(f"  {tag} [{cls:20s}] {len(imgs):4d} imgs ← {folder.name}")

# Filter by minimum images
valid = {k: v for k, v in class_to_paths.items() if len(v) >= MIN_IMAGES}
skipped = {k: len(v) for k, v in class_to_paths.items() if len(v) < MIN_IMAGES}
if skipped: print(f"\n  Skipped (too few): {skipped}")

class_names  = sorted(valid.keys())
num_classes  = len(class_names)
class_to_idx = {n: i for i, n in enumerate(class_names)}
total_imgs   = sum(len(v) for v in valid.values())

print(f"\n  ✅ {num_classes} plant classes  |  {total_imgs} total images")
print(f"     Avg images/class: {total_imgs // num_classes}")
print("\n  Classes:")
for i, name in enumerate(class_names):
    disp = DISPLAY_NAMES.get(name, name)
    cnt  = len(valid[name])
    tag  = "⚠️ TOXIC" if name in TOXIC_PLANTS else "🌿 AYUSH" if name in AYUSH_PLANTS else "       "
    print(f"    [{i:2d}] {name:20s} {cnt:4d} imgs  {tag}  {disp}")

# ─────────────────────────────────────────────────────────────────
# STEP 2: Splits
# ─────────────────────────────────────────────────────────────────
all_paths, all_labels = [], []
for name in class_names:
    for p in valid[name]:
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
# STEP 3: tf.data pipeline
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
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        img = tf.image.random_hue(img, 0.08)
        # Simulate different zoom levels
        img = tf.image.random_crop(
            tf.image.resize_with_pad(img, IMG_SIZE + 32, IMG_SIZE + 32),
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
# STEP 4: Build model (ONE model used for BOTH phases)
# ─────────────────────────────────────────────────────────────────
print(f"\n🌿 Step 4: Building MobileNetV3Small ({num_classes} classes)…\n")

base_model = MobileNetV3Small(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base_model.trainable = False    # start frozen

inp  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x    = base_model(inp)
x    = layers.Dense(512, activation="relu")(x)
x    = layers.BatchNormalization()(x)
x    = layers.Dropout(0.45)(x)
x    = layers.Dense(256, activation="relu")(x)
x    = layers.BatchNormalization()(x)
x    = layers.Dropout(0.35)(x)
out  = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inp, out)

total_params = model.count_params()
print(f"  Total params: {total_params:,}")
print(f"  Base layers : {len(base_model.layers)}")

# ─────────────────────────────────────────────────────────────────
# Load resume state
# ─────────────────────────────────────────────────────────────────
state = load_state()
if state:
    resume_phase   = state["phase"]
    resume_epoch   = state["epoch_done"]
    history_so_far = state["history"]
    print(f"\n♻️  Resuming — Phase {resume_phase}, epoch {resume_epoch} already done.")
    model.load_weights(str(RESUME_CKPT))
    if resume_phase == 2:
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAST]:
            layer.trainable = False
        print(f"  Re-applied fine-tune: last {UNFREEZE_LAST} layers unfrozen.")
else:
    resume_phase   = 1
    resume_epoch   = 0
    history_so_far = {}
    print("\n🆕 No checkpoint — starting fresh.\n")

# ─── Callbacks factory ────────────────────────────────────────────
def make_callbacks(phase, history_so_far):
    return [
        ResumeCheckpoint(phase=phase, history_so_far=history_so_far),
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_CKPT), save_weights_only=True,
            save_best_only=True, monitor="val_accuracy", verbose=1),
        ReduceLROnPlateau(
            factor=0.4, patience=4, min_lr=1e-9,
            monitor="val_loss", verbose=1),
        EarlyStopping(
            patience=12, restore_best_weights=True,
            monitor="val_accuracy", verbose=1),
    ]

# ─────────────────────────────────────────────────────────────────
# PHASE 1: Train head only
# ─────────────────────────────────────────────────────────────────
if resume_phase == 1 and resume_epoch < PHASE1_EPOCHS:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    print(f"\n{'='*62}")
    print(f"  PHASE 1 — Head-only  (epochs {resume_epoch+1}–{PHASE1_EPOCHS})")
    print(f"  Base: FROZEN  |  LR: {LR}  |  Classes: {num_classes}")
    print(f"{'='*62}\n")

    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        initial_epoch=resume_epoch,
        class_weight=class_weights,
        callbacks=make_callbacks(1, history_so_far),
    )
    for k, v in h1.history.items():
        history_so_far.setdefault(k, [])
        history_so_far[k] = history_so_far[k][:resume_epoch] + [float(x) for x in v]

    save_state(phase=2, epoch_done=0, history=history_so_far)
    resume_phase = 2
    resume_epoch = 0

elif resume_phase == 1:
    print("  Phase 1 already complete — jumping to Phase 2.")
    resume_phase = 2
    resume_epoch = 0

# ─────────────────────────────────────────────────────────────────
# PHASE 2: Fine-tune last UNFREEZE_LAST layers
# ─────────────────────────────────────────────────────────────────
if resume_phase == 2 and resume_epoch < PHASE2_EPOCHS:
    # Unfreeze in-place on the SAME model (no rebuild → no forgetting)
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST]:
        layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n{'='*62}")
    print(f"  PHASE 2 — Fine-tune  (epochs {PHASE1_EPOCHS + resume_epoch + 1}–{PHASE1_EPOCHS + PHASE2_EPOCHS})")
    print(f"  Unfrozen: {unfrozen}/{len(base_model.layers)} base layers")
    print(f"  LR: {LR_FINETUNE}  |  Gradient clipping: 1.0")
    print(f"{'='*62}\n")

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
        callbacks=make_callbacks(2, history_so_far),
    )
    for k, v in h2.history.items():
        history_so_far.setdefault(k, [])
        history_so_far[k] = (history_so_far[k][:PHASE1_EPOCHS + resume_epoch]
                             + [float(x) for x in v])

    save_state(phase=2, epoch_done=PHASE2_EPOCHS, history=history_so_far)

# ─────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Evaluating on held-out test set…\n")
if BEST_CKPT.exists():
    model.load_weights(str(BEST_CKPT))
loss, acc, top3 = model.evaluate(test_ds, verbose=1)
print(f"\n{'='*62}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Top-3 Accuracy: {top3*100:.2f}%")
print(f"  Loss          : {loss:.4f}")
print(f"{'='*62}")

# ─────────────────────────────────────────────────────────────────
# Save label map  (21 classes, archive only)
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
print("  Classes:", ", ".join(DISPLAY_NAMES.get(n, n) for n in class_names))

# ─────────────────────────────────────────────────────────────────
# Export TFLite (int8 quantized)
# ─────────────────────────────────────────────────────────────────
print("\n🌿 Converting to TFLite…\n")

def rep_dataset():
    for path in random.sample(tr_p, min(200, len(tr_p))):
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
# Training curve
# ─────────────────────────────────────────────────────────────────
ac = history_so_far.get("accuracy",     [])
vc = history_so_far.get("val_accuracy", [])
if ac:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(ac, label="Train Acc", linewidth=2)
    ax1.plot(vc, label="Val Acc",   linewidth=2)
    if len(ac) > PHASE1_EPOCHS:
        ax1.axvline(x=PHASE1_EPOCHS, color="gray", linestyle="--",
                    label=f"Fine-tune @ ep {PHASE1_EPOCHS}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Accuracy — {num_classes} Archive Plant Classes")
    ax1.legend(); ax1.grid(alpha=0.3)

    ltr = history_so_far.get("loss",     [])
    lva = history_so_far.get("val_loss", [])
    ax2.plot(ltr, label="Train Loss", linewidth=2)
    ax2.plot(lva, label="Val Loss",   linewidth=2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    curve = OUTPUT_DIR / "training_curve_archive.png"
    fig.savefig(curve, dpi=120)
    print(f"\n  Curve → {curve}")

print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  ARCHIVE MODEL TRAINING COMPLETE                 ║
║                                                      ║
║  Classes    : {num_classes:<38}║
║  Accuracy   : {acc*100:.1f}%                                ║
║  Top-3 Acc  : {top3*100:.1f}%                                ║
║  Model      : {str(TFLITE_PATH):<38}║
║  Labels     : {str(LABEL_MAP_PATH):<38}║
║                                                      ║
║  Next:  python server.py                             ║
╚══════════════════════════════════════════════════════╝
""")
