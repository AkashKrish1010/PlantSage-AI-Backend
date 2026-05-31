"""
train_archive_merged.py  —  PlantSage AI Real-World Plant Model
==============================================================
Trains EfficientNetV2S on the pre-merged archive dataset.
Run preprocessing separately before this script.

Dataset : c:\\Users\\PAAL-DABBA\\Desktop\\backup-pdd\\archive_merged
          (22 classes, 7755 real-world smartphone images)

Output  : ml/output/model_a.tflite
          ml/output/label_map_a.json

Usage:
    python ml/train_archive_merged.py           # train / resume
    python ml/train_archive_merged.py --reset   # start fresh
"""

import json, shutil, argparse
import tensorflow as tf
from pathlib import Path
from tensorflow import keras

# ── Config ─────────────────────────────────────────────────────────
DATA_DIR   = Path(r"c:\Users\PAAL-DABBA\Desktop\backup-pdd\archive_merged")
OUTPUT_DIR = Path(__file__).parent / "output"
CKPT_DIR   = OUTPUT_DIR / "resume_a"
BEST_CKPT  = OUTPUT_DIR / "best_model_a.weights.h5"
STATE_FILE = CKPT_DIR / "state.json"
TFLITE_OUT = OUTPUT_DIR / "model_a.tflite"
LABEL_OUT  = OUTPUT_DIR / "label_map_a.json"

IMG_SIZE   = 224
BATCH_SIZE = 32     # reduce to 16 if GPU OOM
P1_EPOCHS  = 15    # head-only
P2_EPOCHS  = 25    # fine-tune conv layers
AUTOTUNE   = tf.data.AUTOTUNE

# ── Args ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="Start training from scratch")
args = parser.parse_args()

if args.reset:
    if CKPT_DIR.exists():  shutil.rmtree(CKPT_DIR)
    if BEST_CKPT.exists(): BEST_CKPT.unlink()
    print("🔄 Reset — starting fresh")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU ────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"🚀 GPU: {gpus[0].name}  |  float16 mixed precision ON")
else:
    print("⚠️  No GPU — CPU training (slow)")

# ── Classes ────────────────────────────────────────────────────────
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
num_classes = len(class_names)
print(f"\n📂 Dataset : {DATA_DIR}")
print(f"   Classes : {num_classes}  →  {class_names}")

# Label map
label_map = {
    str(i): {
        "class_name":       c,
        "display_name":     c.replace("_", " ").title(),
        "is_toxic":         c.lower() in {"lantana"},
        "needs_caution":    c.lower() in {"lantana", "makoy"},
        "ayush_recognized": True,
    }
    for i, c in enumerate(class_names)
}
with open(LABEL_OUT, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"✅ Label map saved → {LABEL_OUT}")

# ── Dataset ────────────────────────────────────────────────────────
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomBrightness(0.25),
    keras.layers.RandomContrast(0.25),
], name="augment")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.15, subset="training",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="int",
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.15, subset="validation",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode="int",
)

def prep_train(x, y):
    return augment(tf.cast(x, tf.float32) / 255.0, training=True), y

def prep_val(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(prep_train, AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(prep_val,   AUTOTUNE).prefetch(AUTOTUNE)

# ── Model ──────────────────────────────────────────────────────────
def build_model():
    base = keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, dtype="float32")(x)
    return keras.Model(inp, out)

def unfreeze_conv_only(model):
    """Keep BatchNorm frozen — prevents catastrophic forgetting."""
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for l in layer.layers:
                l.trainable = isinstance(l, keras.layers.Conv2D)
    model.layers[1].trainable = True

# ── Resume state ───────────────────────────────────────────────────
state = {"phase": 1, "epoch": 0}
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"▶️  Resuming: Phase {state['phase']}, epoch {state['epoch']}")

model = build_model()
if BEST_CKPT.exists():
    model.load_weights(BEST_CKPT)
    print("   Loaded best weights ✓")

class SaveState(keras.callbacks.Callback):
    def __init__(self, next_phase):
        self.next_phase = next_phase
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(str(CKPT_DIR / "last.weights.h5"))
        with open(STATE_FILE, "w") as f:
            json.dump({"phase": self.next_phase, "epoch": epoch + 1}, f)

# ── Phase 1 : Head only ────────────────────────────────────────────
if state["phase"] == 1:
    print(f"\n{'='*55}")
    print(f"🌿 Phase 1 — Head training  ({P1_EPOCHS} epochs)")
    print(f"{'='*55}")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3_acc")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=P1_EPOCHS, initial_epoch=state.get("epoch", 0),
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(BEST_CKPT), save_best_only=True,
                monitor="val_accuracy", save_weights_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy", factor=0.5,
                patience=3, min_lr=1e-6, verbose=1),
            SaveState(next_phase=2),
        ],
    )
    state = {"phase": 2, "epoch": 0}

# ── Phase 2 : Fine-tune Conv layers ───────────────────────────────
if state["phase"] == 2:
    print(f"\n{'='*55}")
    print(f"🌿 Phase 2 — Fine-tune Conv layers  ({P2_EPOCHS} epochs)")
    print(f"{'='*55}")
    if BEST_CKPT.exists():
        model.load_weights(BEST_CKPT)
    unfreeze_conv_only(model)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3_acc")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=P2_EPOCHS, initial_epoch=state.get("epoch", 0),
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                str(BEST_CKPT), save_best_only=True,
                monitor="val_accuracy", save_weights_only=True, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=6,
                restore_best_weights=True, verbose=1),
            SaveState(next_phase=2),
        ],
    )

# ── Evaluate ───────────────────────────────────────────────────────
print("\n🌿 Final evaluation...")
if BEST_CKPT.exists():
    model.load_weights(BEST_CKPT)
loss, acc, top3 = model.evaluate(val_ds, verbose=1)
print(f"\n  ✅ Accuracy      : {acc*100:.2f}%")
print(f"  ✅ Top-3 Accuracy: {top3*100:.2f}%")

# ── Export TFLite ──────────────────────────────────────────────────
print("\n🌿 Exporting TFLite (int8 quantized)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_bytes)

size_mb = TFLITE_OUT.stat().st_size / 1_048_576
print(f"\n{'='*55}")
print(f"✅ model_a.tflite   : {TFLITE_OUT}  ({size_mb:.1f} MB)")
print(f"✅ label_map_a.json : {LABEL_OUT}  ({num_classes} classes)")
print(f"{'='*55}")
print(f"\n▶  Start server: python ml/server.py")
