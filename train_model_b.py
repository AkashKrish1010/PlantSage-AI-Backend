"""
train_model_b.py  —  Leaf Identifier Model (leaf_clean dataset)
================================================================
• Input:     leaf_clean/   (white-background isolated leaf photos)
• Model:     EfficientNetV2S  (stronger feature extraction for fine details)
• Expected:  88-94% accuracy (clean dataset, controlled background)
• Output:    ml/output/model_b.tflite  +  ml/output/label_map_b.json
• Resume:    Fully supported — re-run to continue from last checkpoint
"""

import os, json, argparse, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras

# ── Config ────────────────────────────────────────────────────────
DATA_DIR    = Path(r"c:\Users\PAAL-DABBA\Desktop\backup-pdd\leaf_clean")
OUTPUT_DIR  = Path(__file__).parent / "output"
CKPT_DIR    = OUTPUT_DIR / "resume_b"
BEST_CKPT   = OUTPUT_DIR / "best_model_b.weights.h5"
STATE_FILE  = CKPT_DIR / "state.json"
TFLITE_OUT  = OUTPUT_DIR / "model_b.tflite"
LABEL_MAP_B = OUTPUT_DIR / "label_map_b.json"

IMG_SIZE    = 224
BATCH_SIZE  = 32
P1_EPOCHS   = 15
P2_EPOCHS   = 20
AUTOTUNE    = tf.data.AUTOTUNE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true")
args = parser.parse_args()

if args.reset and CKPT_DIR.exists():
    shutil.rmtree(CKPT_DIR)
    if BEST_CKPT.exists(): BEST_CKPT.unlink()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("🔄 Reset: starting fresh")

# ── GPU Setup ─────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"🚀 GPU: {gpus[0].name}  |  Mixed precision: float16")
else:
    print("⚠️  No GPU found — running on CPU")

# ── Dataset ───────────────────────────────────────────────────────
class_names = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
num_classes = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}
print(f"\n📂 Dataset: {DATA_DIR.name}")
print(f"   Classes : {num_classes}")

label_map = {str(i): {"class_name": c, "display_name": c.replace("_", " ").title()}
             for i, c in enumerate(class_names)}
with open(LABEL_MAP_B, "w") as f:
    json.dump(label_map, f, indent=2)

# Leaf dataset: subtle augmentations (already clean background)
augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.25),      # leaves are often rotated
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomBrightness(0.1),     # less brightness shift (controlled lighting)
    keras.layers.RandomContrast(0.15),
], name="augment")

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.15,
    subset="training",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.15,
    subset="validation",
    seed=42,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

def prep_train(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = augment(x, training=True)
    return x, y

def prep_val(x, y):
    return tf.cast(x, tf.float32) / 255.0, y

train_ds = train_ds.map(prep_train, AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(prep_val,   AUTOTUNE).prefetch(AUTOTUNE)

# ── Build model ───────────────────────────────────────────────────
def build_model():
    base = keras.applications.EfficientNetV2S(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, dtype="float32")(x)
    return keras.Model(inputs, outputs)

def unfreeze_conv_only(model):
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for l in layer.layers:
                if isinstance(l, keras.layers.Conv2D):
                    l.trainable = True
                elif isinstance(l, keras.layers.BatchNormalization):
                    l.trainable = False
    model.layers[1].trainable = True

# ── Resume state ──────────────────────────────────────────────────
state = {"phase": 1, "epoch": 0}
if STATE_FILE.exists():
    with open(STATE_FILE) as f:
        state = json.load(f)
    print(f"▶️  Resuming from Phase {state['phase']}, epoch {state['epoch']}")

model = build_model()
if BEST_CKPT.exists():
    model.load_weights(BEST_CKPT)
    print("   Loaded best weights")

class SaveState(keras.callbacks.Callback):
    def __init__(self, phase):
        self.phase = phase
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(str(CKPT_DIR / "last_epoch.weights.h5"))
        with open(STATE_FILE, "w") as f:
            json.dump({"phase": self.phase, "epoch": epoch + 1}, f)

# ── Phase 1: Head training ────────────────────────────────────────
if state["phase"] == 1:
    print(f"\n🌿 Phase 1 — Head training ({P1_EPOCHS} epochs)")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(3, "top3")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=P1_EPOCHS, initial_epoch=state.get("epoch", 0),
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(BEST_CKPT), save_best_only=True,
                monitor="val_accuracy", save_weights_only=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                factor=0.5, patience=3, min_lr=1e-6),
            SaveState(phase=2),
        ],
    )
    state = {"phase": 2, "epoch": 0}

# ── Phase 2: Fine-tune ────────────────────────────────────────────
if state["phase"] == 2:
    print(f"\n🌿 Phase 2 — Fine-tuning Conv layers ({P2_EPOCHS} epochs)")
    if BEST_CKPT.exists():
        model.load_weights(BEST_CKPT)
    unfreeze_conv_only(model)
    model.compile(
        optimizer=keras.optimizers.Adam(5e-6),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(3, "top3")],
    )
    model.fit(
        train_ds, validation_data=val_ds,
        epochs=P2_EPOCHS, initial_epoch=state.get("epoch", 0),
        callbacks=[
            keras.callbacks.ModelCheckpoint(str(BEST_CKPT), save_best_only=True,
                monitor="val_accuracy", save_weights_only=True),
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            SaveState(phase=2),
        ],
    )

# ── Evaluate ──────────────────────────────────────────────────────
print("\n🌿 Evaluating...")
if BEST_CKPT.exists():
    model.load_weights(BEST_CKPT)
model.evaluate(val_ds)

# ── Convert to TFLite ────────────────────────────────────────────
print("\n🌿 Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(TFLITE_OUT, "wb") as f:
    f.write(tflite_model)

size_mb = TFLITE_OUT.stat().st_size / 1_048_576
print(f"\n✅ Model B saved: {TFLITE_OUT}  ({size_mb:.1f} MB)")
print(f"✅ Label map  B: {LABEL_MAP_B}  ({num_classes} classes)")
