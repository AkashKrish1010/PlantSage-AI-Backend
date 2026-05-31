# VanaVaidhya — Plant Model Training on Google Colab
# =====================================================
# Instructions:
# 1. Upload archive_merged.zip to your Google Drive root folder
# 2. Open this notebook in Google Colab
# 3. Runtime → Change runtime type → T4 GPU
# 4. Run all cells (Ctrl+F9)
# 5. Download model_a.tflite + label_map_a.json from the output

# ── Cell 1: Mount Google Drive ──────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 2: Unzip dataset ──────────────────────────────────────────────────
import zipfile, os, shutil

ZIP_PATH    = '/content/drive/MyDrive/archive_merged.zip'
EXTRACT_DIR = '/content/archive_merged'

if not os.path.exists(EXTRACT_DIR):
    print('Extracting dataset...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall('/content/')
    print('Done!')
else:
    print('Dataset already extracted')

# Verify classes
classes = sorted([d for d in os.listdir(EXTRACT_DIR)
                  if os.path.isdir(os.path.join(EXTRACT_DIR, d))])
total_imgs = sum(
    len(os.listdir(os.path.join(EXTRACT_DIR, c))) for c in classes
)
print(f'\n{len(classes)} classes, {total_imgs} total images')
for c in classes:
    n = len(os.listdir(os.path.join(EXTRACT_DIR, c)))
    print(f'  {c:30s}  {n:4d} imgs')

# ── Cell 3: Check GPU ──────────────────────────────────────────────────────
import tensorflow as tf
print('TF version:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('CUDA built:', tf.test.is_built_with_cuda())

# ── Cell 4: Training config ────────────────────────────────────────────────
from tensorflow import keras
from pathlib import Path
import json, numpy as np

DATA_DIR   = EXTRACT_DIR
OUTPUT_DIR = '/content/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_CKPT  = OUTPUT_DIR + '/best_model_a.weights.h5'
TFLITE_OUT = OUTPUT_DIR + '/model_a.tflite'
LABEL_OUT  = OUTPUT_DIR + '/label_map_a.json'

IMG_SIZE   = 224
BATCH_SIZE = 64    # T4 has 16GB VRAM — can handle large batches
P1_EPOCHS  = 15
P2_EPOCHS  = 25
AUTOTUNE   = tf.data.AUTOTUNE

# Enable mixed precision for faster training
keras.mixed_precision.set_global_policy('mixed_float16')
print('Mixed precision: float16 ON')

# ── Cell 5: Build dataset pipelines ───────────────────────────────────────
class_names = sorted([d for d in os.listdir(DATA_DIR)
                      if os.path.isdir(os.path.join(DATA_DIR, d))])
num_classes = len(class_names)
print(f'{num_classes} classes')

# Save label map
label_map = {
    str(i): {
        'class_name':      c,
        'display_name':    c.replace('_', ' ').title(),
        'is_toxic':        c.lower() in {'lantana'},
        'needs_caution':   c.lower() in {'lantana', 'makoy'},
        'ayush_recognized': True,
    }
    for i, c in enumerate(class_names)
}
with open(LABEL_OUT, 'w') as f:
    json.dump(label_map, f, indent=2)
print(f'Label map saved → {LABEL_OUT}')

# Augmentation — EfficientNetV2S expects [0, 255] inputs (has internal preprocessing)
augment = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
    keras.layers.RandomBrightness(0.25),
    keras.layers.RandomContrast(0.25),
], name='augment')

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.15, subset='training',
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode='int',
)
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.15, subset='validation',
    seed=42, image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, label_mode='int',
)

# KEY FIX: Do NOT divide by 255. EfficientNetV2S includes its own preprocessing
def prep_train(x, y):
    return augment(x, training=True), y   # x stays [0, 255]

def prep_val(x, y):
    return x, y                           # x stays [0, 255]

train_ds = train_ds.map(prep_train, AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds.map(prep_val,   AUTOTUNE).prefetch(AUTOTUNE)

print(f'Train batches: {len(train_ds)}, Val batches: {len(val_ds)}')

# ── Cell 6: Build model ────────────────────────────────────────────────────
def build_model():
    # include_preprocessing=True (default) — model handles [0,255] inputs internally
    base = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_preprocessing=True,    # handles normalization internally
    )
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = keras.layers.GlobalAveragePooling2D()(x)
    x   = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes, dtype='float32')(x)  # float32 for stability
    return keras.Model(inp, out)

def unfreeze_conv_only(model):
    """Keep BatchNorm frozen to prevent catastrophic forgetting."""
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            for l in layer.layers:
                l.trainable = isinstance(l, keras.layers.Conv2D)
    model.layers[1].trainable = True

model = build_model()
model.summary()

# ── Cell 7: Phase 1 — Head training ───────────────────────────────────────
print('\n' + '='*55)
print(f'Phase 1 — Head training ({P1_EPOCHS} epochs)')
print('='*55)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.SparseTopKCategoricalAccuracy(3, name='top3_acc')],
)

history1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=P1_EPOCHS,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            BEST_CKPT, save_best_only=True,
            monitor='val_accuracy', save_weights_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=3, min_lr=1e-6, verbose=1),
    ],
)

# ── Cell 8: Phase 2 — Fine-tune ───────────────────────────────────────────
print('\n' + '='*55)
print(f'Phase 2 — Fine-tune Conv layers ({P2_EPOCHS} epochs)')
print('='*55)

model.load_weights(BEST_CKPT)
unfreeze_conv_only(model)

model.compile(
    optimizer=keras.optimizers.Adam(5e-6),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy',
             keras.metrics.SparseTopKCategoricalAccuracy(3, name='top3_acc')],
)

history2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=P2_EPOCHS,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            BEST_CKPT, save_best_only=True,
            monitor='val_accuracy', save_weights_only=True, verbose=1),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=6,
            restore_best_weights=True, verbose=1),
    ],
)

# ── Cell 9: Evaluate + Export TFLite ──────────────────────────────────────
print('\nFinal evaluation...')
model.load_weights(BEST_CKPT)
loss, acc, top3 = model.evaluate(val_ds)
print(f'\n✅ Accuracy      : {acc*100:.2f}%')
print(f'✅ Top-3 Accuracy: {top3*100:.2f}%')

print('\nExporting TFLite (int8 quantized)...')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_bytes)

size_mb = os.path.getsize(TFLITE_OUT) / 1_048_576
print(f'\n✅ model_a.tflite   : {TFLITE_OUT}  ({size_mb:.1f} MB)')
print(f'✅ label_map_a.json : {LABEL_OUT}')

# ── Cell 10: Copy output to Google Drive (so you don't lose it) ───────────
DRIVE_OUTPUT = '/content/drive/MyDrive/VanaVaidhya_model'
os.makedirs(DRIVE_OUTPUT, exist_ok=True)
shutil.copy(TFLITE_OUT, DRIVE_OUTPUT + '/model_a.tflite')
shutil.copy(LABEL_OUT,  DRIVE_OUTPUT + '/label_map_a.json')
print(f'\n✅ Files saved to Google Drive: {DRIVE_OUTPUT}')
print('Download them from Drive and place in:')
print('  VanaVaidhya/ml/output/model_a.tflite')
print('  VanaVaidhya/ml/output/label_map_a.json')
