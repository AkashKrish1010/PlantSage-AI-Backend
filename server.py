"""
server.py  —  VanaVaidhya Plant Identification API  (v2 — EfficientNetV2S / Keras)
====================================================================================
Model: EfficientNetV2S trained on 71 Indian medicinal plant species
       val_accuracy: 96.6%  |  TTA accuracy: 98.4%

NOTE: We load the Keras model (best_v2.weights.h5) directly instead of the TFLite
      file, because the exported TFLite uses SELECT_TF_OPS (Flex ops) which require
      the Flex delegate — not available in the standard desktop TFLite interpreter.
      Loading via Keras/TF is equivalent in accuracy and simpler on desktop.

Input preprocessing (matches training):
  - Resize to 300×300
  - float32 in [0, 255]   ← model's built-in preprocessing rescales to [-1, 1]

TTA (Test-Time Augmentation): optional, enabled with ?tta=true query param.

Usage:
    python ml/server.py

API:  http://localhost:8000
Docs: http://localhost:8000/docs
"""

import io, json, base64
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import gemini_service
import auth_service

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN noise
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, Model

# ─── Paths ────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent          # VanaVaidhya/
MODEL_DIR    = ROOT_DIR / "assets" / "model"
WEIGHTS_PATH = MODEL_DIR / "best_v2.weights.h5"
LABEL_PATH   = MODEL_DIR / "label_map.json"

IMG_SIZE    = 300
NUM_CLASSES = 71   # fixed — matches training

# ─── Build & load Keras model ─────────────────────────────────────
print("🌿 VanaVaidhya AI — Building EfficientNetV2S model...")

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Weights not found: {WEIGHTS_PATH}\n"
        "Copy best_v2.weights.h5 into VanaVaidhya/assets/model/"
    )
if not LABEL_PATH.exists():
    raise FileNotFoundError(
        f"Label map not found: {LABEL_PATH}\n"
        "Copy label_map.json into VanaVaidhya/assets/model/"
    )

# Suppress GPU memory grabs on machines with limited VRAM
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Rebuild the exact same architecture as training (Cell 9 of the notebook)
base = EfficientNetV2S(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights=None,          # we load from weights file below
    pooling="avg",
    # include_preprocessing=True is default — model rescales [0,255]→[-1,1] internally
)

inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp)
x   = layers.Dense(512, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
x   = layers.Dense(256, activation="relu")(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = Model(inp, out)

# Compile (needed so optimizer variables are initialised before loading weights)
model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print(f"  Loading weights from {WEIGHTS_PATH.name} …")
model.load_weights(str(WEIGHTS_PATH))
model.trainable = False   # inference only

# Load label map
with open(LABEL_PATH) as f:
    labels: dict = json.load(f)

print(f"  ✅ Model ready — {NUM_CLASSES} classes | Input: {IMG_SIZE}×{IMG_SIZE}")
print(f"  🎯 Accuracy: 96.6% val  /  98.4% TTA")


# ─── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="VanaVaidhya Plant Identification API",
    description=(
        "EfficientNetV2S trained on 71 Indian medicinal plant species. "
        "Accuracy: 96.6% val / 98.4% TTA."
    ),
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Create MongoDB indexes on startup."""
    try:
        await auth_service.ensure_indexes()
        print("  ✅ MongoDB indexes ensured (users.email, users.google_id)")
    except Exception as e:
        print(f"  ⚠️  MongoDB startup warning: {e}")
        print("     Check MONGODB_URI in ml/.env — MongoDB may not be running.")

# ─── Auth helpers ─────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """Dependency: validate bearer token and return the user payload."""
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = auth_service.decode_token(creds.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


# ─── Preprocessing ────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize + convert to float32 in [0, 255]. Model handles rescaling internally."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)    # already in [0, 255]
    return np.expand_dims(arr, axis=0)        # (1, 300, 300, 3)


def augment(image: Image.Image, seed: int) -> np.ndarray:
    """Single augmented pass for TTA — flip + slight crop."""
    rng = np.random.default_rng(seed)
    img = image.convert("RGB")

    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    w, h = img.size
    pad = int(min(w, h) * 0.08)
    padded = Image.new("RGB", (w + pad * 2, h + pad * 2))
    padded.paste(img, (pad, pad))
    x0 = rng.integers(0, pad * 2 + 1)
    y0 = rng.integers(0, pad * 2 + 1)
    img = padded.crop((x0, y0, x0 + w, y0 + h))

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


# ─── Inference ────────────────────────────────────────────────────
def run_inference(img_array: np.ndarray) -> np.ndarray:
    """Run a single forward pass; return probability vector."""
    return model(img_array, training=False).numpy()[0]


def top5_from_probs(probs: np.ndarray) -> list[dict]:
    """Convert probability array to ranked top-5 prediction list."""
    top5_idx = probs.argsort()[-5:][::-1]
    results = []
    for rank, idx in enumerate(top5_idx, start=1):
        entry = labels.get(str(idx), {})
        results.append({
            "rank":             rank,
            "class_name":       entry.get("class_name", f"class_{idx}"),
            "display_name":     entry.get("display_name", f"class_{idx}"),
            "confidence":       round(float(probs[idx]) * 100, 2),
            "is_toxic":         entry.get("is_toxic", False),
            "needs_caution":    entry.get("needs_caution", False),
            "ayush_recognized": entry.get("ayush_recognized", False),
            "model_source":     "EfficientNetV2S-v2",
        })
    return results


def identify(image: Image.Image, tta: bool = False) -> dict:
    """Core identification — single pass or 4-pass TTA."""
    probs = run_inference(preprocess(image))

    if tta:
        for seed in range(1, 4):
            probs = probs + run_inference(augment(image, seed=seed))
        probs = probs / 4.0
        method = "TTA (4 passes)"
    else:
        method = "single pass"

    top5 = top5_from_probs(probs)
    top  = top5[0]

    return {
        "success":        True,
        "top_prediction": top,
        "top5":           top5,
        "routing":        f"EfficientNetV2S — {method}",
        "message": (
            f"⚠️ {top['display_name']} is toxic — do not consume."
            if top.get("is_toxic") else None
        ),
    }


def decode_b64(b64_string: str) -> Image.Image:
    """Decode a base64 string (with or without data-URI prefix) to PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64_string)))


# ─── Request / Response models ────────────────────────────────────
class IdentifyRequest(BaseModel):
    base64_image: str

class PredictionResult(BaseModel):
    rank:             int
    class_name:       str
    display_name:     str
    confidence:       float
    is_toxic:         bool
    needs_caution:    bool
    ayush_recognized: bool
    model_source:     Optional[str] = None

class IdentifyResponse(BaseModel):
    success:        bool
    top_prediction: PredictionResult
    top5:           list[PredictionResult]
    routing:        Optional[str] = None
    message:        Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":   "VanaVaidhya Plant Identification API v2",
        "model":     "EfficientNetV2S (Keras, float32)",
        "classes":   NUM_CLASSES,
        "accuracy":  "96.6% val / 98.4% TTA",
        "img_size":  f"{IMG_SIZE}x{IMG_SIZE}",
        "status":    "ready",
    }


@app.get("/classes")
def get_classes():
    return {"classes": list(labels.values())}


@app.post("/identify", response_model=IdentifyResponse)
async def identify_base64(
    request: IdentifyRequest,
    tta: bool = Query(default=False, description="Enable Test-Time Augmentation (4 passes)"),
):
    try:
        image  = decode_b64(request.base64_image)
        result = identify(image, tta=tta)
        return IdentifyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Identification failed: {e}")


@app.post("/identify/upload", response_model=IdentifyResponse)
async def identify_upload(
    file: UploadFile = File(...),
    tta:  bool = Query(default=False, description="Enable Test-Time Augmentation (4 passes)"),
):
    try:
        image  = Image.open(io.BytesIO(await file.read()))
        result = identify(image, tta=tta)
        return IdentifyResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {e}")


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       "EfficientNetV2S-v2",
        "num_classes": NUM_CLASSES,
        "img_size":    IMG_SIZE,
        "model_ok":    True,
    }


# ─── Gemini proxy routes ─────────────────────────────────────────
# These endpoints expose Gemini AI features securely from the backend.
# The API key lives in ml/.env — never in the mobile client bundle.

class PlantInfoRequest(BaseModel):
    plant_name: str
    botanical_name: Optional[str] = None
    class_label: Optional[str] = None

class SymptomSearchRequest(BaseModel):
    symptom: str

class VerifyPlantRequest(BaseModel):
    base64_image: str

class VerifyVisionRequest(BaseModel):
    base64_image: str
    ml_plant_name: str
    ml_class_name: str
    ml_confidence: float


@app.post("/ai/plant-info")
async def gemini_plant_info(request: PlantInfoRequest):
    """Get full plant info (hybrid PFAF + Gemini). Mirrors getPlantInfo() in geminiService.ts."""
    try:
        result = await gemini_service.get_plant_info(
            request.plant_name,
            request.botanical_name,
            request.class_label,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini plant info failed: {e}")


@app.post("/ai/symptom-search")
async def gemini_symptom_search(request: SymptomSearchRequest):
    """Search plants by symptom using Gemini. Mirrors searchPlantsBySymptom() in geminiService.ts."""
    try:
        result = await gemini_service.search_plants_by_symptom(request.symptom)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini symptom search failed: {e}")


@app.post("/ai/verify-plant")
async def gemini_verify_plant(request: VerifyPlantRequest):
    """Check if image contains a plant. Mirrors verifyIsPlant() in geminiService.ts."""
    try:
        is_plant = await gemini_service.verify_is_plant(request.base64_image)
        return {"is_plant": is_plant}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini verify plant failed: {e}")


@app.post("/ai/verify-vision")
async def gemini_verify_vision(request: VerifyVisionRequest):
    """Cross-verify ML prediction with Gemini vision. Mirrors verifyPlantWithVision() in geminiService.ts."""
    try:
        result = await gemini_service.verify_plant_with_vision(
            request.base64_image,
            request.ml_plant_name,
            request.ml_class_name,
            request.ml_confidence,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini vision verify failed: {e}")


# ─── Auth Request / Response models ──────────────────────────────
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    id_token: str          # Google ID token from expo-auth-session

class RefreshRequest(BaseModel):
    refresh_token: str

class LogoutRequest(BaseModel):
    refresh_token: str

class AuthResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "Bearer"
    user: dict


# ─── Auth endpoints ───────────────────────────────────────────────

@app.post("/auth/signup", response_model=AuthResponse)
async def auth_signup(req: SignupRequest):
    """Register a new user with name, email, and password."""
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Name is required.")
    if not req.email.strip() or "@" not in req.email:
        raise HTTPException(status_code=422, detail="Valid email is required.")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters.")
    try:
        user = await auth_service.create_email_user(req.name, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {e}")

    access_token  = auth_service.create_access_token(user["id"], user["email"], user["name"])
    refresh_token = auth_service.create_refresh_token(user["id"])
    await auth_service.store_refresh_token(user["id"], refresh_token)
    return AuthResponse(access_token=access_token, refresh_token=refresh_token, user=user)


@app.post("/auth/login", response_model=AuthResponse)
async def auth_login(req: LoginRequest):
    """Login with email + password."""
    doc = await auth_service.get_user_by_email(req.email)
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not doc.get("password_hash"):
        raise HTTPException(status_code=401, detail="This account uses Google Sign-In. Please continue with Google.")
    if not auth_service.verify_password(req.password, doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    from bson import ObjectId
    user_id = str(doc["_id"])
    user    = auth_service._serialize_user(doc)
    access_token  = auth_service.create_access_token(user_id, user["email"], user["name"])
    refresh_token = auth_service.create_refresh_token(user_id)
    await auth_service.store_refresh_token(user_id, refresh_token)
    return AuthResponse(access_token=access_token, refresh_token=refresh_token, user=user)


@app.post("/auth/google", response_model=AuthResponse)
async def auth_google(req: GoogleAuthRequest):
    """Authenticate with a Google ID token obtained via expo-auth-session."""
    google_info = await auth_service.verify_google_id_token(req.id_token)
    if not google_info:
        raise HTTPException(status_code=401, detail="Invalid Google token. Please try again.")

    user = await auth_service.upsert_google_user(
        google_id = google_info["google_id"],
        email     = google_info["email"],
        name      = google_info["name"],
        photo_url = google_info["photo_url"],
    )
    access_token  = auth_service.create_access_token(user["id"], user["email"], user["name"])
    refresh_token = auth_service.create_refresh_token(user["id"])
    await auth_service.store_refresh_token(user["id"], refresh_token)
    return AuthResponse(access_token=access_token, refresh_token=refresh_token, user=user)


@app.post("/auth/refresh")
async def auth_refresh(req: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = auth_service.decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token.")
    is_valid = await auth_service.validate_refresh_token(req.refresh_token)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Refresh token revoked or expired.")

    user_id = payload["sub"]
    doc     = await auth_service.get_user_by_id(user_id)
    if not doc:
        raise HTTPException(status_code=401, detail="User not found.")

    user         = auth_service._serialize_user(doc)
    access_token = auth_service.create_access_token(user_id, user["email"], user["name"])
    return {"access_token": access_token, "token_type": "Bearer", "user": user}


@app.get("/auth/me")
async def auth_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user."""
    doc = await auth_service.get_user_by_id(current_user["sub"])
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    return auth_service._serialize_user(doc)


@app.post("/auth/logout")
async def auth_logout(req: LogoutRequest):
    """Revoke the given refresh token (logout this session)."""
    await auth_service.revoke_refresh_token(req.refresh_token)
    return {"success": True, "message": "Logged out successfully."}


# ─── Run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🚀 VanaVaidhya AI v2 — Server starting...")
    print(f"   Model  : EfficientNetV2S — {NUM_CLASSES} classes")
    print(f"   Accuracy: 96.6% val  /  98.4% TTA")
    print(f"   Docs   : http://localhost:8000/docs")
    print(f"   TTA    : POST /identify?tta=true")
    print(f"   Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
