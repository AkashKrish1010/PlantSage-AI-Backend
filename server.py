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

import io, json, base64, sys
from pathlib import Path
from typing import Optional

# Support console outputs with emojis on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from PIL import Image
import uvicorn
import re
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import gemini_service
import auth_service

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN noise

# ─── Paths ────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent          # VanaVaidhya/
ML_DIR       = Path(__file__).parent

# Check if running in a standalone deployment where files are local to ml/
if (ML_DIR / "label_map.json").exists():
    LABEL_PATH = ML_DIR / "label_map.json"
else:
    LABEL_PATH = ROOT_DIR / "assets" / "model" / "label_map.json"

if (ML_DIR / "best_v2.weights.h5").exists():
    WEIGHTS_PATH = ML_DIR / "best_v2.weights.h5"
else:
    WEIGHTS_PATH = ROOT_DIR / "assets" / "model" / "best_v2.weights.h5"

IMG_SIZE    = 300
NUM_CLASSES = 71   # fixed — matches training

# Load label map
labels = {}
if LABEL_PATH.exists():
    with open(LABEL_PATH, encoding="utf-8") as f:
        labels = json.load(f)
else:
    print(f"⚠️ Label map not found at {LABEL_PATH}")

# Render Free instance has only 512MB RAM, importing tensorflow causes OOM crashes.
# We auto-switch to Gemini-only mode if weights are missing (e.g. ignored by git)
# or if specifically forced via env variable.
USE_LOCAL_MODEL = WEIGHTS_PATH.exists() and os.environ.get("USE_GEMINI_ONLY", "false").lower() != "true"

model = None
if USE_LOCAL_MODEL:
    print("🌿 VanaVaidhya AI — Loading local Keras model...")
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import EfficientNetV2S
        from tensorflow.keras import layers, Model

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

        print(f"  ✅ Local Keras model ready — {NUM_CLASSES} classes | Input: {IMG_SIZE}×{IMG_SIZE}")
        print(f"  🎯 Accuracy: 96.6% val  /  98.4% TTA")
    except Exception as e:
        print(f"  ⚠️ Failed to load local model: {e}")
        print("  🔄 Switching to Gemini-only Cloud mode.")
        USE_LOCAL_MODEL = False
else:
    print("☁️ VanaVaidhya AI — Running in Gemini-only Cloud mode (no local TensorFlow)...")


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
WEB_CLIENT_SECRET = os.environ.get("WEB_CLIENT_SECRET", "plantsage_web_bypass_key_2026")

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    """Dependency: validate bearer token and return the user payload."""
    if not creds:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = auth_service.decode_token(creds.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

async def verify_api_access(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    x_client_secret: Optional[str] = Header(None, alias="X-Client-Secret"),
) -> dict:
    """
    Allow access either through a valid Bearer JWT token (mobile app)
    or a static Client Secret (web app).
    """
    # 1. Check if client secret is provided and matches
    if x_client_secret and x_client_secret == WEB_CLIENT_SECRET:
        return {"sub": "web_client", "type": "bypass"}

    # 2. Check Bearer token
    if creds:
        token = creds.credentials
        # E2E test mock token bypass (only allowed with default dev secret)
        if token.endswith(".signature") and auth_service.JWT_SECRET == "3f8a2b4c9d1e6f7a0b5c2d8e3f4a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a":
            try:
                import base64, json
                parts = token.split(".")
                if len(parts) == 3:
                    payload_b64 = parts[1]
                    payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
                    payload_json = base64.b64decode(payload_b64).decode()
                    payload = json.loads(payload_json)
                    if payload.get("user_id") == "123":
                        return {"sub": "123", "type": "access"}
            except Exception:
                pass

        payload = auth_service.decode_token(token)
        if payload and payload.get("type") == "access":
            # Verify user still exists in the database
            user_doc = await auth_service.get_user_by_id(payload["sub"])
            if user_doc:
                return payload

    raise HTTPException(status_code=401, detail="Invalid or missing authentication credentials.")

# ─── Sanitization helpers for Prompt Injection Prevention ───────────
def sanitize_input_string(value: str, max_length: int = 100) -> str:
    if not value:
        return ""
    value = value.strip()[:max_length]
    # Keep only safe alphanumeric characters, spaces, hyphens, underscores, dots, and parentheses
    return re.sub(r"[^\w\s\-\.\(\)]", "", value)

def sanitize_symptom_string(value: str, max_length: int = 200) -> str:
    if not value:
        return ""
    value = value.strip()[:max_length]
    # Keep alphanumeric characters, spaces, and basic punctuation (, . ? - /)
    return re.sub(r"[^\w\s\,\.\?\-\/]", "", value)


# ─── Preprocessing ────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize + convert to float32 in [0, 255]. Model handles rescaling internally."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    try:
        preview_path = Path(__file__).parent / "preprocessed_preview.jpg"
        img.save(preview_path, "JPEG")
    except Exception as e:
        print(f"⚠️ Failed to save preprocess preview image: {e}")
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


def get_label_by_class_name(class_name: str) -> dict:
    """Helper to retrieve labels entry by its class_name."""
    for idx, entry in labels.items():
        if entry.get("class_name") == class_name:
            return entry
    return {
        "class_name": class_name,
        "display_name": class_name.title().replace("_", " "),
        "is_toxic": False,
        "needs_caution": False,
        "ayush_recognized": False
    }


async def identify(image: Image.Image, tta: bool = False) -> dict:
    """Core identification — single pass or 4-pass TTA on local model, or Gemini fallback."""
    if USE_LOCAL_MODEL:
        probs = run_inference(preprocess(image))

        if tta:
            for seed in range(1, 4):
                probs = probs + run_inference(augment(image, seed=seed))
            probs = probs / 4.0
            method = "TTA (4 passes)"
        else:
            method = "single pass"

        print(f"🌿 [ML INFERENCE] Local EfficientNetV2S model used ({method}) for classification.")

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
    else:
        print("☁️ [ML INFERENCE] Gemini Cloud Vision used for classification.")
        # Convert PIL image to base64 to send to Gemini Vision
        buffered = io.BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Call Gemini to identify the plant
        gemini_res = await gemini_service.identify_plant_with_gemini(img_str)
        top5_classes = gemini_res.get("top5_class_names", [])
        top_confidence = gemini_res.get("confidence", 90.0)

        # Ensure we have exactly 5 items
        while len(top5_classes) < 5:
            for idx, entry in labels.items():
                c_name = entry.get("class_name")
                if c_name not in top5_classes:
                    top5_classes.append(c_name)
                if len(top5_classes) == 5:
                    break

        # Build the top5 prediction results
        top5 = []
        for rank, class_name in enumerate(top5_classes[:5], start=1):
            entry = get_label_by_class_name(class_name)
            # Extrapolate confidence: top gets top_confidence, others get stepped down
            if rank == 1:
                conf = top_confidence
            else:
                conf = max(5.0, top_confidence - (rank - 1) * 15.0)

            top5.append({
                "rank":             rank,
                "class_name":       entry.get("class_name", class_name),
                "display_name":     entry.get("display_name", class_name),
                "confidence":       round(float(conf), 2),
                "is_toxic":         entry.get("is_toxic", False),
                "needs_caution":    entry.get("needs_caution", False),
                "ayush_recognized": entry.get("ayush_recognized", False),
                "model_source":     "Gemini-Vision-v2",
            })

        top = top5[0]
        return {
            "success":        True,
            "top_prediction": top,
            "top5":           top5,
            "routing":        "Gemini Vision Cloud",
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


@app.post("/debug/save-image")
async def save_debug_image(request: IdentifyRequest):
    """Debug endpoint: save the received base64 image as a 300x300 preview JPEG file."""
    try:
        image = decode_b64(request.base64_image)
        img_resized = image.convert("RGB").resize((300, 300), Image.LANCZOS)
        preview_path = Path(__file__).parent / "preprocessed_preview.jpg"
        img_resized.save(preview_path, "JPEG")
        print(f"📸 Debug image saved to: {preview_path.resolve()}")
        return {"success": True, "path": str(preview_path.resolve())}
    except Exception as e:
        print(f"⚠️ Failed to save debug image: {e}")
        return {"success": False, "error": str(e)}


@app.post("/identify", response_model=IdentifyResponse)
async def identify_base64(
    request: IdentifyRequest,
    tta: bool = Query(default=False, description="Enable Test-Time Augmentation (4 passes)"),
    auth: dict = Depends(verify_api_access)
):
    try:
        image  = decode_b64(request.base64_image)
        result = await identify(image, tta=tta)
        return IdentifyResponse(**result)
    except Exception as e:
        print(f"⚠️ Base64 identification error: {e}")
        raise HTTPException(status_code=400, detail="Identification failed. Please verify the base64 payload is a valid image.")


@app.post("/identify/upload", response_model=IdentifyResponse)
async def identify_upload(
    file: UploadFile = File(...),
    tta:  bool = Query(default=False, description="Enable Test-Time Augmentation (4 passes)"),
    auth: dict = Depends(verify_api_access)
):
    # Enforce image-only MIME type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed.")

    # Read in chunks up to 5MB to avoid memory overflow (DoS)
    max_size = 5 * 1024 * 1024  # 5MB
    contents = b""
    while True:
        chunk = await file.read(8192)
        if not chunk:
            break
        contents += chunk
        if len(contents) > max_size:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 5MB.")

    try:
        image = Image.open(io.BytesIO(contents))
        # Verify it can be parsed as a valid image format
        image.verify()
        # Re-open after verify() because verify() invalidates the image object
        image = Image.open(io.BytesIO(contents))
        result = await identify(image, tta=tta)
        return IdentifyResponse(**result)
    except Exception as e:
        print(f"⚠️ Upload identification error: {e}")
        raise HTTPException(status_code=400, detail="Upload failed. The uploaded file is invalid or corrupt.")


@app.get("/health")
@app.head("/health")
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
    is_toxic: Optional[bool] = False

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
async def gemini_plant_info(request: PlantInfoRequest, auth: dict = Depends(verify_api_access)):
    """Get full plant info (hybrid PFAF + Gemini). Mirrors getPlantInfo() in geminiService.ts."""
    sanitized_plant = sanitize_input_string(request.plant_name)
    sanitized_botanical = sanitize_input_string(request.botanical_name) if request.botanical_name else None
    sanitized_class = sanitize_input_string(request.class_label) if request.class_label else None
    try:
        result = await gemini_service.get_plant_info(
            sanitized_plant,
            sanitized_botanical,
            sanitized_class,
            request.is_toxic,
        )
        return result
    except Exception as e:
        print(f"⚠️ Gemini plant info error: {e}")
        raise HTTPException(status_code=500, detail="Gemini plant info query failed. Please try again.")


@app.post("/ai/symptom-search")
async def gemini_symptom_search(request: SymptomSearchRequest, auth: dict = Depends(verify_api_access)):
    """Search plants by symptom using Gemini. Mirrors searchPlantsBySymptom() in geminiService.ts."""
    sanitized_symptom = sanitize_symptom_string(request.symptom)
    try:
        result = await gemini_service.search_plants_by_symptom(sanitized_symptom)
        return result
    except Exception as e:
        print(f"⚠️ Gemini symptom search error: {e}")
        raise HTTPException(status_code=500, detail="Gemini symptom search query failed. Please try again.")


@app.post("/ai/verify-plant")
async def gemini_verify_plant(request: VerifyPlantRequest, auth: dict = Depends(verify_api_access)):
    """Check if image contains a plant. Mirrors verifyIsPlant() in geminiService.ts."""
    try:
        is_plant = await gemini_service.verify_is_plant(request.base64_image)
        return {"is_plant": is_plant}
    except Exception as e:
        print(f"⚠️ Gemini verify plant error: {e}")
        raise HTTPException(status_code=500, detail="Gemini plant presence check failed.")


@app.post("/ai/verify-vision")
async def gemini_verify_vision(request: VerifyVisionRequest, auth: dict = Depends(verify_api_access)):
    """Cross-verify ML prediction with Gemini vision. Mirrors verifyPlantWithVision() in geminiService.ts."""
    sanitized_plant = sanitize_input_string(request.ml_plant_name)
    sanitized_class = sanitize_input_string(request.ml_class_name)
    
    print(f"🔍 [ML CROSS-VERIFICATION] Consulting AI (Gemini) to verify local model prediction '{sanitized_plant}' (ML Confidence: {request.ml_confidence:.1f}%)")
    
    try:
        result = await gemini_service.verify_plant_with_vision(
            request.base64_image,
            sanitized_plant,
            sanitized_class,
            request.ml_confidence,
        )
        return result
    except Exception as e:
        print(f"⚠️ Gemini vision verify error: {e}")
        raise HTTPException(status_code=500, detail="Gemini cross-verification failed.")


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
    if len(req.password) < 8:
        raise HTTPException(status_code=422, detail="Password must be at least 8 characters.")
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
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 VanaVaidhya AI v2 — Server starting...")
    print(f"   Model  : EfficientNetV2S — {NUM_CLASSES} classes")
    print(f"   Accuracy: 96.6% val  /  98.4% TTA")
    print(f"   Docs   : http://localhost:{port}/docs")
    print(f"   TTA    : POST /identify?tta=true")
    print(f"   Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

