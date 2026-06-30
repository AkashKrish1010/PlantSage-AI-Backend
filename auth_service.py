"""
auth_service.py — VanaVaidhya Authentication Service
=====================================================
Handles:
  • MongoDB user CRUD (async via motor)
  • Password hashing (bcrypt via passlib)
  • JWT access + refresh token creation/decoding
  • Google ID token verification (server-side via google-auth)

Collections:
  users  — registered accounts
    _id, name, email, password_hash, google_id, photo_url, created_at, updated_at
  refresh_tokens — persisted refresh tokens (optional revocation support)
    _id, user_id, token_hash, created_at, expires_at, revoked
"""

import os
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

from pathlib import Path
from dotenv import load_dotenv

_ML_DIR = Path(__file__).parent
load_dotenv(_ML_DIR / ".env")

# ── Passlib (bcrypt) ──────────────────────────────────────────────────────────
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────
from jose import jwt, JWTError

JWT_SECRET    = os.environ.get("JWT_SECRET", "changeme-in-production")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_EXPIRE  = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_EXPIRE = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS",   "7"))

def create_access_token(user_id: str, email: str, name: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_EXPIRE)
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "type":  "access",
        "exp":   expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE)
    payload = {
        "sub":  user_id,
        "type": "refresh",
        "exp":  expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    """Returns the decoded payload dict, or None if invalid/expired."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


# ── Motor (async MongoDB) ─────────────────────────────────────────────────────
import motor.motor_asyncio

MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
MONGO_DB  = os.environ.get("MONGODB_DB",  "vanavaidhya")

_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None

def get_db():
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    return _client[MONGO_DB]

async def ensure_indexes():
    """Call once at startup to create unique indexes on email and google_id."""
    db = get_db()
    await db.users.create_index("email",     unique=True, sparse=True)
    await db.users.create_index("google_id", unique=True, sparse=True)


# ── User CRUD ─────────────────────────────────────────────────────────────────

def _serialize_user(doc: dict) -> dict:
    """Convert MongoDB doc to safe JSON-serializable dict (no password_hash)."""
    return {
        "id":        str(doc["_id"]),
        "name":      doc.get("name", ""),
        "email":     doc.get("email", ""),
        "photo_url": doc.get("photo_url"),
        "login_method": doc.get("login_method", "email"),
        "created_at": doc.get("created_at", "").isoformat() if isinstance(doc.get("created_at"), datetime) else doc.get("created_at", ""),
    }


async def get_user_by_email(email: str) -> Optional[dict]:
    db = get_db()
    doc = await db.users.find_one({"email": email.lower().strip()})
    return doc  # raw doc (includes password_hash for verification)


async def get_user_by_id(user_id: str) -> Optional[dict]:
    from bson import ObjectId
    db = get_db()
    try:
        doc = await db.users.find_one({"_id": ObjectId(user_id)})
        return doc
    except Exception:
        return None


async def get_user_by_google_id(google_id: str) -> Optional[dict]:
    db = get_db()
    return await db.users.find_one({"google_id": google_id})


async def create_email_user(name: str, email: str, password: str) -> dict:
    """Create a new user with email/password. Raises ValueError on duplicate email."""
    db = get_db()
    now = datetime.now(timezone.utc)
    doc = {
        "name":          name.strip(),
        "email":         email.lower().strip(),
        "password_hash": hash_password(password),
        "login_method":  "email",
        "created_at":    now,
        "updated_at":    now,
    }
    try:
        result = await db.users.insert_one(doc)
        doc["_id"] = result.inserted_id
        return _serialize_user(doc)
    except Exception as e:
        if "duplicate key" in str(e).lower() or "E11000" in str(e):
            raise ValueError("An account with this email already exists.")
        raise


async def upsert_google_user(google_id: str, email: str, name: str, photo_url: Optional[str]) -> dict:
    """
    Find-or-create a user who signed in with Google.
    If the email already exists (email-signup), links the Google ID.
    """
    db = get_db()
    now = datetime.now(timezone.utc)

    # Try by google_id first
    doc = await get_user_by_google_id(google_id)
    if doc:
        return _serialize_user(doc)

    # Try by email (link accounts)
    doc = await get_user_by_email(email)
    if doc:
        await db.users.update_one(
            {"_id": doc["_id"]},
            {"$set": {"google_id": google_id, "photo_url": photo_url, "updated_at": now}},
        )
        doc["google_id"]  = google_id
        doc["photo_url"]  = photo_url
        return _serialize_user(doc)

    # Brand new Google user
    new_doc = {
        "name":          name,
        "email":         email.lower().strip(),
        "password_hash": None,
        "google_id":     google_id,
        "photo_url":     photo_url,
        "login_method":  "google",
        "created_at":    now,
        "updated_at":    now,
    }
    result = await db.users.insert_one(new_doc)
    new_doc["_id"] = result.inserted_id
    return _serialize_user(new_doc)


# ── Refresh Token Store ───────────────────────────────────────────────────────

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

async def store_refresh_token(user_id: str, token: str):
    db = get_db()
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_EXPIRE)
    await db.refresh_tokens.insert_one({
        "user_id":    user_id,
        "token_hash": _hash_token(token),
        "created_at": datetime.now(timezone.utc),
        "expires_at": expires_at,
        "revoked":    False,
    })

async def validate_refresh_token(token: str) -> bool:
    """Returns True if the token exists, is not revoked, and is not expired."""
    db = get_db()
    doc = await db.refresh_tokens.find_one({
        "token_hash": _hash_token(token),
        "revoked":    False,
        "expires_at": {"$gt": datetime.now(timezone.utc)},
    })
    return doc is not None

async def revoke_refresh_token(token: str):
    db = get_db()
    await db.refresh_tokens.update_one(
        {"token_hash": _hash_token(token)},
        {"$set": {"revoked": True}},
    )

async def revoke_all_user_tokens(user_id: str):
    """Used on logout to invalidate all sessions for a user."""
    db = get_db()
    await db.refresh_tokens.update_many(
        {"user_id": user_id},
        {"$set": {"revoked": True}},
    )


# ── Google Token Verification ─────────────────────────────────────────────────
import httpx

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_TOKEN_INFO_URL = "https://oauth2.googleapis.com/tokeninfo"

async def verify_google_id_token(id_token: str) -> Optional[dict]:
    """
    Verify a Google ID token obtained by the mobile app via expo-auth-session.
    Returns a dict with {google_id, email, name, photo_url} on success, or None.

    NOTE: For production, use google-auth library verification. This HTTP approach
    works without service-account setup and is suitable for development.
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                GOOGLE_TOKEN_INFO_URL,
                params={"id_token": id_token},
                timeout=10.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()

            # Verify audience matches our client ID (enforced for security)
            if not GOOGLE_CLIENT_ID or GOOGLE_CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID_HERE":
                print("⚠️ Google ID token verification rejected: GOOGLE_CLIENT_ID is not configured on the server.")
                return None

            aud = data.get("aud", "")
            if GOOGLE_CLIENT_ID not in aud:
                print(f"⚠️ Google ID token verification failed: audience mismatch (aud: {aud}, expected: {GOOGLE_CLIENT_ID})")
                return None

            return {
                "google_id": data.get("sub"),
                "email":     data.get("email", ""),
                "name":      data.get("name", data.get("email", "").split("@")[0]),
                "photo_url": data.get("picture"),
            }
        except Exception as e:
            print(f"⚠️ Google token verification exception: {e}")
            return None
