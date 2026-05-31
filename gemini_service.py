"""
gemini_service.py  —  VanaVaidhya Gemini AI Service
=====================================================
Python port of src/services/geminiService.ts

Provides:
  • get_plant_info()            → hybrid PFAF + Gemini plant details
  • search_plants_by_symptom()  → Gemini symptom → plant lookup
  • verify_is_plant()           → lightweight vision gate (Gemini vision)
  • verify_plant_with_vision()  → full ML cross-verification (Gemini vision)

The Gemini API key is loaded from ml/.env via python-dotenv.
All prompts are kept identical to the TypeScript originals.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# ─── Load environment ──────────────────────────────────────────────
_ML_DIR = Path(__file__).parent
load_dotenv(_ML_DIR / ".env")

GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-3.1-flash-lite")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Add it to ml/.env or set it as an environment variable."
    )

# ─── PFAF plant database ──────────────────────────────────────────
# Same plant_db.json the frontend uses — load once at module import.
_ROOT_DIR    = _ML_DIR.parent
_PLANT_DB_PATH = _ROOT_DIR / "src" / "data" / "plant_db.json"

PLANT_DB: dict[str, Any] = {}
if _PLANT_DB_PATH.exists():
    with open(_PLANT_DB_PATH, encoding="utf-8") as _f:
        PLANT_DB = json.load(_f)
else:
    print(f"⚠️  plant_db.json not found at {_PLANT_DB_PATH} — PFAF lookup disabled")


# ─── PFAF helpers (mirror of plantDatabase.ts) ───────────────────

def _get_plant_by_class(class_name: str) -> Optional[dict]:
    """Look up a plant entry by its ML class name (e.g. 'tulsi', 'neem')."""
    key = class_name.lower().replace(" ", "_").replace("-", "_")
    entry = PLANT_DB.get(key)
    if entry and entry.get("is_in_pfaf"):
        return entry
    return None


def _get_plant_by_name(name: str) -> Optional[dict]:
    """Fuzzy lookup by common or scientific name."""
    needle = name.lower()
    for entry in PLANT_DB.values():
        if not entry.get("is_in_pfaf"):
            continue
        if (
            needle in entry.get("common_name", "").lower()
            or needle in entry.get("scientific_name", "").lower()
        ):
            return entry
    return None


# Category map mirrors buildUsesFromPFAF() from plantDatabase.ts exactly.
_PFAF_CATEGORIES = [
    {
        "condition": "Fever & Infections",
        "keywords": ["antibacterial", "antiviral", "antimicrobial", "antiseptic", "febrifuge", "antipyretic"],
        "description": "Helps fight bacterial and viral infections; traditionally used to reduce fever.",
    },
    {
        "condition": "Digestive Health",
        "keywords": ["carminative", "stomachic", "digestive", "antidiarrhoeal", "laxative", "appetizer"],
        "description": "Aids digestion, relieves bloating, gas, and stomach cramps.",
    },
    {
        "condition": "Respiratory Health",
        "keywords": ["expectorant", "antiasthmatic", "bronchitis", "pulmonary", "pectoral"],
        "description": "Helpful in cough, asthma, bronchitis, and respiratory congestion.",
    },
    {
        "condition": "Anti-inflammatory & Pain",
        "keywords": ["antiinflammatory", "anti-inflammatory", "analgesic", "anodyne", "antirheumatic", "antiarthritic"],
        "description": "Reduces inflammation and pain, useful in arthritis and joint conditions.",
    },
    {
        "condition": "Skin Health",
        "keywords": ["skin", "vulnerary", "astringent", "emollient", "eczema", "dermatitis", "antipruritic"],
        "description": "Used topically for wounds, skin infections, rashes, and inflammatory skin conditions.",
    },
    {
        "condition": "Immunity & Vitality",
        "keywords": ["adaptogen", "tonic", "immunomodulatory", "alterative", "nutritive", "stimulant"],
        "description": "Strengthens the immune system and improves overall vitality and resilience.",
    },
    {
        "condition": "Diuretic & Kidney Support",
        "keywords": ["diuretic", "lithontripic", "urinary", "nephritis"],
        "description": "Promotes healthy kidney function and urine flow; may help dissolve kidney stones.",
    },
    {
        "condition": "Liver & Blood",
        "keywords": ["hepatic", "depurative", "cholagogue", "haemostatic", "anticholesterolemic", "hypoglycaemic"],
        "description": "Supports liver function, purifies blood, and may help regulate blood sugar and cholesterol.",
    },
    {
        "condition": "Nervous System",
        "keywords": ["nervine", "sedative", "antispasmodic", "epilepsy", "anticonvulsant"],
        "description": "Calms the nervous system; traditionally used for anxiety, spasms, and epilepsy.",
    },
    {
        "condition": "Women's Health",
        "keywords": ["emmenagogue", "galactogogue", "uterine", "women's complaints", "contraceptive"],
        "description": "Used in traditional medicine to support menstruation and women's reproductive health.",
    },
    {
        "condition": "Wound Healing",
        "keywords": ["vulnerary", "wound", "cicatrizant", "haemostatic"],
        "description": "Promotes wound healing and helps stop bleeding.",
    },
    {
        "condition": "Cancer & Antioxidant",
        "keywords": ["cancer", "cytostatic", "antioxidant", "free radical"],
        "description": "Contains compounds studied for antioxidant and potential anti-tumour properties.",
    },
]


def _build_uses_from_pfaf(props: list[str]) -> list[dict]:
    """Convert PFAF medicinal_properties list into use-case descriptions."""
    lower_props = [p.lower() for p in props]
    results = []
    for cat in _PFAF_CATEGORIES:
        if any(kw in lp for kw in cat["keywords"] for lp in lower_props):
            results.append({"condition": cat["condition"], "description": cat["description"]})
        if len(results) >= 7:
            break
    return results


# ─── Core Gemini & Groq API callers ────────────────────────────────

async def _call_groq_fallback(gemini_body: dict) -> str:
    """
    Translates a Gemini API request body into OpenAI format,
    sends it to Groq API, and returns the response text.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured. Cannot fallback to Groq.")

    # 1. Extract prompt and base64 image (if present) from Gemini payload
    parts = gemini_body.get("contents", [{}])[0].get("parts", [])
    prompt = ""
    base64_image = ""
    for part in parts:
        if "text" in part:
            prompt = part["text"]
        elif "inline_data" in part:
            base64_image = part["inline_data"].get("data", "")

    # 2. Build Groq request messages
    messages = []
    if base64_image:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
    else:
        messages.append({
            "role": "user",
            "content": prompt
        })

    # 3. Build Groq payload
    groq_payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": gemini_body.get("generationConfig", {}).get("temperature", 0.7),
    }

    # If the Gemini request asked for JSON, use Groq JSON Mode
    is_json = gemini_body.get("generationConfig", {}).get("response_mime_type") == "application/json"
    if is_json:
        groq_payload["response_format"] = {"type": "json_object"}

    # 4. Post to Groq API
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=groq_payload, headers=headers)

    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Empty response from Groq")

    return content


async def _call_gemini_http(body: dict, model: str) -> str:
    """
    Makes a raw HTTP POST request to the Gemini API endpoint.
    No automatic logging or fallbacks.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
    data = resp.json()
    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text")
    )
    if not text:
        raise RuntimeError("Empty response from Gemini")
    return text


async def _call_gemini(body: dict, model: str = GEMINI_MODEL) -> str:
    """
    POST to Gemini generateContent endpoint (Tier 1).
    If the call fails and a Groq API Key is present, falls back to Groq (Tier 2).
    """
    print(f"[PlantSage AI] Calling primary model: {model}")
    try:
        return await _call_gemini_http(body, model)
    except Exception as e:
        if GROQ_API_KEY:
            print(f"[PlantSage AI] Gemini failed: {e}. Falling back to Groq model: {GROQ_MODEL}")
            return await _call_groq_fallback(body)
        else:
            print(f"[PlantSage AI] Gemini failed: {e}. No GROQ_API_KEY available for fallback.")
            raise e


def _parse_json_safe(text: str) -> Any:
    """
    Strip markdown fences and attempt to recover truncated JSON.
    Mirrors parseJSON() from geminiService.ts exactly.
    """
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to auto-close unclosed braces/brackets (Gemini truncation recovery)
        attempt = cleaned
        open_stack: list[str] = []
        in_str = False
        escape = False
        for ch in attempt:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in "{[":
                open_stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if open_stack:
                    open_stack.pop()
        attempt = re.sub(r",\s*$", "", attempt)
        attempt += "".join(reversed(open_stack))
        return json.loads(attempt)


# ─── Part A — Remedies-only from Gemini (PFAF hit path) ──────────

async def _get_remedies_from_gemini(
    plant_name: str,
    botanical_name: str,
    known_hazards: str,
    medicinal_properties: list[str],
) -> dict:
    """
    Mirrors getRemediesFromGemini() from geminiService.ts.
    Called when PFAF data already supplies static fields.
    """
    props_hint = (
        f"Known properties from botanical database: {', '.join(medicinal_properties[:8])}."
        if medicinal_properties else ""
    )
    hazard_hint = (
        f"Known hazards from botanical database: {known_hazards[:200]}"
        if known_hazards else ""
    )
    from urllib.parse import quote
    encoded_name      = quote(plant_name)
    encoded_botanical = quote(botanical_name)
    encoded_bot_wiki  = encoded_botanical.replace("%20", "_")
    plant_lower       = plant_name.lower().replace(" ", "-")

    prompt = (
        f"Expert Ayurvedic physician. Plant: {plant_name} ({botanical_name}). "
        f"{props_hint} {hazard_hint}\n\n"
        "Return ONLY valid JSON (no markdown, no extra text). Use REAL URLs only — "
        "YouTube search URLs and Wikipedia page URLs.\n"
        '{"family":"...","commonNames":["Hindi name","Sanskrit name","Tamil name"],'
        '"ayushRecognized":true,"doctorVerified":true,'
        '"partsUsed":[{"part":"Leaf","preparation":"Tea/juice"},{"part":"Root","preparation":"Powder"}],'
        '"homeRemedies":['
        '{"name":"Remedy 1","forCondition":"condition","ingredients":["ingredient1","ingredient2"],'
        '"steps":["Step 1.","Step 2.","Step 3."],"difficulty":"Easy","prepTime":"10 min"},'
        '{"name":"Remedy 2","forCondition":"condition","ingredients":["ingredient1"],'
        '"steps":["Step 1.","Step 2."],"difficulty":"Easy","prepTime":"5 min"},'
        '{"name":"Remedy 3","forCondition":"condition","ingredients":["ingredient1"],'
        '"steps":["Step 1.","Step 2."],"difficulty":"Easy","prepTime":"5 min"}],'
        '"dosage":"Adults: dosage with timing","cautions":["caution1"],"toxicLookalike":null,'
        '"learnMoreLinks":['
        f'{{"title":"How to make {plant_name} remedy","url":"https://www.youtube.com/results?search_query={encoded_name}+home+remedy+ayurveda","type":"youtube"}},'
        f'{{"title":"{plant_name} medicinal uses","url":"https://www.youtube.com/results?search_query={encoded_name}+medicinal+uses","type":"youtube"}},'
        f'{{"title":"{plant_name} - Wikipedia","url":"https://en.wikipedia.org/wiki/{encoded_bot_wiki}","type":"wikipedia"}},'
        f'{{"title":"{plant_name} on WebMD","url":"https://www.webmd.com/vitamins/ai/ingredientmono-{plant_lower}","type":"article"}}],'
        f'"relatedSearches":{{"google":["{plant_name} Ayurvedic uses","{plant_name} side effects"],'
        f'"youtube":["{plant_name} home remedy"],"news":["{plant_name} research 2024"]}}}}'
    )

    raw = await _call_gemini({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 3500,
        },
    })
    return _parse_json_safe(raw)


# ─── Part B — Full Gemini fallback (plant NOT in PFAF) ───────────

async def _get_full_plant_from_gemini(
    plant_name: str,
    botanical_name: Optional[str] = None,
) -> dict:
    """
    Mirrors getFullPlantFromGemini() from geminiService.ts.
    Called when the plant is not found in the PFAF database.
    """
    from urllib.parse import quote
    plant_label   = f"{plant_name} ({botanical_name})" if botanical_name else plant_name
    encoded_name  = quote(plant_name)
    encoded_bot   = quote(botanical_name).replace("%20", "_") if botanical_name else encoded_name
    plant_lower   = plant_name.lower().replace(" ", "-")

    prompt = (
        f"Expert AYUSH Ayurvedic botanist. Provide info about: {plant_label}\n\n"
        "Return ONLY valid JSON (no markdown). Use REAL YouTube search URLs and real Wikipedia page URLs in learnMoreLinks.\n"
        '{"name":"Common name","botanicalName":"Genus species","family":"Family",'
        '"commonNames":["Hindi","Sanskrit","Tamil","Other"],'
        '"description":"2-3 sentences about medicinal significance.",'
        '"ayushRecognized":true,"doctorVerified":true,'
        '"partsUsed":[{"part":"Leaf","preparation":"Tea/juice/paste"},{"part":"Root","preparation":"Powder/decoction"},{"part":"Seed","preparation":"Oil/paste"}],'
        '"medicinalProperties":["prop1","prop2","prop3","prop4","prop5"],'
        '"uses":[{"condition":"Condition 1","description":"How it helps."},{"condition":"Condition 2","description":"How it helps."},{"condition":"Condition 3","description":"How it helps."},{"condition":"Condition 4","description":"How it helps."},{"condition":"Condition 5","description":"How it helps."}],'
        '"homeRemedies":['
        '{"name":"Remedy 1","forCondition":"condition","ingredients":["ingredient1","1 cup water"],"steps":["Step 1.","Step 2.","Step 3."],"difficulty":"Easy","prepTime":"10 min"},'
        '{"name":"Remedy 2","forCondition":"condition","ingredients":["ingredient1"],"steps":["Step 1.","Step 2."],"difficulty":"Easy","prepTime":"5 min"},'
        '{"name":"Remedy 3","forCondition":"condition","ingredients":["ingredient1"],"steps":["Step 1.","Step 2."],"difficulty":"Medium","prepTime":"15 min"}],'
        '"dosage":"Adults: specific dose with timing.",'
        '"cautions":["caution1","caution2"],"toxicLookalike":null,'
        '"learnMoreLinks":['
        f'{{"title":"How to prepare {plant_name} home remedy","url":"https://www.youtube.com/results?search_query={encoded_name}+home+remedy+ayurveda","type":"youtube"}},'
        f'{{"title":"{plant_name} health benefits","url":"https://www.youtube.com/results?search_query={encoded_name}+health+benefits+medicinal","type":"youtube"}},'
        f'{{"title":"{plant_name} Ayurvedic uses","url":"https://www.youtube.com/results?search_query={encoded_name}+ayurvedic+uses+hindi","type":"youtube"}},'
        f'{{"title":"{plant_name} - Wikipedia","url":"https://en.wikipedia.org/wiki/{encoded_bot}","type":"wikipedia"}},'
        f'{{"title":"{plant_name} on WebMD","url":"https://www.webmd.com/vitamins/ai/ingredientmono-{plant_lower}","type":"article"}}],'
        f'"relatedSearches":{{"google":["{plant_name} Ayurvedic uses","{plant_name} side effects","{plant_name} AYUSH research"],'
        f'"youtube":["{plant_name} home remedy","{plant_name} benefits Hindi"],'
        f'"news":["{plant_name} research 2024"]}}}}'
    )

    raw = await _call_gemini({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 4096,
        },
    })
    result = _parse_json_safe(raw)
    result.setdefault("learnMoreLinks", [])
    result["dataSource"] = "gemini"
    return result


# ─── 1. Main export: Get full plant info ─────────────────────────

async def get_plant_info(
    plant_name: str,
    botanical_name: Optional[str] = None,
    class_label: Optional[str] = None,
) -> dict:
    """
    Mirrors getPlantInfo() from geminiService.ts.
    Hybrid: PFAF static data + Gemini-generated remedies, or full Gemini fallback.
    """
    # Step 1: Try PFAF database
    pfaf_entry = (
        (_get_plant_by_class(class_label) if class_label else None)
        or (_get_plant_by_name(botanical_name) if botanical_name else None)
        or _get_plant_by_name(plant_name)
    )

    # Step 2: PFAF hit → hybrid response
    if pfaf_entry:
        sci_name = botanical_name or pfaf_entry.get("scientific_name", "")

        # Build cautions from known hazards
        cautions: list[str] = []
        known_hazards = pfaf_entry.get("known_hazards", "")
        if known_hazards:
            parts = [s.strip() for s in re.split(r"[.;]", known_hazards)]
            cautions = [s for s in parts if len(s) > 10][:3]

        # Build description
        description = (
            pfaf_entry.get("summary")
            or pfaf_entry.get("edible_uses")
            or f"{plant_name} is a medicinal plant with the following properties: "
               f"{', '.join(pfaf_entry.get('medicinal_properties', [])[:5])}."
        )

        # Build uses from medicinal properties
        uses = _build_uses_from_pfaf(pfaf_entry.get("medicinal_properties", []))

        # Gemini: only remedies, dosage, partsUsed, searches
        gemini_data = await _get_remedies_from_gemini(
            plant_name,
            sci_name,
            known_hazards,
            pfaf_entry.get("medicinal_properties", []),
        )

        return {
            "name":               plant_name,
            "botanicalName":      sci_name or gemini_data.get("family", ""),
            "family":             gemini_data.get("family", ""),
            "commonNames":        gemini_data.get("commonNames", []),
            "description":        description[:600],
            "ayushRecognized":    gemini_data.get("ayushRecognized", False),
            "doctorVerified":     gemini_data.get("doctorVerified", False),
            "partsUsed":          gemini_data.get("partsUsed", []),
            "medicinalProperties": pfaf_entry.get("medicinal_properties", []),
            "uses": (
                uses if uses
                else [{"condition": p["part"], "description": p["preparation"]}
                      for p in gemini_data.get("partsUsed", [])]
            ),
            "cautions":       cautions if cautions else ["Consult a physician before use."],
            "homeRemedies":   gemini_data.get("homeRemedies", []),
            "dosage":         gemini_data.get("dosage", ""),
            "toxicLookalike": gemini_data.get("toxicLookalike"),
            "learnMoreLinks": gemini_data.get("learnMoreLinks", []),
            "relatedSearches": gemini_data.get("relatedSearches", {}),
            "dataSource":     "pfaf+gemini",
            "pfafUrl":        pfaf_entry.get("pfaf_url", ""),
        }

    # Step 3: Not in PFAF → full Gemini fallback
    print(f"[PlantSage AI] {plant_name} not in PFAF — using full Gemini")
    return await _get_full_plant_from_gemini(plant_name, botanical_name)


# ─── 2. Search plants by symptom ─────────────────────────────────

async def search_plants_by_symptom(symptom: str) -> list[dict]:
    """
    Mirrors searchPlantsBySymptom() from geminiService.ts.
    Returns a list of SymptomPlant objects.
    """
    prompt = (
        f'You are an expert Ayurvedic physician. A patient describes their problem as: "{symptom}"\n\n'
        "Return ONLY a valid JSON array of the most relevant medicinal plants (no markdown):\n"
        "[\n"
        "  {\n"
        '    "plantName": "Common name",\n'
        '    "botanicalName": "Genus species",\n'
        '    "ayushRecognized": true,\n'
        '    "relevanceScore": 95,\n'
        '    "howItHelps": "2-3 sentence specific explanation of mechanism and clinical evidence",\n'
        '    "primaryPreparation": "Tea / Decoction / Paste / Juice / Powder",\n'
        '    "quickRemedy": "One-line actionable remedy the patient can try immediately",\n'
        '    "caution": null\n'
        "  }\n"
        "]\n\n"
        "Rules:\n"
        f'- Return 5-7 plants sorted by relevance (most effective first)\n'
        f'- Only include plants with demonstrated efficacy for "{symptom}"\n'
        "- Set caution to a string if there is an important warning for that plant\n"
        "- Focus on plants available in Indian homes/markets\n"
        "- Include both classical Ayurvedic and scientifically validated plants"
    )

    raw = await _call_gemini({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.3,
            "maxOutputTokens": 2000,
        },
    })
    return _parse_json_safe(raw)


# ─── 3. Verify is plant (vision gate) ────────────────────────────

async def verify_is_plant(base64_image: str) -> bool:
    """
    Lightweight check to verify if the primary subject is a plant.
    Primary: Groq (meta-llama/llama-4-scout-17b-16e-instruct)
    Fallback: Gemini (gemini-3.1-flash-lite)
    Fails open (returns True) on any error.
    """
    prompt = (
        "Look at this image carefully.\n\n"
        "Does this image contain a plant, leaf, herb, flower, tree, or any vegetation "
        "as the primary subject?\n\n"
        "Return ONLY a valid JSON object (no markdown, no extra text):\n"
        '{"is_plant": true}\n'
        "or\n"
        '{"is_plant": false}\n\n'
        "Rules:\n"
        "- Return true ONLY if the primary subject is a plant or vegetation\n"
        "- Return false for: people, animals, food dishes, buildings, objects, "
        "landscapes without plants as the main subject, screenshots, abstract images, etc.\n"
        "- If a plant appears as a small background element but is NOT the main subject, return false"
    )

    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0,
            "maxOutputTokens": 50,
        },
    }

    raw = ""
    if GROQ_API_KEY:
        try:
            print(f"[PlantSage AI] Calling primary model: {GROQ_MODEL}")
            raw = await _call_groq_fallback(body)
        except Exception as e:
            print(f"[PlantSage AI] Groq failed: {e}. Falling back to Gemini model: {GEMINI_VISION_MODEL}")
            try:
                raw = await _call_gemini_http(body, model=GEMINI_VISION_MODEL)
            except Exception as e2:
                print(f"[PlantSage AI] Gemini fallback also failed: {e2}")
                return True # fail open
    else:
        try:
            print(f"[PlantSage AI] No GROQ_API_KEY configured. Calling Gemini fallback model: {GEMINI_VISION_MODEL}")
            raw = await _call_gemini_http(body, model=GEMINI_VISION_MODEL)
        except Exception as e2:
            print(f"[PlantSage AI] Gemini fallback call failed: {e2}")
            return True # fail open

    try:
        result = _parse_json_safe(raw)
        return result.get("is_plant") is True
    except Exception:
        return True  # fail open


async def verify_plant_with_vision(
    base64_image: str,
    ml_plant_name: str,
    ml_class_name: str,
    ml_confidence: float,
) -> dict:
    """
    Mirrors verifyPlantWithVision() from geminiService.ts.
    Returns a VisionVerifyResult dict. Gracefully falls back on errors.
    """
    prompt = (
        "You are an expert Indian botanist specialising in medicinal plants.\n\n"
        f'A machine learning model has identified this plant as: "{ml_plant_name}" '
        f"(class: {ml_class_name}) with {ml_confidence:.1f}% confidence.\n\n"
        "Look carefully at the image and:\n"
        "1. FIRST determine if this image actually shows a plant, leaf, herb, flower or tree.\n"
        "2. If it IS a plant — determine what plant and decide whether you agree with the ML prediction.\n"
        "3. If it is NOT a plant (person, animal, food, object, building, etc.) — flag it accordingly.\n\n"
        "Return ONLY a valid JSON object (no markdown, no extra text):\n"
        "{\n"
        '  "agrees": false,\n'
        '  "geminiPlant": "NOT_A_PLANT",\n'
        '  "geminiConfidence": "high",\n'
        '  "reasoning": "This image shows [what you see] and does not contain a plant as the primary subject.",\n'
        '  "caution": null,\n'
        '  "plant_care": "Moderate",\n'
        '  "about_plant": "A 2-3 sentence description of the subject."\n'
        "}\n\n"
        "If it IS a plant:\n"
        "{\n"
        '  "agrees": true,\n'
        f'  "geminiPlant": "{ml_plant_name}",\n'
        '  "geminiConfidence": "high",\n'
        '  "reasoning": "One clear sentence — mention the key visual feature (leaf shape, stem color, serration, etc.)",\n'
        '  "caution": null,\n'
        '  "plant_care": "Easy",\n'
        '  "about_plant": "A 2-3 sentence description of the plant, its origin, and its traditional Ayurvedic/medicinal value."\n'
        "}\n\n"
        "Rules:\n"
        '- If the image is NOT a plant, set geminiPlant to exactly "NOT_A_PLANT" and agrees to false\n'
        '- "agrees" is true only when the image IS a plant AND you agree with the ML prediction\n'
        '- "geminiPlant" is your best identification — use the common English name, or "NOT_A_PLANT"\n'
        '- "geminiConfidence": "high" if certain, "medium" if likely, "low" if unclear\n'
        '- "reasoning" must be exactly ONE sentence, specific to what you see\n'
        '- "caution" is null unless the plant is toxic/dangerous\n'
        '- "plant_care" must be exactly one of: "Easy", "Moderate", or "Difficult" based on how easy it is to grow/maintain this plant\n'
        '- "about_plant" must be exactly a 2-3 sentence descriptive summary of the plant, its medicinal significance, and its traditional uses'
    )

    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 300,
        },
    }

    try:
        raw = await _call_gemini(body, model=GEMINI_VISION_MODEL)
        return _parse_json_safe(raw)
    except Exception as exc:
        print(f"[VisionVerify] Failed: {exc}")
        # Graceful fallback — matches the TS behaviour exactly
        return {
            "agrees": True,
            "geminiPlant": ml_plant_name,
            "geminiConfidence": "low",
            "reasoning": "Visual verification unavailable — ML prediction shown.",
            "caution": None,
            "plant_care": "Moderate",
            "about_plant": f"{ml_plant_name} is a medicinal plant traditionally utilized for its herbal health remedies.",
        }
