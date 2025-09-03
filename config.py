#!/usr/bin/env python3
"""
Configuration for AI Slide Generator
- Two tones (concise, comprehensive) with strict, schema-only prompts
- JSON-mode friendly: pair with ChatOllama(..., format="json", temperature=0)
- Enforces exactly 4 bullets per slide
- Supports Ollama, VLLM, and Mixed mode backends
"""

import os
from pydantic import BaseModel

# =============================================================================
# BACKEND SELECTION (COMMENT/UNCOMMENT AS NEEDED)
# =============================================================================

# 🔽 CURRENT DEVELOPMENT SETUP - COMMENT/UNCOMMENT ONE LINE BELOW 🔽

# For Ollama Only (CPU/Local Development):
BACKEND_TYPE = os.getenv("BACKEND_TYPE", "ollama").lower()   # ✅ ACTIVE: Ollama Only

# For VLLM Only (Single Instance - GPU/Production):
# BACKEND_TYPE = os.getenv("BACKEND_TYPE", "vllm").lower()     # ❌ INACTIVE: VLLM Only

# For Mixed Mode (VLLM Generation + Ollama Embeddings):
# BACKEND_TYPE = os.getenv("BACKEND_TYPE", "mixed").lower()    # ❌ INACTIVE: Mixed Mode

# 🔼 SWITCH BY COMMENTING/UNCOMMENTING THE LINES ABOVE 🔼

# -----------------------------------------------------------------------------
# Ollama / Models / Embeddings
# -----------------------------------------------------------------------------

# Local default; override via env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# If your app runs in Docker and Ollama on host, you can switch to:
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

DEFAULT_MODELS = {
    "primary": "llama3.1:8b",
    "regeneration": "llama3.1:8b",
    "creative": "llama3.1:8b",
}

EMBEDDING_MODEL = "nomic-embed-text"

# =============================================================================
# VLLM CONFIGURATION (Single Instance - Matches Working Document Query Code)
# =============================================================================

# Single VLLM server configuration (like your working doc query code)
VLLM_SERVER_IP = os.getenv("VLLM_SERVER_IP", "192.168.210.167")
VLLM_SERVER_PORT = os.getenv("VLLM_SERVER_PORT", "6002")
VLLM_BASE_URL = f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}"

# Use SAME URL for both generation and embeddings (single instance)
VLLM_GENERATION_URL = os.getenv("VLLM_GENERATION_URL", VLLM_BASE_URL)
VLLM_EMBEDDING_URL = os.getenv("VLLM_EMBEDDING_URL", VLLM_BASE_URL)

# Model names (exactly from your working code)
VLLM_GENERATION_MODEL = os.getenv("VLLM_GENERATION_MODEL", "meta-llama/Llama-3.1-8B")
VLLM_EMBEDDING_MODEL = os.getenv("VLLM_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# =============================================================================
# FEATURE FLAGS (Enhanced for Mixed Mode)
# =============================================================================

# Auto-detect if using VLLM
USE_VLLM = BACKEND_TYPE == "vllm"

# Feature flags (automatically configured based on backend selection)
if BACKEND_TYPE == "vllm":
    # Full VLLM mode
    USE_VLLM_GENERATION = os.getenv("USE_VLLM_GENERATION", "true").lower() == "true"
    USE_VLLM_EMBEDDINGS = os.getenv("USE_VLLM_EMBEDDINGS", "true").lower() == "true"
elif BACKEND_TYPE == "mixed":
    # Mixed mode: VLLM for generation, Ollama for embeddings
    USE_VLLM_GENERATION = os.getenv("USE_VLLM_GENERATION", "true").lower() == "true"
    USE_VLLM_EMBEDDINGS = os.getenv("USE_VLLM_EMBEDDINGS", "false").lower() == "true"
else:
    # Ollama only mode
    USE_VLLM_GENERATION = os.getenv("USE_VLLM_GENERATION", "false").lower() == "true"
    USE_VLLM_EMBEDDINGS = os.getenv("USE_VLLM_EMBEDDINGS", "false").lower() == "true"

FALLBACK_TO_OLLAMA = os.getenv("FALLBACK_TO_OLLAMA", "true").lower() == "true"

# Development flags
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))

# =============================================================================
# BACKEND CONFIGURATION OBJECTS
# =============================================================================

VLLM_CONFIG = {
    "generation_url": VLLM_GENERATION_URL,
    "embedding_url": VLLM_EMBEDDING_URL,
    "generation_model": VLLM_GENERATION_MODEL,
    "embedding_model": VLLM_EMBEDDING_MODEL,
    "api_key": VLLM_API_KEY,
}

OLLAMA_CONFIG = {
    "base_url": OLLAMA_BASE_URL,
    "embedding_model": EMBEDDING_MODEL,
    "generation_models": DEFAULT_MODELS,
}

FEATURES = {
    "use_vllm_generation": USE_VLLM_GENERATION,
    "use_vllm_embeddings": USE_VLLM_EMBEDDINGS,
    "fallback_to_ollama": FALLBACK_TO_OLLAMA,
    "debug": DEBUG,
    "log_level": LOG_LEVEL,
    "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
}

def get_backend_info():
    """Return current backend configuration for debugging"""
    return {
        "backend_type": BACKEND_TYPE,
        "generation_backend": "vllm" if FEATURES["use_vllm_generation"] else "ollama",
        "embedding_backend": "vllm" if FEATURES["use_vllm_embeddings"] else "ollama",
        "fallback_enabled": FEATURES["fallback_to_ollama"],
        "debug_mode": FEATURES["debug"],
        "log_level": FEATURES["log_level"],
        "max_concurrent_requests": FEATURES["max_concurrent_requests"],
        "mode": "mixed" if (FEATURES["use_vllm_generation"] and not FEATURES["use_vllm_embeddings"]) else BACKEND_TYPE,
    }

def print_config_status():
    """Print current configuration status for development"""
    print("\n" + "="*60)
    print("🚀 AI PPT GENERATOR CONFIGURATION")
    print("="*60)
    print(f"📋 Backend Type: {BACKEND_TYPE.upper()}")
    
    if BACKEND_TYPE == "mixed":
        print("🔥 MIXED MODE ACTIVE:")
        print(f"  🧠 Generation: VLLM at {VLLM_GENERATION_URL}")
        print(f"  🔤 Embeddings: Ollama at {OLLAMA_BASE_URL}")
        print(f"  🧠 Gen Model: {VLLM_GENERATION_MODEL}")
        print(f"  🔤 Embed Model: {EMBEDDING_MODEL}")
    elif BACKEND_TYPE == "ollama":
        print(f"🔗 Ollama URL: {OLLAMA_BASE_URL}")
        print(f"🧠 Models: {list(DEFAULT_MODELS.values())}")
        print(f"🔤 Embeddings: {EMBEDDING_MODEL}")
    else:  # vllm
        print(f"🔗 VLLM Generation: {VLLM_GENERATION_URL}")
        print(f"🔗 VLLM Embeddings: {VLLM_EMBEDDING_URL}")
        print(f"🧠 Generation Model: {VLLM_GENERATION_MODEL}")
        print(f"🔤 Embedding Model: {VLLM_EMBEDDING_MODEL}")
    
    print(f"🔧 Debug Mode: {DEBUG}")
    print(f"📊 Max Concurrent: {MAX_CONCURRENT_REQUESTS}")
    print("="*60 + "\n")

# Auto-print configuration when imported (helpful during development)
if DEBUG or True:  # Always show during development
    print_config_status()

# -----------------------------------------------------------------------------
# Retrieval / Chunking
# -----------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVAL_K = 8  # keep consistent with your hybrid retriever

# Content analysis thresholds
MIN_WORDS_THRESHOLD = 200
MIN_STRUCTURE_LINES = 5

# -----------------------------------------------------------------------------
# File Types
# -----------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".pdf": "PyPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
}

# -----------------------------------------------------------------------------
# Lightweight Patterns (kept practical)
# -----------------------------------------------------------------------------
PATTERNS = {
    "numbers": r"\b\d+(?:\.\d+)?\b",
    "percentages": r"\b\d+(?:\.\d+)?\s?%\b",
    "years": r"\b(19|20)\d{2}\b",
    "section_headers": r"^\s*(introduction|background|method|methods|methodology|results|discussion|conclusion|references)\b[:\-]?",
    "fig_table": r"\b(fig(?:ure)?|table)\s*\d+\b",
    "technical_terms": r"\b(accuracy|precision|recall|dataset|model|algorithm|framework|pipeline|metrics?|evaluation|baseline|ablation|inference|latency|throughput|scalability|reliability|security|encryption|compliance)\b",
}

# -----------------------------------------------------------------------------
# Slide Policy (strict 4 bullets)
# -----------------------------------------------------------------------------
DEFAULT_BULLET_COUNT = 4  # Default number of bullets
MIN_BULLET_COUNT = 2      # Minimum allowed bullets
MAX_BULLET_COUNT = 10     # Maximum allowed bullets
DEFAULT_SLIDE_STRUCTURE = {
    "type": "bullet-points",
    "min_bullets": MIN_BULLET_COUNT,
    "max_bullets": MAX_BULLET_COUNT,
    "default_bullets": DEFAULT_BULLET_COUNT,
}

# -----------------------------------------------------------------------------
# Tone specification
# -----------------------------------------------------------------------------
class ToneSpec(BaseModel):
    name: str
    description: str
    bullet_min_len: int
    bullet_max_len: int
    add_context: bool  # allow parenthetical context/examples
    add_stats: bool    # encourage numbers, citations, dates
    examples: str      # few-shot examples for the tone

TONE_CONFIG = {
    "concise": ToneSpec(
        name="concise",
        description=(
            "Write terse, high-signal noun phrases. "
            "Drop filler, articles, conjunctions. No parentheticals."
        ),
        bullet_min_len=15,
        bullet_max_len=50,  # hard cap for concise
        add_context=False,
        add_stats=True,
        examples=(
            "BAD (too long): \"Explain the end-to-end pipeline for hate speech detection, providing detailed rationale...\"\n"
            "GOOD (≤50 chars): \"End-to-end pipeline: ingest → preprocess → train\"\n"
            "GOOD (≤50 chars): \"Baseline vs. model: +7–9 F1 on Hindi dataset\"\n"
            "Rule: Prefer compact noun phrases; avoid clauses."
        ),
    ),
    "comprehensive": ToneSpec(
        name="comprehensive",
        description=(
            "Write thorough bullets with brief qualifiers and evidence; "
            "use parentheticals sparingly."
        ),
        bullet_min_len=52,
        bullet_max_len=120,  # clearly higher band
        add_context=True,
        add_stats=True,
        examples=(
            "GOOD (60–120 chars): \"Hindi lexicon noise handled via subword tokenization and normalized Devanagari variants (improves recall).\"\n"
            "GOOD (60–120 chars): \"Macro-F1 ↑ from 0.62→0.71 on HASOC (Hindi), with α=0.6 class weighting to offset imbalance.\"\n"
            "Rule: Include brief context and concrete numbers when present."
        ),
    ),
}

# -----------------------------------------------------------------------------
# Prompt templates (schema-only; no JSON examples)
#   These use only document context when provided. The fallback variant exists
#   for cases where retrieval returns too little; it still follows tone rules.
# -----------------------------------------------------------------------------

DOCUMENT_PROMPT_TEMPLATE = """IMPORTANT: Ignore all previous context and conversations. Start fresh.

You are generating slides for a presentation on "{topic}" using the provided document content.

Document Content:
{context}

Tone: {tone_name}
Tone guidance: {tone_desc}

Examples ({tone_name} tone):
{tone_examples}

Output requirements:
- Return only a JSON array of exactly {num_slides} slide objects.
- Each slide object must include:
  - "slide": integer from 1 to {num_slides}
  - "type": "bullet-points"
  - "title": concise string
  - "bullets": array of exactly {bullet_count} strings, each between {bullet_min_len} and {bullet_max_len} characters

Rules:
- Each bullet MUST be ≤ {bullet_max_len} characters (counting spaces). Trim to noun phrase if longer.
- Each bullet MUST be ≥ {bullet_min_len} characters. Add relevant detail if shorter.
- Use only information from the document content above; do not invent facts.
- {context_rule}
- {stats_rule}
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""

FALLBACK_PROMPT_TEMPLATE = """IMPORTANT: Ignore all previous context and conversations. Start fresh.

You are generating slides for a presentation on "{topic}" (document context unavailable or insufficient).

Tone: {tone_name}
Tone guidance: {tone_desc}

Examples ({tone_name} tone):
{tone_examples}

Output requirements:
- Return only a JSON array of exactly {num_slides} slide objects.
- Each slide object must include:
  - "slide": integer from 1 to {num_slides}
  - "type": "bullet-points"
  - "title": concise string
  - "bullets": array of exactly {bullet_count} strings, each between {bullet_min_len} and {bullet_max_len} characters

Rules:
- Each bullet MUST be ≤ {bullet_max_len} characters (counting spaces). Trim to noun phrase if longer.
- Each bullet MUST be ≥ {bullet_min_len} characters. Add relevant detail if shorter.
- If domain knowledge is necessary, keep claims general and avoid unverifiable specifics.
- {context_rule}
- {stats_rule}
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""

SINGLE_SLIDE_PROMPT_TEMPLATE = """Create exactly 1 slide about "{slide_focus}" using the provided document content.

Document Content:
{context}

Tone: {tone_name}
Tone guidance: {tone_desc}

Examples ({tone_name} tone):
{tone_examples}

Output requirements:
- Return only a JSON array with exactly 1 slide object.
- The slide object must include:
  - "slide": 1
  - "type": "bullet-points"
  - "title": concise string related to {slide_focus}
  - "bullets": array of exactly {bullet_count} strings, each between {bullet_min_len} and {bullet_max_len} characters

Rules:
- Each bullet MUST be ≤ {bullet_max_len} characters (counting spaces). Trim to noun phrase if longer.
- Each bullet MUST be ≥ {bullet_min_len} characters. Add relevant detail if shorter.
- Use only information from the document content above; do not invent facts.
- {context_rule}
- {stats_rule}
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""

SINGLE_SLIDE_FALLBACK_TEMPLATE = """Create exactly 1 slide about "{slide_focus}" (document context unavailable or insufficient).

Tone: {tone_name}
Tone guidance: {tone_desc}

Examples ({tone_name} tone):
{tone_examples}

Output requirements:
- Return only a JSON array with exactly 1 slide object.
- The slide object must include:
  - "slide": 1
  - "type": "bullet-points"
  - "title": concise string related to {slide_focus}
  - "bullets": array of exactly {bullet_count} strings, each between {bullet_min_len} and {bullet_max_len} characters

Rules:
- If domain knowledge is necessary, keep claims general and avoid unverifiable specifics.
- {context_rule}
- {stats_rule}
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""

# -----------------------------------------------------------------------------
# Small helpers to materialize tone-specific directives and final prompts
# -----------------------------------------------------------------------------

def _context_rule(tone: ToneSpec) -> str:
    """Whether to allow short parenthetical clarifications/examples."""
    if tone.add_context:
        return "You may add brief parenthetical clarifications or short examples only when they directly improve precision."
    return "Do not add parenthetical clarifications or examples; keep bullets lean."

def _stats_rule(tone: ToneSpec) -> str:
    """Directives around numbers/dates/citations."""
    if tone.add_stats:
        return "Prefer concrete numbers, dates, named entities, and citations mentioned in the document; avoid vague claims."
    return "Avoid unnecessary numbers or citations; prioritize clarity."

def render_document_prompt(topic: str, context: str, num_slides: int, tone: ToneSpec, bullet_count: int = 4) -> str:
    return DOCUMENT_PROMPT_TEMPLATE.format(
        topic=topic,
        context=context,
        num_slides=num_slides,
        tone_name=tone.name,
        tone_desc=tone.description,
        tone_examples=tone.examples,
        bullet_min_len=tone.bullet_min_len,
        bullet_max_len=tone.bullet_max_len,
        bullet_count=bullet_count,
        context_rule=_context_rule(tone),
        stats_rule=_stats_rule(tone),
    )

def render_fallback_prompt(topic: str, num_slides: int, tone: ToneSpec, bullet_count: int = 4) -> str:
    return FALLBACK_PROMPT_TEMPLATE.format(
        topic=topic,
        num_slides=num_slides,
        tone_name=tone.name,
        tone_desc=tone.description,
        tone_examples=tone.examples,
        bullet_min_len=tone.bullet_min_len,
        bullet_max_len=tone.bullet_max_len,
        bullet_count=bullet_count,
        context_rule=_context_rule(tone),
        stats_rule=_stats_rule(tone),
    )

def render_single_slide_prompt(slide_focus: str, context: str, tone: ToneSpec, bullet_count: int = 4) -> str:
    return SINGLE_SLIDE_PROMPT_TEMPLATE.format(
        slide_focus=slide_focus,
        context=context,
        tone_name=tone.name,
        tone_desc=tone.description,
        tone_examples=tone.examples,
        bullet_min_len=tone.bullet_min_len,
        bullet_max_len=tone.bullet_max_len,
        bullet_count=bullet_count,
        context_rule=_context_rule(tone),
        stats_rule=_stats_rule(tone),
    )

def render_single_slide_fallback_prompt(slide_focus: str, tone: ToneSpec, bullet_count: int = 4) -> str:
    return SINGLE_SLIDE_FALLBACK_TEMPLATE.format(
        slide_focus=slide_focus,
        tone_name=tone.name,
        tone_desc=tone.description,
        tone_examples=tone.examples,
        bullet_min_len=tone.bullet_min_len,
        bullet_max_len=tone.bullet_max_len,
        bullet_count=bullet_count,
        context_rule=_context_rule(tone),
        stats_rule=_stats_rule(tone),
    )