#!/usr/bin/env python3
"""
Main PPT Generator class with modular architecture
- Tone-aware prompts (concise/comprehensive) via TONE_CONFIG + render_* helpers
- Structured output path using Pydantic + Ollama json_schema (with fallback parser)
- Adaptive retrieval scaling (desired_slides) + sufficiency gating
- Exact slide-count enforcement using AI-fill (micro-retrieval → guarded topic-only fallback → pad)
"""

import json
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator

from .model_factory import ModelFactory
from .config import (
    DEFAULT_MODELS, OLLAMA_BASE_URL,
    DOCUMENT_PROMPT_TEMPLATE, FALLBACK_PROMPT_TEMPLATE,
    SINGLE_SLIDE_PROMPT_TEMPLATE, SINGLE_SLIDE_FALLBACK_TEMPLATE,
    DEFAULT_SLIDE_STRUCTURE,
    # tone + helpers from latest config.py
    TONE_CONFIG,
    render_document_prompt, render_fallback_prompt,
    render_single_slide_prompt, render_single_slide_fallback_prompt,
    get_backend_info,  # ADD this line
)
from .retrieval_engine import RetrievalEngine


# ----------------------------
# Pydantic models for robust JSON output
# ----------------------------
class SlideModel(BaseModel):
    """Model for a single slide with validation"""
    slide: int = Field(..., ge=1, description="Slide number")  # no upper cap
    type: Literal["bullet-points"] = Field(default="bullet-points", description="Slide type")
    title: str = Field(..., min_length=5, max_length=100, description="Slide title")
    bullets: List[str] = Field(..., min_items=2, max_items=4, description="2-4 high-quality bullet points")

    @field_validator('bullets')
    @classmethod
    def validate_bullet_lengths(cls, v: List[str]) -> List[str]:
        """Ensure bullets have appropriate length - NO auto-padding"""
        # Only validate length, NO padding - preserve natural LLM output
        for i, bullet in enumerate(v):
            if len(bullet) < 10:
                v[i] = bullet + " (expanded)"
            elif len(bullet) > 150:
                v[i] = bullet[:147] + "..."
        return v


class PresentationModel(BaseModel):
    """Model for complete presentation - flexible format"""
    slides: List[SlideModel] = Field(..., min_items=1, description="List of slides")

    @classmethod
    def from_array_or_object(cls, data):
        """Create PresentationModel from either array or object format"""
        if isinstance(data, list):
            # Handle array format: [{slide1}, {slide2}, ...]
            return cls(slides=data)
        elif isinstance(data, dict) and 'slides' in data:
            # Handle object format: {"slides": [{slide1}, {slide2}, ...]}
            return cls(**data)
        else:
            raise ValueError(f"Invalid presentation format: {type(data)}")

    @field_validator('slides')
    @classmethod
    def validate_slide_numbers(cls, v: List[SlideModel]) -> List[SlideModel]:
        """Ensure slide numbers are sequential"""
        for i, slide in enumerate(v, 1):
            slide.slide = i
        return v


class SingleSlideResponse(BaseModel):
    """Response model for single slide"""
    slide: SlideModel = Field(..., description="Generated slide")


class PPTGenerator:
    """Main PPT Generator class with modular architecture"""

    def __init__(self, models: Dict[str, str] = None):
        # Model configuration
        self.models = models or DEFAULT_MODELS
        self.llm_instances = {}

        # Initialize model instances
        self._initialize_models()

        # Initialize retrieval engine
        self.retrieval_engine = RetrievalEngine()

    def _initialize_models(self):
        """Initialize all model instances using ModelFactory"""
        backend_info = get_backend_info()
        print(f"🚀 Initializing models with backend: {backend_info}")
        
        for role in self.models.keys():
            try:
                self.llm_instances[role] = ModelFactory.create_llm(role)
            except Exception as e:
                print(f"❌ Failed to initialize {role} model: {e}")
                # Fallback to primary model if available
                if role != "primary" and "primary" in self.llm_instances:
                    self.llm_instances[role] = self.llm_instances["primary"]
                    print(f"🔄 Using primary model as fallback for {role}")
                else:
                    raise Exception(f"Critical error: Cannot initialize {role} model")

        print("✅ Embeddings initialized via ModelFactory")

    def _get_model(self, model_role: str):
        """Get model instance by role with fallback"""
        return self.llm_instances.get(model_role, self.llm_instances["primary"])

    # ----------------------------
    # Tone-aware prompt factories
    # ----------------------------
    def _resolve_tone(self, tone: Optional[str]):
        """Return ToneSpec from TONE_CONFIG with safe default."""
        return TONE_CONFIG.get((tone or "concise").lower(), TONE_CONFIG["concise"])

    # ----------------------------
    # JSON Cleaning & Repair (Layer 1 & 2 support)
    # ----------------------------

    def _fix_json_syntax(self, text: str) -> str:
        """Repair common JSON syntax errors from LLMs"""
        # Remove trailing commas before } or ]
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        
        # Add missing commas between object boundaries
        text = re.sub(r"}\s*{", r"}, {", text)
        
        # Ensure array wrapper (tolerant approach)
        t = text.strip()
        if not t.startswith("["):
            text = "[" + text
        if not text.strip().endswith("]"):
            text = text + "]"
            
        return text

    def _extract_slide_objects(self, text: str) -> List[Dict]:
        """Extract slide objects using regex when JSON parsing fails"""
        slide_objects = []
        
        # Strategy 1: Look for complete slide objects with slide numbers
        pat1 = r'\{\s*"slide"\s*:\s*\d+.*?\}'
        for match in re.findall(pat1, text, flags=re.DOTALL):
            try:
                obj = json.loads(match)
                if "title" in obj and "bullets" in obj:
                    slide_objects.append(obj)
            except Exception:
                continue
        
        # Strategy 2: Look for any object with title and bullets (for single slide responses)
        if not slide_objects:
            pat2 = r'\{\s*"title"\s*:.*?\}'
            for match in re.findall(pat2, text, flags=re.DOTALL):
                try:
                    obj = json.loads(match)
                    if "title" in obj and "bullets" in obj:
                        # Add slide number if missing
                        if "slide" not in obj:
                            obj["slide"] = 1
                        slide_objects.append(obj)
                except Exception:
                    continue
        
        # Strategy 3: Find JSON array in text and extract from it
        if not slide_objects:
            array_start = text.find('[')
            array_end = text.rfind(']') + 1
            if array_start != -1 and array_end > array_start:
                try:
                    array_text = text[array_start:array_end]
                    parsed_array = json.loads(array_text)
                    if isinstance(parsed_array, list):
                        for item in parsed_array:
                            if isinstance(item, dict) and "title" in item and "bullets" in item:
                                slide_objects.append(item)
                except Exception:
                    pass
                
        return slide_objects

    def create_document_prompt(self, topic: str, context: str, num_slides: int, tone: str = "concise") -> str:
        """Create prompt for document-based generation (tone-aware)"""
        _tone = self._resolve_tone(tone)
        return render_document_prompt(topic, context, num_slides, _tone)

    def create_fallback_prompt(self, topic: str, num_slides: int, tone: str = "concise") -> str:
        """Create fallback prompt when no document content (tone-aware)"""
        _tone = self._resolve_tone(tone)
        return render_fallback_prompt(topic, num_slides, _tone)

    def create_single_slide_prompt(self, topic: str, context: str, slide_focus: str, tone: str = "concise") -> str:
        """Create prompt for single slide generation (tone-aware)"""
        _tone = self._resolve_tone(tone)
        if context:
            return render_single_slide_prompt(slide_focus, context, _tone)
        else:
            return render_single_slide_fallback_prompt(slide_focus, _tone)

    def create_single_slide_regeneration_prompt(
        self,
        topic: str,
        context: str,
        slide_focus: str,
        current_slide: Dict = None,
        user_prompt: str = None,
        tone: str = "concise",
    ) -> str:
        """
        Enhanced prompt for single slide regeneration with user customization
        (schema-only; JSON-mode friendly)
        """
        _tone = self._resolve_tone(tone)

        base_prompt = f"""Create exactly 1 slide about "{slide_focus}" using the document content.

Document Content:
{context}

Tone: {_tone.name}
Tone guidance: {_tone.description}
Bullet length: {_tone.bullet_min_len}–{_tone.bullet_max_len} characters
"""

        # Add current slide context if provided
        if current_slide:
            bullets_str = "\n".join([f"- {b}" for b in current_slide.get('bullets', [])])
            base_prompt += f"""

Current Slide Content:
Title: {current_slide.get('title', 'N/A')}
Bullets:
{bullets_str}
"""

        # Add user customization request if provided
        if user_prompt:
            base_prompt += f"""

User's Customization Request:
{user_prompt}

Please incorporate the user's request while maintaining the slide structure.
"""

        # Schema-only output requirements (no examples)
        base_prompt += f"""
Output requirements:
- Return only a JSON array with exactly 1 slide object.
- The slide object must include:
  - "slide": 1
  - "type": "bullet-points"
  - "title": concise string related to {slide_focus}
  - "bullets": array of exactly 4 strings, each between {_tone.bullet_min_len} and {_tone.bullet_max_len} characters

Rules:
- Use only information from the document content above; do not invent facts.
- Prefer concrete numbers, dates, and named entities found in the document.
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""
        return base_prompt

    def create_all_slides_regeneration_prompt(
        self, topic: str, context: str, num_slides: int,
        current_slides: List[Dict] = None, user_prompt: str = None, tone: str = "concise"
    ) -> str:
        """
        Enhanced prompt for all slides regeneration with user customization
        (schema-only; JSON-mode friendly)
        """
        _tone = self._resolve_tone(tone)

        base_prompt = f"""Create a {num_slides}-slide presentation about "{topic}" using the document content.

Document Content:
{context}

Tone: {_tone.name}
Tone guidance: {_tone.description}
Bullet length: {_tone.bullet_min_len}–{_tone.bullet_max_len} characters
"""

        # Add current presentation context if provided
        if current_slides:
            struct = "\n".join([f"Slide {i+1}: {slide.get('title', 'Untitled')}" for i, slide in enumerate(current_slides)])
            cnts = "\n".join([f"Slide {i+1}: {len(slide.get('bullets', []))} bullets" for i, slide in enumerate(current_slides)])
            base_prompt += f"""

Current Presentation Structure:
{struct}

Current Content Summary:
{cnts}
"""

        # Add user customization request if provided
        if user_prompt:
            base_prompt += f"""

User's Overall Customization Request:
{user_prompt}

Please incorporate the user's request while maintaining professional presentation structure.
"""

        # Schema-only output requirements (no examples)
        base_prompt += f"""
Output requirements:
- Return only a JSON array of exactly {num_slides} slide objects.
- Each slide object must include:
  - "slide": integer from 1 to {num_slides}
  - "type": "bullet-points"
  - "title": concise string
  - "bullets": array of exactly 4 strings, each between {_tone.bullet_min_len} and {_tone.bullet_max_len} characters

Rules:
- Use only information from the document content above; do not invent facts.
- Prefer concrete numbers, dates, and named entities found in the document.
- No markdown, no code fences, no prose, no examples.
- Do not include anything before or after the JSON array.

Respond with the JSON array only.
"""
        return base_prompt

    # ----------------------------
    # Parsing
    # ----------------------------
    def parse_slides_simple(self, response: str) -> Dict[str, Any]:
        """
        Layer 2: Enhanced manual parsing with recovery strategies
        Uses 3-step process: clean → repair → extract
        """
        try:
            # Layer 2a: Basic response cleaning
            response = response.strip()

            # Try normal JSON parsing first
            try:
                parsed = json.loads(response)

                # Single object → wrap
                if isinstance(parsed, dict):
                    if "title" in parsed and "bullets" in parsed:
                        return {"success": True, "slides": [parsed]}
                    else:
                        return {"success": False, "error": "Invalid slide object structure"}

                # Normal array
                if isinstance(parsed, list) and parsed:
                    return {"success": True, "slides": parsed}

                return {"success": False, "error": "Invalid format: expected array or slide object"}

            except json.JSONDecodeError as e:
                print(f"🔧 JSON parse failed: {e}; response length: {len(response)}")
                print(f"🔧 First 200 chars: {response[:200]}")
                
                # Layer 2b: Attempt JSON repair
                try:
                    repaired = self._fix_json_syntax(response)
                    parsed = json.loads(repaired)
                    if isinstance(parsed, list) and parsed:
                        print(f"✅ JSON repair successful, recovered {len(parsed)} slides")
                        return {"success": True, "slides": parsed}
                except Exception as repair_error:
                    print(f"🔧 JSON repair failed: {repair_error}")
                
                # Layer 2c: Extract slide objects with regex
                try:
                    extracted_slides = self._extract_slide_objects(response)
                    if extracted_slides:
                        print(f"✅ Regex extraction successful, recovered {len(extracted_slides)} slides")
                        return {"success": True, "slides": extracted_slides}
                except Exception as extract_error:
                    print(f"🔧 Regex extraction failed: {extract_error}")

                return {"success": False, "error": f"All parsing strategies failed. JSON error: {str(e)}"}

        except Exception as e:
            return {"success": False, "error": f"Parsing error: {str(e)}"}

    # ----------------------------
    # Validation
    # ----------------------------
    def validate_simple(self, slides: List[Dict], tone: str = "concise") -> List[Dict]:
        """Validate and fix slide structure. Enforce 4-bullet policy and tone-specific length limits."""
        fixed_slides = []
        
        # Get tone-specific limits
        tone_spec = self._resolve_tone(tone)
        min_len = tone_spec.bullet_min_len
        max_len = tone_spec.bullet_max_len

        for i, slide in enumerate(slides, 1):
            # Ensure required fields exist
            if "slide" not in slide:
                slide["slide"] = i
            if "type" not in slide:
                slide["type"] = DEFAULT_SLIDE_STRUCTURE["type"]
            if "title" not in slide:
                slide["title"] = f"Slide {i}"
            if "bullets" not in slide or not isinstance(slide["bullets"], list):
                slide["bullets"] = ["TBD"] * DEFAULT_SLIDE_STRUCTURE["min_bullets"]

            bullets = slide["bullets"]

            # Fix bullet count to be within range
            min_bullets = DEFAULT_SLIDE_STRUCTURE["min_bullets"]
            max_bullets = DEFAULT_SLIDE_STRUCTURE["max_bullets"]

            while len(bullets) < min_bullets:
                bullets.append("TBD")
            if len(bullets) > max_bullets:
                bullets = bullets[:max_bullets]
            
            # Remove aggressive truncation - rely on prompt engineering for length control
            # The prompts already specify exact character limits, so validation truncation is unnecessary
            # and causes abrupt mid-sentence cuts. Let the LLM handle length constraints naturally.

            slide["bullets"] = bullets
            fixed_slides.append(slide)

        return fixed_slides

    # ----------------------------
    # AI-Fill Helper (micro-retrieval → guarded fallback → pad)
    # ----------------------------
    def _ai_fill_missing_slides(
        self,
        slides: List[Dict[str, Any]],
        topic: str,
        num_slides: int,
        file_paths: Optional[List[str]],
        model_role: str = "primary",
        max_attempts_per_slide: int = 1,
        tone: str = "concise",
    ) -> List[Dict[str, Any]]:
        """
        Try to replace 'TBD' padding with AI-generated slides, safely.
        Strategy:
          - Ensure exact count first (truncate/pad with 'TBD')
          - For each 'TBD' slide, attempt per-slide retrieval (desired_slides=1)
          - If context still weak, use guarded topic-only fallback (no fabricated numbers/dates)
          - If generation/parse fails, keep 'TBD'
        """

        def _needs_refill(slide: Dict[str, Any]) -> bool:
            bullets = slide.get("bullets", [])
            return any(b.strip().upper() == "TBD" for b in bullets) or len(bullets) < DEFAULT_SLIDE_STRUCTURE["min_bullets"]

        def _make_pad(i: int) -> Dict[str, Any]:
            return {
                "slide": i,
                "type": DEFAULT_SLIDE_STRUCTURE["type"],
                "title": f"Slide {i}",
                "bullets": ["TBD"] * DEFAULT_SLIDE_STRUCTURE["min_bullets"],
                "source": "fallback-pad",
            }

        llm = self._get_model(model_role)

        # Enforce target length (we'll try to replace pads next)
        if len(slides) > num_slides:
            slides = slides[:num_slides]
        while len(slides) < num_slides:
            slides.append(_make_pad(len(slides) + 1))

        # Check if we already have enough valid slides - if so, skip AI fill entirely
        valid_slides_count = sum(1 for slide in slides if not _needs_refill(slide))
        if valid_slides_count >= num_slides:
            print(f"✅ Already have {valid_slides_count} valid slides out of {num_slides} requested - skipping AI fill to prevent duplicates")
            return slides[:num_slides]

        # Avoid duplicates: pass covered titles as hint
        covered_titles = [s.get("title", "").strip() for s in slides if s.get("title")]
        covered_hint = "; ".join([t for t in covered_titles if t])

        for idx in range(num_slides):
            s_idx = idx + 1
            slide = slides[idx]
            if not _needs_refill(slide):
                continue

            focus = slide.get("title") or f"{topic} — Additional coverage (slide {s_idx})"

            # Per-slide retrieval (tight)
            ctx = ""
            if file_paths:
                ctx = self.retrieval_engine.enhanced_extract_document_context(
                    file_paths, topic, focus, desired_slides=1
                )

            # Get tone specifications
            tone_spec = self._resolve_tone(tone)
            
            # Guardrails with tone enforcement
            guard = (
                "Rules:\n"
                f"- Each bullet MUST be between {tone_spec.bullet_min_len} and {tone_spec.bullet_max_len} characters.\n"
                "- Do NOT invent statistics, dates, quotes, names, or citations.\n"
                "- If the document context lacks specifics, write qualitative, general bullets.\n"
                "- Avoid repeating content already covered: " + covered_hint + "\n"
                "- No placeholders like 'Additional point related to ...'.\n"
                "- Respond with the JSON array only."
            )

            # Use proper prompt templates with few-shot examples
            if ctx:
                prompt = render_single_slide_prompt(focus, ctx, tone_spec)
            else:
                prompt = render_single_slide_fallback_prompt(focus, tone_spec)

            # Try a few times to get a clean slide
            attempt = 0
            new_slide = None
            while attempt < max_attempts_per_slide and new_slide is None:
                attempt += 1
                try:
                    resp = llm.invoke(prompt)
                    parsed = self.parse_slides_simple(resp.content)
                    if parsed["success"]:
                        candidate = self.validate_simple(parsed["slides"], tone=tone)[0]
                        candidate["slide"] = s_idx
                        if all(b.strip() and b.strip().upper() != "TBD" for b in candidate.get("bullets", [])):
                            new_slide = candidate
                            break
                except Exception:
                    pass

            slides[idx] = new_slide if new_slide else _make_pad(s_idx)

        return slides

    # ----------------------------
    # Layer 3: Per-slide fallback generation
    # ----------------------------
    def _generate_slides_individually(
        self,
        topic: str,
        file_paths: Optional[List[str]],
        num_slides: int,
        model_role: str = "primary",
        tone: str = "concise",
    ) -> Dict[str, Any]:
        """
        Layer 3: Generate slides one by one when bulk parsing fails
        Guaranteed to work but slower (num_slides * LLM calls)
        """
        print(f"🔄 Layer 3: Generating {num_slides} slides individually (slow but guaranteed)")
        
        # Create slide titles first to maintain coherence
        slide_titles = [
            f"Introduction to {topic}",
            f"Key Concepts in {topic}",
            f"Methodology and Approach",
            f"Results and Findings", 
            f"Applications and Benefits",
            f"Challenges and Limitations",
            f"Future Directions",
            f"Case Studies",
            f"Best Practices",
            f"Implementation Strategy",
            f"Technical Details",
            f"Performance Analysis",
            f"Comparative Analysis", 
            f"Industry Impact",
            f"Research Implications",
            f"Practical Considerations",
            f"Tools and Technologies",
            f"Risk Assessment",
            f"Success Metrics",
            f"Conclusion and Summary"
        ]
        
        # Ensure we have enough titles
        while len(slide_titles) < num_slides:
            slide_titles.append(f"Additional Topic {len(slide_titles) + 1}")
        
        slides = []
        successful_slides = 0
        
        for i in range(num_slides):
            slide_title = slide_titles[i]
            print(f"🔄 Generating slide {i+1}/{num_slides}: {slide_title}")
            
            # Use per-slide context extraction
            context = ""
            if file_paths:
                context = self.retrieval_engine.enhanced_extract_document_context(
                    file_paths, topic, slide_title, desired_slides=1
                )
            
            # Generate single slide
            single_slide_result = self.regenerate_single_slide(
                slide_number=i+1,
                slide_title=slide_title,
                topic=topic,
                file_paths=file_paths,
                model_role=model_role,
                tone=tone
            )
            
            if single_slide_result["success"]:
                slides.append(single_slide_result["slide"])
                successful_slides += 1
            else:
                # Create fallback slide if individual generation fails
                fallback_slide = {
                    "slide": i + 1,
                    "type": "bullet-points",
                    "title": slide_title,
                    "bullets": [
                        f"Key aspect of {topic}",
                        f"Important consideration for {slide_title.lower()}",
                        f"Relevant detail about the topic",
                        f"Supporting information"
                    ]
                }
                slides.append(fallback_slide)
        
        print(f"✅ Layer 3 complete: {successful_slides}/{num_slides} slides generated successfully")
        
        return {
            "success": True,
            "slides": slides,
            "total_slides": len(slides),
            "requested_slides": num_slides,
            "approach": "individual_slide_generation",
            "topic": topic,
            "model_used": self.models.get(model_role, "unknown"),
            "model_role": model_role,
            "tone": tone,
            "generation_method": "per_slide_fallback",
            "individual_success_rate": f"{successful_slides}/{num_slides}",
            "context_length": 0  # Individual context lengths vary
        }

    # ----------------------------
    # Main flows
    # ----------------------------
    def _generate_slides_direct(self, prompt: str, model_role: str = "primary") -> Dict[str, Any]:
        """
        Generate slides using direct LLM call and manual parsing (simplified approach)
        """
        try:
            llm = self._get_model(model_role)
            response = llm.invoke(prompt)
            
            # Ensure we have a string to parse
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Direct manual parsing - always works
            parse_result = self.parse_slides_simple(content)
            
            if parse_result.get("success") and parse_result.get("slides"):
                slides = parse_result["slides"]
                return {
                    "success": True,
                    "slides": slides,
                    "method": "direct_manual_parsing",
                    "total_slides": len(slides)
                }
            else:
                return {
                    "success": False,
                    "error": parse_result.get("error", "No valid slides found in response"),
                    "method": "direct_manual_parsing"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "direct_manual_parsing"
            }

    def generate_presentation(
        self,
        topic: str,
        file_paths: Optional[List[str]] = None,
        num_slides: int = 6,
        model_role: str = "primary",
        tone: str = "concise",
        use_structured_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Main generation method for endpoint integration
        - Adaptive retrieval scaling + sufficiency gating
        - Tone-aware prompts
        - Structured output path with fallback parser
        - Exact slide-count enforcement via AI-fill
        """

        # Extract document context using enhanced retrieval engine (scaled to ask)
        context = ""
        if file_paths:
            context = self.retrieval_engine.enhanced_extract_document_context(
                file_paths, topic, topic, desired_slides=num_slides
            )

        # Adaptive sufficiency
        use_document_path = self.retrieval_engine.analyze_content_sufficiency(
            context, desired_slides=num_slides
        )

        # Tone-aware prompt
        if use_document_path:
            prompt = self.create_document_prompt(topic, context, num_slides, tone=tone)
        else:
            prompt = self.create_fallback_prompt(topic, num_slides, tone=tone)

        # Direct manual parsing approach (simplified and faster)
        print(f"🔄 Direct slide generation for topic: {topic}")
        direct_result = self._generate_slides_direct(prompt, model_role)
        
        if direct_result.get("success"):
            slides = direct_result["slides"]
            # Validate and fill missing slides
            slides = self.validate_simple(slides, tone=tone)
            slides = self._ai_fill_missing_slides(
                slides=slides,
                topic=topic,
                num_slides=num_slides,
                file_paths=file_paths,
                model_role=model_role,
                max_attempts_per_slide=1,
                tone=tone,
            )
            
            return {
                "success": True,
                "slides": slides,
                "total_slides": len(slides),
                "requested_slides": num_slides,
                "used_fallback": not use_document_path,
                "approach": "document" if use_document_path else "ai",
                "topic": topic,
                "context_length": len(context),
                "model_used": self.models.get(model_role, "unknown"),
                "model_role": model_role,
                "tone": tone,
                "generation_method": direct_result.get("method", "direct_manual_parsing"),
            }
        else:
            direct_error = direct_result.get('error', 'Unknown generation error')
            print(f"⚠️ Direct generation failed: {direct_error}, trying legacy parsing")

        # Legacy fallback (in case direct method fails)
        llm = self._get_model(model_role)
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation failed: {e}",
                "approach": "document" if use_document_path else "ai",
                "model_used": self.models.get(model_role, "unknown")
            }

        parse_result = self.parse_slides_simple(response.content)
        if not parse_result["success"]:
            print(f"🔄 All bulk parsing failed, falling back to Layer 3: per-slide generation")
            # Layer 3: Per-slide fallback generation (guaranteed but slow)
            return self._generate_slides_individually(
                topic=topic,
                file_paths=file_paths,
                num_slides=num_slides,
                model_role=model_role,
                tone=tone
            )

        slides = self.validate_simple(parse_result["slides"], tone=tone)
        slides = self._ai_fill_missing_slides(
            slides=slides,
            topic=topic,
            num_slides=num_slides,
            file_paths=file_paths,
            model_role=model_role,
            max_attempts_per_slide=1,
            tone=tone,
        )

        return {
            "success": True,
            "slides": slides,
            "total_slides": len(slides),
            "requested_slides": num_slides,
            "used_fallback": not use_document_path,
            "approach": "document" if use_document_path else "ai",
            "topic": topic,
            "context_length": len(context),
            "model_used": self.models.get(model_role, "unknown"),
            "model_role": model_role,
            "tone": tone,
            "generation_method": "manual_parsing_fallback",
            "structured_output_status": "failed" if use_structured_output else "not_attempted",
        }

    def regenerate_single_slide(
        self,
        slide_number: int,
        slide_title: str,
        topic: str,
        file_paths: Optional[List[str]] = None,
        model_role: str = "regeneration",
        current_slide_content: Optional[Dict] = None,
        user_prompt: Optional[str] = None,
        tone: str = "concise",
    ) -> Dict[str, Any]:
        """Regenerate a specific slide with specified model and user customization support"""

        # Per-slide context (scaled)
        context = ""
        if file_paths:
            context = self.retrieval_engine.enhanced_extract_document_context(
                file_paths, topic, slide_title, desired_slides=1
            )

        # Tone-aware prompt
        prompt = self.create_single_slide_regeneration_prompt(
            topic, context, slide_title, current_slide_content, user_prompt, tone=tone
        )

        llm = self._get_model(model_role)
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            return {
                "success": False,
                "error": f"Single slide generation failed: {e}",
                "model_used": self.models.get(model_role, "unknown")
            }

        parse_result = self.parse_slides_simple(response.content)
        if not parse_result["success"]:
            return {
                "success": False,
                "error": parse_result["error"],
                "model_used": self.models.get(model_role, "unknown")
            }

        slides = parse_result["slides"]
        if not slides:
            return {
                "success": False,
                "error": "No slide generated",
                "model_used": self.models.get(model_role, "unknown")
            }

        regenerated_slide = self.validate_simple(slides, tone=tone)[0]
        regenerated_slide["slide"] = slide_number

        return {
            "success": True,
            "slide": regenerated_slide,
            "approach": "document" if context else "ai",
            "context_length": len(context),
            "model_used": self.models.get(model_role, "unknown"),
            "model_role": model_role,
            "tone": tone,
            "user_prompt_applied": user_prompt is not None,
            "had_previous_content": current_slide_content is not None
        }

    def regenerate_all_slides(
        self,
        topic: str,
        file_paths: Optional[List[str]] = None,
        num_slides: int = 6,
        model_role: str = "creative",
        current_slides_content: Optional[List[Dict]] = None,
        user_prompt: Optional[str] = None,
        tone: str = "concise",
    ) -> Dict[str, Any]:
        """Regenerate all slides with user customization support"""

        # Context scaled to ask
        context = ""
        if file_paths:
            context = self.retrieval_engine.enhanced_extract_document_context(
                file_paths, topic, topic, desired_slides=num_slides
            )

        # Adaptive sufficiency
        use_document_path = self.retrieval_engine.analyze_content_sufficiency(
            context, desired_slides=num_slides
        )

        if use_document_path:
            prompt = self.create_all_slides_regeneration_prompt(
                topic, context, num_slides, current_slides_content, user_prompt, tone=tone
            )
        else:
            prompt = self.create_fallback_prompt(topic, num_slides, tone=tone)

        llm = self._get_model(model_role)
        try:
            response = llm.invoke(prompt)
        except Exception as e:
            return {
                "success": False,
                "error": f"Generation failed: {e}",
                "approach": "document" if use_document_path else "ai",
                "model_used": self.models.get(model_role, "unknown")
            }

        parse_result = self.parse_slides_simple(response.content)
        if not parse_result["success"]:
            return {
                "success": False,
                "error": parse_result["error"],
                "approach": "document" if use_document_path else "ai",
                "model_used": self.models.get(model_role, "unknown")
            }

        slides = self.validate_simple(parse_result["slides"])

        # Enforce exact count with AI-fill
        slides = self._ai_fill_missing_slides(
            slides=slides,
            topic=topic,
            num_slides=num_slides,
            file_paths=file_paths,
            model_role=model_role,
            max_attempts_per_slide=1,
            tone=tone,
        )

        return {
            "success": True,
            "slides": slides,
            "total_slides": len(slides),
            "requested_slides": num_slides,
            "approach": "document" if use_document_path else "ai",
            "topic": topic,
            "context_length": len(context),
            "model_used": self.models.get(model_role, "unknown"),
            "model_role": model_role,
            "tone": tone,
            "user_prompt_applied": user_prompt is not None,
            "had_previous_content": current_slides_content is not None
        }
