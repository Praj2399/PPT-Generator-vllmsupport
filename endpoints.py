from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.responses import FileResponse
from typing import Optional, List
import os
import json
import tempfile
from pathlib import Path

# Import your models and classes
from models import (
    LogoSettings,
    TemplateType,
    LogoPosition1,
    LogoPosition2,
)
from ppt import PPTXGenerator
from template_manager import TemplateManager

# Import our enhanced PPTGenerator with Phase 1+2 features
# from ppt_generator import PPTGenerator
from ppt_generator.ppt_generator import PPTGenerator

# Initialize FastAPI app
app = FastAPI(
    title="PPT Generator API",
    description="API for generating PowerPoint presentations from documents and topics",
    version="1.0.0",
)

# Create router
router = APIRouter(prefix="/api", tags=["presentations"])

# Initialize generators
pptx_generator = PPTXGenerator(TemplateManager())
ppt_generator = PPTGenerator()


async def _save_uploaded_files(files: List[UploadFile]) -> List[str]:
    """Save uploaded files temporarily and return their paths"""
    file_paths = []

    for file in files:
        if file and file.filename:
            try:
                # Create temporary file with original extension
                suffix = Path(file.filename).suffix
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    file_paths.append(tmp_file.name)

                # Reset file pointer for potential reuse
                await file.seek(0)
            except Exception as e:
                # Clean up any files we've already saved
                for path in file_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                raise HTTPException(
                    status_code=400,
                    detail=f"Error saving uploaded file {file.filename}: {str(e)}",
                )

    return file_paths


def _cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except:
            pass  # Ignore cleanup errors


@router.post("/generate-content")
async def generate_content_form(
    topic: str = Form(..., min_length=3, description="Presentation topic"),
    num_slides: int = Form(6, ge=1, le=20, description="Number of slides"),
    files: List[UploadFile] = File(
        default=[], description="Optional document files (PDF, DOCX, TXT)"
    ),
    model_role: str = Form(
        "primary", description="Model role: primary, regeneration, or creative"
    ),
    tone: str = Form(
        "concise", description="Tone: concise (25-60 chars) or comprehensive (60-110 chars)"
    ),
):
    """
    Generate presentation content using PPTGenerator with optional document files

    - **topic**: The main topic for the presentation
    - **num_slides**: Number of slides to generate (1-20)
    - **files**: Optional document files to extract content from
    - **model_role**: Which model to use (primary, regeneration, creative)
    - **tone**: Writing style - 'concise' for brief bullets or 'comprehensive' for detailed bullets

    Returns generated slides with metadata about the generation process.
    """
    temp_file_paths = []

    try:
        # Save uploaded files temporarily
        if files and any(f.filename for f in files):
            temp_file_paths = await _save_uploaded_files(
                [f for f in files if f.filename]
            )

        # Generate presentation using PPTGenerator
        result = ppt_generator.generate_presentation(
            topic=topic,
            file_paths=temp_file_paths if temp_file_paths else None,
            num_slides=num_slides,
            model_role=model_role,
            tone=tone,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Content generation failed: {result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "slides": result["slides"],
            "total_slides": result["total_slides"],
            "approach": result["approach"],
            "topic": result["topic"],
            "context_length": result.get("context_length", 0),
            "model_used": result.get("model_used", "unknown"),
            "model_role": result.get("model_role", model_role),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Content generation failed: {str(e)}"
        )
    finally:
        # Clean up temporary files
        _cleanup_temp_files(temp_file_paths)


@router.post("/regenerate-slide")
async def regenerate_slide_form(
    slide_number: int = Form(..., ge=1, description="Slide number to regenerate"),
    slide_title: str = Form(..., min_length=1, description="Current slide title"),
    topic: str = Form(..., min_length=3, description="Presentation topic"),
    files: List[UploadFile] = File(default=[], description="Optional document files"),
    model_role: str = Form("regeneration", description="Model role for regeneration"),
    current_slide: Optional[str] = Form(
        None, description="JSON string of current slide content"
    ),
    user_prompt: Optional[str] = Form(None, description="User customization request"),
    tone: str = Form(
        "concise", description="Tone: concise (25-60 chars) or comprehensive (60-110 chars)"
    ),
):
    """
    Regenerate a specific slide using PPTGenerator

    - **slide_number**: The slide number to regenerate (1-based index)
    - **slide_title**: Current title of the slide to regenerate
    - **topic**: Main presentation topic for context
    - **files**: Optional document files for content extraction
    - **model_role**: Model to use for regeneration
    - **tone**: Writing style - 'concise' for brief bullets or 'comprehensive' for detailed bullets

    Returns the regenerated slide with metadata.
    """
    temp_file_paths = []

    try:
        # Save uploaded files temporarily
        if files and any(f.filename for f in files):
            temp_file_paths = await _save_uploaded_files(
                [f for f in files if f.filename]
            )

        # Parse current slide if provided
        current_slide_content = None
        if current_slide:
            try:
                current_slide_content = json.loads(current_slide)
            except:
                pass

        # Regenerate single slide with Phase 1 user prompt support
        result = ppt_generator.regenerate_single_slide(
            slide_number=slide_number,
            slide_title=slide_title,
            topic=topic,
            file_paths=temp_file_paths if temp_file_paths else None,
            model_role=model_role,
            current_slide_content=current_slide_content,
            user_prompt=user_prompt,
            tone=tone,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Slide regeneration failed: {result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "slide": result["slide"],
            "approach": result["approach"],
            "context_length": result.get("context_length", 0),
            "model_used": result.get("model_used", "unknown"),
            "model_role": result.get("model_role", model_role),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Slide regeneration failed: {str(e)}"
        )
    finally:
        # Clean up temporary files
        _cleanup_temp_files(temp_file_paths)


@router.post("/regenerate-all-slides")
async def regenerate_all_slides_form(
    topic: str = Form(..., min_length=3, description="Presentation topic"),
    num_slides: int = Form(6, ge=1, le=20, description="Number of slides"),
    files: List[UploadFile] = File(default=[], description="Optional document files"),
    model_role: str = Form(
        "creative", description="Model role for creative regeneration"
    ),
    current_slides: Optional[str] = Form(
        None, description="JSON string of current slides"
    ),
    user_prompt: Optional[str] = Form(
        None, description="User customization request for all slides"
    ),
    tone: str = Form(
        "concise", description="Tone: concise (25-60 chars) or comprehensive (60-110 chars)"
    ),
):
    """
    Regenerate all slides with creative variation using PPTGenerator

    - **topic**: The main topic for the presentation
    - **num_slides**: Number of slides to generate
    - **files**: Optional document files for content extraction
    - **model_role**: Model to use (typically 'creative' for variety)
    - **tone**: Writing style - 'concise' for brief bullets or 'comprehensive' for detailed bullets

    Returns all regenerated slides with metadata.
    """
    temp_file_paths = []

    try:
        # Save uploaded files temporarily
        if files and any(f.filename for f in files):
            temp_file_paths = await _save_uploaded_files(
                [f for f in files if f.filename]
            )

        # Parse current slides if provided
        current_slides_content = None
        if current_slides:
            try:
                current_slides_content = json.loads(current_slides)
            except:
                pass

        # Regenerate all slides with Phase 1 user prompt support
        result = ppt_generator.regenerate_all_slides(
            topic=topic,
            file_paths=temp_file_paths if temp_file_paths else None,
            num_slides=num_slides,
            model_role=model_role,
            current_slides_content=current_slides_content,
            user_prompt=user_prompt,
            tone=tone,
        )

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"All slides regeneration failed: {result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "slides": result["slides"],
            "total_slides": result["total_slides"],
            "approach": result["approach"],
            "topic": result["topic"],
            "context_length": result.get("context_length", 0),
            "model_used": result.get("model_used", "unknown"),
            "model_role": result.get("model_role", model_role),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"All slides regeneration failed: {str(e)}"
        )
    finally:
        # Clean up temporary files
        _cleanup_temp_files(temp_file_paths)


@router.post("/create-pptx")
async def create_pptx(
    slides: str = Form(..., description="JSON string of slides data"),
    template: TemplateType = Form(
        TemplateType.TEMPLATE_1, description="Presentation template"
    ),
    logo_1: UploadFile = File(None, description="First logo file"),
    logo1_position: LogoPosition1 = Form(
        LogoPosition1.NONE, description="First logo position"
    ),
    logo_2: UploadFile = File(None, description="Second logo file"),
    logo2_position: LogoPosition2 = Form(
        LogoPosition2.NONE, description="Second logo position"
    ),
    watermark: Optional[str] = Form(None, description="Watermark text"),
):
    """
    Create PPTX file from slides data

    - **slides**: JSON string containing slide data
    - **template**: Template type to use for the presentation
    - **logo_1**: Optional first logo file
    - **logo1_position**: Position for the first logo
    - **logo_2**: Optional second logo file
    - **logo2_position**: Position for the second logo
    - **watermark**: Optional watermark text

    Returns the generated PPTX file for download.
    """
    try:
        slides_data = json.loads(slides)

        if not slides_data:
            raise HTTPException(status_code=400, detail="No slides data provided")

        logo_settings = None
        if logo1_position != LogoPosition1.NONE or logo2_position != LogoPosition2.NONE:
            logo_settings = LogoSettings(
                position_1=logo1_position,
                position_2=logo2_position,
                apply_to_all_slides=True,
            )

        filepath = pptx_generator.create_presentation(
            slides_data=slides_data,
            template_type=template,
            logo_1=logo_1,
            logo_2=logo_2,
            logo_settings=logo_settings,
            watermark=watermark,
        )

        if not os.path.exists(filepath):
            raise HTTPException(
                status_code=500, detail="PPTX file was not created successfully"
            )

        return FileResponse(
            path=filepath,
            filename=os.path.basename(filepath),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format in slides data"
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PPTX creation failed: {str(e)}")
