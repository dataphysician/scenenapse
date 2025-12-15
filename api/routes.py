"""
API Routes for SceneNapse Studio.

Defines the FastAPI endpoints for scene generation, refinement, and state management.
"""

import base64
import os
import traceback
from typing import Any

import requests
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
import dspy
import fal_client

from .schemas import (
    SceneResponse,
    RefineRequest,
    RefineResponse,
    HealthResponse,
    TimingInfo,
)
from .dependencies import get_generation_pipeline, get_refinement_pipeline
from .state import get_scene_state, reset_scene_state, DEFAULT_CINEMATOGRAPHY, ReferenceImage, GeneratedImage

from backend.dspy_signatures import (
    Elements,
    Element,
    SceneObject,
    Action,
    Cinematography,
    CinematographyCamera,
    CinematographyLighting,
    CinematographyComposition,
    CinematographyLook,
    Scene,
)


# ============================================================
# Request/Response Models for State Management
# ============================================================

class SceneStateResponse(BaseModel):
    """Current scene state from backend memory."""
    elements: dict[str, Any]
    objects: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    cinematography: dict[str, Any] | None
    short_description: str
    critic_score: float
    critic_issues: list[str]
    reference_images: list[dict[str, Any]] = []
    selected_reference: dict[str, Any] | None = None
    generated_images: list[dict[str, Any]] = []
    selected_generated: dict[str, Any] | None = None


class ElementsUpdate(BaseModel):
    """Update request for elements."""
    elements: list[dict[str, Any]]


class ObjectsUpdate(BaseModel):
    """Update request for objects."""
    objects: list[dict[str, Any]]


class ActionsUpdate(BaseModel):
    """Update request for actions."""
    actions: list[dict[str, Any]]


class CinematographyUpdate(BaseModel):
    """Update request for cinematography."""
    cinematography: dict[str, Any] | None

router = APIRouter()


@router.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@router.post("/api/generate", response_model=SceneResponse)
async def generate_scene(
    prompt: str = Form(..., description="Scene description prompt"),
    image: UploadFile | None = File(None, description="Optional reference image"),
    video: UploadFile | None = File(None, description="Optional reference video"),
    audio: UploadFile | None = File(None, description="Optional reference audio"),
) -> SceneResponse:
    """
    Generate a complete scene from a prompt and optional multimodal inputs.

    - **prompt**: Natural language description of the desired scene
    - **image**: Optional reference image file
    - **video**: Optional reference video file
    - **audio**: Optional reference audio file

    Returns a complete Scene with elements, objects, actions, cinematography,
    critic validation, and a summary description.
    """
    try:
        pipeline = get_generation_pipeline()
        state = get_scene_state()

        # Check for selected reference image from state
        reference_context = ""
        if state.selected_reference:
            ref_url = state.selected_reference.url
            ref_title = state.selected_reference.title or "reference image"
            reference_context = f"\n\n[Use this reference image for visual style and composition: {ref_url} ({ref_title})]"
            print(f"[generate] Including selected reference image: {ref_title}")

        # Append reference context to prompt if available
        if reference_context:
            prompt = prompt + reference_context

        # Convert uploaded files to DSPy types
        # DSPy Image/Audio accept bytes via the `url` parameter
        image_data = None
        if image and image.filename:
            image_bytes = await image.read()
            if image_bytes:
                image_data = dspy.Image(url=image_bytes)

        video_data = None
        if video and video.filename:
            # Note: DSPy doesn't natively support video files.
            # Video would need frame extraction (opencv) to work as image input.
            # For now, skip video processing and log a warning.
            print(f"Warning: Video file '{video.filename}' uploaded but video processing is not yet supported. Skipping.")
            # TODO: Implement frame extraction from video using opencv-python
            # video_bytes = await video.read()
            # if video_bytes:
            #     # Extract first frame using cv2
            #     video_data = dspy.Image(url=first_frame_bytes)

        audio_data = None
        if audio and audio.filename:
            audio_bytes = await audio.read()
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                # Extract format from mime type (e.g., 'audio/mpeg' -> 'mpeg')
                audio_format = (audio.content_type or "audio/mpeg").split('/')[-1]
                audio_data = dspy.Audio(data=audio_b64, audio_format=audio_format)

        # Run the async pipeline
        result = await pipeline.aforward(
            prompt=prompt,
            image=image_data,
            video=video_data,
            audio=audio_data,
            verbose=True,  # Enable verbose logging for debugging
        )

        # Convert timing dataclass to our schema
        timing = TimingInfo(
            stage1_elements_sec=result.timing.stage1_elements_sec,
            stage2_parallel_sec=result.timing.stage2_parallel_sec,
            stage3_parallel_sec=result.timing.stage3_parallel_sec,
            total_sec=result.timing.total_sec,
        )

        # Store the generated scene in backend state
        state = get_scene_state()
        state.from_scene(result.scene)
        state.short_description = result.short_description
        state.critic_score = result.critic_score
        state.critic_issues = result.critic_issues

        return SceneResponse(
            scene=result.scene,
            critic_issues=result.critic_issues,
            critic_score=result.critic_score,
            short_description=result.short_description,
            timing=timing,
            summary_tokens=result.summary_tokens,
            retry_count=result.retry_count,
        )

    except Exception as e:
        print(f"[API] Error in /api/generate: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/refine", response_model=RefineResponse)
async def refine_scene(
    scene_json: str = Form(..., description="JSON string of the current scene"),
    instruction: str = Form(..., description="Natural language refinement instruction"),
    regenerate_summary: bool = Form(True, description="Whether to regenerate the scene summary"),
    image: UploadFile | None = File(None, description="Optional reference image for refinement"),
    audio: UploadFile | None = File(None, description="Optional reference audio for refinement"),
) -> RefineResponse:
    """
    Refine an existing scene with a natural language instruction.

    - **scene_json**: JSON string of the current complete scene to modify
    - **instruction**: Natural language instruction describing the change
    - **regenerate_summary**: Whether to regenerate the scene summary
    - **image**: Optional reference image to guide refinement
    - **audio**: Optional reference audio to guide refinement

    Returns the refined scene with critic validation and summary.

    CRITICAL: If the instruction affects ELEMENTS (entity changes like woman→man),
    all dependent heads (OBJECTS, ACTIONS) are regenerated for consistency.
    """
    try:
        # Parse the scene JSON
        import json
        scene_dict = json.loads(scene_json)
        scene = Scene(**scene_dict)

        pipeline = get_refinement_pipeline()
        state = get_scene_state()

        # Process multimodal inputs and add context to instruction
        multimodal_context = []

        # Include selected reference image from state if available
        if state.selected_reference:
            ref_url = state.selected_reference.url
            ref_title = state.selected_reference.title or "reference image"
            multimodal_context.append(f"Use this reference image for visual style and composition: {ref_url} ({ref_title})")
            print(f"[refine] Including selected reference image: {ref_title}")

        # Process uploaded image if provided
        if image and image.filename:
            image_bytes = await image.read()
            if image_bytes:
                multimodal_context.append("Uploaded reference image provided - incorporate its visual style and composition")

        # Process uploaded audio if provided
        if audio and audio.filename:
            audio_bytes = await audio.read()
            if audio_bytes:
                multimodal_context.append("Reference audio provided - incorporate its mood, tempo, and atmosphere")

        # Append multimodal context to instruction
        if multimodal_context:
            instruction = f"{instruction}\n\n[{'; '.join(multimodal_context)}]"

        # Run the refinement pipeline - now returns RefinementResult
        result = pipeline.forward(
            scene=scene,
            instruction=instruction,
            regenerate_summary=regenerate_summary,
            verbose=True,  # Enable verbose for debugging
        )

        # Update backend state with refined scene
        state = get_scene_state()
        state.from_scene(result.scene)
        state.short_description = result.short_description or ""
        state.critic_score = result.critic_score
        state.critic_issues = result.critic_issues

        return RefineResponse(
            scene=result.scene,
            short_description=result.short_description,
            affected_heads=result.affected_heads,
            critic_score=result.critic_score,
            critic_issues=result.critic_issues,
            retry_count=result.retry_count,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Scene State Management Endpoints
# ============================================================

@router.get("/api/scene", response_model=SceneStateResponse)
async def get_scene() -> SceneStateResponse:
    """
    Get the current scene state from backend memory.

    Returns the complete scene with all components (elements, objects,
    actions, cinematography) as well as metadata.
    """
    state = get_scene_state()
    data = state.to_dict()
    return SceneStateResponse(**data)


@router.delete("/api/scene")
async def clear_scene() -> dict[str, str]:
    """
    Clear the current scene state and reset to empty.
    """
    reset_scene_state()
    return {"status": "cleared"}


@router.put("/api/scene/elements", response_model=SceneStateResponse)
async def update_elements(update: ElementsUpdate) -> SceneStateResponse:
    """
    Update the elements in backend memory.

    Validates the elements against the Pydantic model and stores them.
    """
    try:
        state = get_scene_state()

        # Parse and validate each element
        validated_elements = []
        for elem_dict in update.elements:
            element = Element(**elem_dict)
            validated_elements.append(element)

        state.elements = Elements(elements=validated_elements)

        data = state.to_dict()
        return SceneStateResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid elements: {str(e)}")


@router.put("/api/scene/objects", response_model=SceneStateResponse)
async def update_objects(update: ObjectsUpdate) -> SceneStateResponse:
    """
    Update the objects in backend memory.

    Validates the objects against the Pydantic model and stores them.
    """
    try:
        state = get_scene_state()

        # Parse and validate each object
        validated_objects = []
        for obj_dict in update.objects:
            obj = SceneObject(**obj_dict)
            validated_objects.append(obj)

        state.objects = validated_objects

        data = state.to_dict()
        return SceneStateResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid objects: {str(e)}")


@router.put("/api/scene/actions", response_model=SceneStateResponse)
async def update_actions(update: ActionsUpdate) -> SceneStateResponse:
    """
    Update the actions in backend memory.

    Validates the actions against the Pydantic model and stores them.
    """
    try:
        state = get_scene_state()

        # Parse and validate each action
        validated_actions = []
        for act_dict in update.actions:
            action = Action(**act_dict)
            validated_actions.append(action)

        state.actions = validated_actions

        data = state.to_dict()
        return SceneStateResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid actions: {str(e)}")


@router.put("/api/scene/cinematography", response_model=SceneStateResponse)
async def update_cinematography(update: CinematographyUpdate) -> SceneStateResponse:
    """
    Update the cinematography in backend memory.

    Validates the cinematography against the Pydantic model and stores it.
    """
    try:
        state = get_scene_state()

        if update.cinematography is None:
            state.cinematography = None
        else:
            # Parse nested cinematography components
            cine_dict = update.cinematography

            camera = CinematographyCamera(**(cine_dict.get("camera", {})))
            lighting = CinematographyLighting(**(cine_dict.get("lighting", {})))
            composition = CinematographyComposition(**(cine_dict.get("composition", {})))
            look = CinematographyLook(**(cine_dict.get("look", {})))

            state.cinematography = Cinematography(
                dependencies=cine_dict.get("dependencies", []),
                camera=camera,
                lighting=lighting,
                composition=composition,
                look=look,
            )

        data = state.to_dict()
        return SceneStateResponse(**data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid cinematography: {str(e)}")


@router.get("/api/scene/assembled")
async def get_assembled_scene() -> dict[str, Any]:
    """
    Get the fully assembled Scene as a validated Python object, serialized to JSON.

    This ensures all components are properly validated and combined into a
    coherent Scene object before returning.
    """
    try:
        state = get_scene_state()
        scene = state.to_scene()
        return scene.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assemble scene: {str(e)}")


class AssembleRequest(BaseModel):
    """Request for assembling/validating existing scene data."""
    pass  # No body needed - uses current backend state


class AssembleResponse(BaseModel):
    """Response from the assemble endpoint."""
    scene: dict[str, Any]
    critic_issues: list[str]
    critic_score: float
    short_description: str


@router.post("/api/assemble", response_model=AssembleResponse)
async def assemble_scene() -> AssembleResponse:
    """
    Assemble and validate the current scene state (manual mode).

    This endpoint is used when the user has manually filled in scene components
    (elements, objects, actions, cinematography) via the UI and wants to:
    1. Validate the assembled scene with the critic
    2. Generate a summary description

    Unlike /api/generate, this does NOT run the full PromptToElements pipeline.
    It only runs Stage 3 (Critic + Summary) on the existing scene data.

    Returns the validated scene with critic score and summary.
    """
    import asyncio
    import dspy

    from backend.dspy_signatures import SceneCritic, SceneSummary
    from backend.dspy_pipeline import MODEL_CRITIC, MODEL_SUMMARY, summary_not_truncated

    try:
        state = get_scene_state()

        # Check if we have minimum data to assemble
        if not state.elements.elements:
            raise HTTPException(
                status_code=400,
                detail="No elements defined. Add at least one element before assembling."
            )

        # Build the Scene from current state
        scene = state.to_scene()

        # Run Stage 3: Critic + Summary
        scene_critic = dspy.Predict(SceneCritic)
        scene_summary = dspy.Predict(SceneSummary, max_tokens=512)

        # Run critic
        critic_lm = dspy.LM(MODEL_CRITIC)
        with dspy.settings.context(lm=critic_lm):
            critic_result = await scene_critic.acall(scene=scene)

        # Run summary with BestOfN for quality
        summary_lm = dspy.LM(MODEL_SUMMARY, max_tokens=512)
        with dspy.settings.context(lm=summary_lm):
            summary_predictor = dspy.BestOfN(
                module=dspy.Predict(SceneSummary),
                N=3,
                reward_fn=summary_not_truncated,
                threshold=1.0,
            )
            summary_result = await asyncio.to_thread(summary_predictor, scene=scene)

        # Update state with results
        state.short_description = summary_result.short_description
        state.critic_score = critic_result.consistency_score
        state.critic_issues = critic_result.issues

        return AssembleResponse(
            scene=scene.model_dump(),
            critic_issues=critic_result.issues,
            critic_score=critic_result.consistency_score,
            short_description=summary_result.short_description,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assembly failed: {str(e)}")


# ============================================================
# Image Generation Endpoint (Fal AI)
# ============================================================

class ImageGenerateRequest(BaseModel):
    """Request for image generation."""
    prompt: str | None = None  # Optional - will use scene summary if not provided
    num_images: int = 3
    aspect_ratio: str = "1:1"
    output_format: str = "png"
    resolution: str = "1K"


class GeneratedImageResult(BaseModel):
    """A single generated image result from API."""
    url: str
    width: int
    height: int
    content_type: str


class ImageGenerateResponse(BaseModel):
    """Response from image generation."""
    images: list[GeneratedImageResult]
    prompt_used: str
    request_id: str


@router.post("/api/generate-images", response_model=ImageGenerateResponse)
async def generate_images(request: ImageGenerateRequest) -> ImageGenerateResponse:
    """
    Generate images using Fal AI's nano-banana-pro model.

    - **prompt**: Optional prompt override. If not provided, uses the scene summary.
    - **num_images**: Number of images to generate (default: 3)
    - **aspect_ratio**: Aspect ratio of images (default: "1:1")
    - **output_format**: Output format - png or jpeg (default: "png")
    - **resolution**: Resolution - 1K or 2K (default: "1K")

    Uses asyncio.gather to generate multiple images in parallel for speed.
    Returns generated image URLs and metadata.
    """
    import asyncio

    try:
        # Determine the prompt to use
        prompt = request.prompt
        if not prompt:
            state = get_scene_state()
            prompt = state.short_description
            if not prompt:
                raise HTTPException(
                    status_code=400,
                    detail="No prompt provided and no scene summary available. Generate a scene first or provide a prompt."
                )

        # Generate a single image (used for parallel generation)
        async def generate_single_image() -> dict | None:
            try:
                handler = await fal_client.submit_async(
                    "fal-ai/nano-banana-pro",
                    arguments={
                        "prompt": prompt,
                        "num_images": 1,
                        "aspect_ratio": "1:1",
                        "output_format": "png",
                        "resolution": "1K",
                    },
                )
                result = await handler.get()
                imgs = result.get("images", [])
                return imgs[0] if imgs else None
            except Exception as e:
                print(f"Single image generation failed: {e}")
                return None

        # Generate all images in parallel using asyncio.gather (4 images)
        tasks = [generate_single_image() for _ in range(4)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract successful images
        images = []
        state = get_scene_state()
        state.generated_images = []  # Clear previous generated images
        state.selected_generated = None  # Clear selection

        for img in results:
            # Skip failed results
            if img is None or isinstance(img, Exception):
                continue
            img_result = GeneratedImageResult(
                url=img.get("url", ""),
                width=img.get("width", 0),
                height=img.get("height", 0),
                content_type=img.get("content_type", f"image/{request.output_format}"),
            )
            images.append(img_result)

            # Store in state with 1-based index
            state.generated_images.append(GeneratedImage(
                index=len(state.generated_images) + 1,
                url=img_result.url,
                width=img_result.width,
                height=img_result.height,
            ))

        import uuid
        return ImageGenerateResponse(
            images=images,
            prompt_used=prompt,
            request_id=str(uuid.uuid4()),  # Generate unique request ID
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


# ============================================================
# Image Generation Streaming Endpoint (SSE)
# ============================================================

@router.get("/api/generate-images-stream")
async def generate_images_stream(prompt: str | None = None):
    """
    Stream generated images via Server-Sent Events as each completes.

    Launches 4 parallel image generation tasks and streams each result
    as it becomes available, allowing the frontend to display images
    progressively.

    Includes selected reference image URL in the prompt if available.
    """
    import asyncio

    state = get_scene_state()

    # Get prompt from param or state
    actual_prompt = prompt
    if not actual_prompt:
        actual_prompt = state.short_description
        if not actual_prompt:
            # Return error event
            async def error_gen():
                yield f"data: {json.dumps({'error': 'No prompt available'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")

    # Include selected reference image in the prompt if available
    if state.selected_reference:
        ref_url = state.selected_reference.url
        ref_title = state.selected_reference.title or "reference image"
        actual_prompt = f"{actual_prompt}\n\nUse this reference image for visual style and composition: {ref_url} ({ref_title})"
        print(f"[generate-images-stream] Including reference image: {ref_title}")

    async def generate_single():
        try:
            handler = await fal_client.submit_async(
                "fal-ai/nano-banana-pro",
                arguments={
                    "prompt": actual_prompt,
                    "num_images": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                    "resolution": "1K",
                },
            )
            result = await handler.get()
            imgs = result.get("images", [])
            return imgs[0] if imgs else None
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None

    async def score_image_from_url(url: str) -> float | None:
        """Download image from URL and score with JoyQuality."""
        from backend.joy_quality import get_joy_quality_selector
        from PIL import Image
        import io

        selector = get_joy_quality_selector()
        if not selector:
            return None

        try:
            # Download image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")

            # Score image
            score = selector.score_image(image)
            return score
        except Exception as e:
            print(f"[JoyQuality] Failed to score image: {e}")
            return None

    async def event_generator():
        # Clear state
        state = get_scene_state()
        state.generated_images = []
        state.selected_generated = None

        # Create 4 tasks
        tasks = [asyncio.create_task(generate_single()) for _ in range(4)]

        index = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                index += 1
                url = result.get("url", "")

                # Score image with JoyQuality
                quality_score = await asyncio.to_thread(
                    lambda: score_image_from_url.__wrapped__(url) if hasattr(score_image_from_url, '__wrapped__') else None
                )
                # Actually call the sync version directly in thread
                from backend.joy_quality import get_joy_quality_selector
                from PIL import Image
                import io

                selector = get_joy_quality_selector()
                if selector and url:
                    try:
                        response = requests.get(url, timeout=10)
                        response.raise_for_status()
                        image = Image.open(io.BytesIO(response.content)).convert("RGB")
                        quality_score = selector.score_image(image)
                        print(f"[JoyQuality] Image {index} score: {quality_score:.4f}")
                    except Exception as e:
                        print(f"[JoyQuality] Failed to score image {index}: {e}")
                        quality_score = None
                else:
                    quality_score = None

                img_data = {
                    "url": url,
                    "width": result.get("width", 0),
                    "height": result.get("height", 0),
                    "index": index,
                    "quality_score": quality_score
                }
                # Store in state
                state.generated_images.append(GeneratedImage(
                    index=index,
                    url=img_data["url"],
                    width=img_data["width"],
                    height=img_data["height"],
                    quality_score=quality_score,
                ))
                yield f"data: {json.dumps(img_data)}\n\n"

        # Send done event
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================
# Image Search Endpoint (Freepik API)
# ============================================================

class ImageSearchRequest(BaseModel):
    """Request for image search."""
    query: str | None = None  # Optional - will use scene summary if not provided


class ImageSearchResult(BaseModel):
    """A single image search result."""
    id: int
    url: str
    thumbnail: str | None = None
    title: str | None = None


class ImageSearchResponse(BaseModel):
    """Response from image search."""
    images: list[ImageSearchResult]
    query_used: str
    total: int


@router.post("/api/search-images", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest) -> ImageSearchResponse:
    """
    Search for reference images using Freepik API.

    - **query**: Optional search query. If not provided, uses the scene summary.

    Returns image results with URLs and thumbnails.
    """
    try:
        # Determine the query to use
        query = request.query
        if not query:
            state = get_scene_state()
            query = state.short_description
            if not query:
                raise HTTPException(
                    status_code=400,
                    detail="No query provided and no scene summary available. Generate a scene first or provide a query."
                )

        # Freepik API configuration
        api_key = os.getenv("FREEPIK_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="FREEPIK_API_KEY environment variable not set."
            )

        headers = {
            "x-freepik-api-key": api_key,
            "Accept-Language": "en-US"
        }

        # Build params with proper filter syntax per Freepik docs
        params: dict[str, Any] = {
            "order": "relevance",
            "term": query,
            "page": 1,
            "limit": 12,
            "filters[content_type][photo]": "1",
            "filters[orientation][landscape]": "1",
        }

        response = requests.get(
            "https://api.freepik.com/v1/resources",
            headers=headers,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print(f"Freepik API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Freepik API error: {response.text}"
            )

        data = response.json()
        print(f"Freepik response: {len(data.get('data', []))} results")

        # Extract image results
        images = []
        for item in data.get("data", []):
            # Get the image URL from the source
            thumbnail_url = None
            image_data = item.get("image", {})
            if image_data:
                source = image_data.get("source", {})
                thumbnail_url = source.get("url")

            # The item URL is the Freepik page URL for that resource
            item_url = item.get("url", "")

            images.append(ImageSearchResult(
                id=item.get("id", 0),
                url=item_url,
                thumbnail=thumbnail_url,
                title=item.get("title"),
            ))

        # Meta structure per docs: meta.total, meta.current_page, meta.last_page
        meta = data.get("meta", {})
        total = meta.get("total", len(images))

        # Store results in state for later selection by index
        state = get_scene_state()
        state.reference_images = [
            ReferenceImage(id=img.id, url=img.url, thumbnail=img.thumbnail, title=img.title)
            for img in images
        ]
        state.selected_reference = None  # Clear selection on new search

        return ImageSearchResponse(
            images=images,
            query_used=query,
            total=total,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


# ============================================================
# Select Reference Image Endpoint
# ============================================================

class SelectReferenceRequest(BaseModel):
    """Request to select a reference image by index (1-based)."""
    index: int = Field(..., ge=1, description="1-based index of the reference image to select")


class SelectReferenceResponse(BaseModel):
    """Response after selecting a reference image."""
    success: bool
    selected: ImageSearchResult | None
    message: str


@router.post("/api/select-reference", response_model=SelectReferenceResponse)
async def select_reference(request: SelectReferenceRequest) -> SelectReferenceResponse:
    """
    Select a reference image by index (1-based) from the last search results.

    - **index**: 1-based index of the image to select (1 = first image)

    Returns the selected image details. Use index 0 or call again with same index to deselect.
    """
    try:
        state = get_scene_state()

        if not state.reference_images:
            raise HTTPException(
                status_code=400,
                detail="No reference images available. Run a search first."
            )

        # Convert 1-based index to 0-based
        idx = request.index - 1

        if idx < 0 or idx >= len(state.reference_images):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid index {request.index}. Available range: 1-{len(state.reference_images)}"
            )

        ref_img = state.reference_images[idx]

        # Toggle: if already selected, deselect
        if state.selected_reference and state.selected_reference.id == ref_img.id:
            state.selected_reference = None
            return SelectReferenceResponse(
                success=True,
                selected=None,
                message=f"Deselected reference image {request.index}"
            )

        # Select the image
        state.selected_reference = ref_img

        return SelectReferenceResponse(
            success=True,
            selected=ImageSearchResult(
                id=ref_img.id,
                url=ref_img.url,
                thumbnail=ref_img.thumbnail,
                title=ref_img.title,
            ),
            message=f"Selected reference image {request.index}: {ref_img.title or 'Untitled'}"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to select reference: {str(e)}")


# ============================================================
# Deselect Reference Image Endpoint
# ============================================================

@router.post("/api/deselect-reference")
async def deselect_reference():
    """
    Deselect the currently selected reference image.
    """
    try:
        state = get_scene_state()
        state.selected_reference = None
        return {"success": True, "message": "Reference image deselected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deselect reference: {str(e)}")


# ============================================================
# Select Generated Image Endpoint
# ============================================================

class SelectGeneratedRequest(BaseModel):
    """Request to select a generated image by index (1-based)."""
    index: int = Field(..., ge=1, description="1-based index of the generated image to select")


class SelectGeneratedImageResult(BaseModel):
    """A selected generated image."""
    index: int
    url: str
    width: int
    height: int


class SelectGeneratedResponse(BaseModel):
    """Response after selecting a generated image."""
    success: bool
    selected: SelectGeneratedImageResult | None
    message: str


@router.post("/api/select-generated", response_model=SelectGeneratedResponse)
async def select_generated(request: SelectGeneratedRequest) -> SelectGeneratedResponse:
    """
    Select a generated image by index (1-based) from the last generation.

    - **index**: 1-based index of the image to select (1 = first image)

    Returns the selected image details. Call again with same index to deselect.
    """
    try:
        state = get_scene_state()

        if not state.generated_images:
            raise HTTPException(
                status_code=400,
                detail="No generated images available. Generate images first."
            )

        # Convert 1-based index to 0-based
        idx = request.index - 1

        if idx < 0 or idx >= len(state.generated_images):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid index {request.index}. Available range: 1-{len(state.generated_images)}"
            )

        gen_img = state.generated_images[idx]

        # Toggle: if already selected, deselect
        if state.selected_generated and state.selected_generated.index == gen_img.index:
            state.selected_generated = None
            return SelectGeneratedResponse(
                success=True,
                selected=None,
                message=f"Deselected generated image {request.index}"
            )

        # Select the image
        state.selected_generated = gen_img

        return SelectGeneratedResponse(
            success=True,
            selected=SelectGeneratedImageResult(
                index=gen_img.index,
                url=gen_img.url,
                width=gen_img.width,
                height=gen_img.height,
            ),
            message=f"Selected generated image {request.index}"
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to select generated image: {str(e)}")
