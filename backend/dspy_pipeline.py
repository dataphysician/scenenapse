"""
Scene Generation Pipeline using DSPy.

This module provides the SceneGenerationPipeline class that orchestrates
the multi-head scene generation process using the signatures defined in
fibo_signatures.py.

Supports multimodal inputs:
  - prompt: str - Text description of the scene
  - image: dspy.Image | None - Reference image
  - video: dspy.Image | None - Reference video (as frames)
  - audio: dspy.Audio | None - Reference audio
"""

import asyncio
import time
from dataclasses import dataclass, field

import dspy

from .dspy_signatures import (
    # Pydantic models
    Elements,
    ObjectsBlock,
    ActionsBlock,
    Cinematography,
    Scene,
    # DSPy Signatures
    PromptToElements,
    SceneObjects,
    SceneActions,
    SceneCinematography,
    SceneCritic,
    SceneSummary,
)

# # Model constants
# MODEL_DEFAULT = "gemini/gemini-flash-latest"  # Fast, text-only
# MODEL_MULTIMODAL = "gemini/gemini-3-pro-preview"  # For image/video/audio
# MODEL_SUMMARY = "gemini/gemini-3-pro-preview"  # Best at respecting token limits
# MODEL_CRITIC = "gemini/gemini-2.5-flash-lite"  # Fast, cheap for critic

# Model constants
MODEL_DEFAULT = "openai/gpt-5.2"  # Fast, text-only
MODEL_MULTIMODAL = "gemini/gemini-2.5-flash-lite"  # For image/video/audio
MODEL_SUMMARY = "gemini/gemini-2.5-flash-lite"  # Best at respecting token limits
MODEL_CRITIC = "gemini/gemini-2.5-flash-lite"  # Fast, cheap for critic

# Pipeline constants
CRITIC_THRESHOLD = 0.85  # Minimum acceptable critic score
MAX_RETRIES = 2  # Maximum retry attempts if critic score is below threshold
BEST_OF_N = 3  # Number of parallel candidates to generate per attempt


def summary_not_truncated(args, pred) -> float:
    """Reward function: 1.0 if summary ends with proper punctuation, 0.0 if truncated."""
    text = getattr(pred, "short_description", None)
    if not text or not text.strip():
        return 0.0
    # Check if ends with sentence-ending punctuation (not truncated mid-sentence)
    return 1.0 if text.strip().endswith((".", "!", "?", '"', "'")) else 0.0


@dataclass
class StageTiming:
    """Timing information for pipeline stages."""
    stage1_elements_sec: float = 0.0
    stage2_parallel_sec: float = 0.0
    stage3_parallel_sec: float = 0.0
    total_sec: float = 0.0

    def __str__(self) -> str:
        return (
            f"Stage 1 (Elements):            {self.stage1_elements_sec:.2f}s\n"
            f"Stage 2 (Objects/Actions/Cin): {self.stage2_parallel_sec:.2f}s\n"
            f"Stage 3 (Critic/Summary):      {self.stage3_parallel_sec:.2f}s\n"
            f"Total:                         {self.total_sec:.2f}s"
        )


@dataclass
class ScenePipelineOutput:
    """Output of the scene generation pipeline."""
    scene: Scene
    critic_issues: list[str]
    critic_score: float
    short_description: str
    timing: StageTiming = field(default_factory=StageTiming)
    summary_tokens: int = 0
    retry_count: int = 0  # Number of retries due to low critic score


class SceneGenerationPipeline(dspy.Module):
    """
    DSPy module that generates a full scene from a prompt, including
    critic evaluation and summary.

    Pipeline stages:
      1. Prompt -> Elements (sequential)
      2. Prompt + Elements -> Objects, Actions, Cinematography (parallel)
      3. Scene -> Critic + Summary (parallel)
    """

    def __init__(self) -> None:
        super().__init__()
        # Stage 1: Prompt -> Elements
        self.prompt_to_elements = dspy.Predict(PromptToElements)

        # Stage 2: Parallel heads
        self.scene_objects = dspy.Predict(SceneObjects)
        self.scene_actions = dspy.Predict(SceneActions)
        self.scene_cinematography = dspy.Predict(SceneCinematography)

        # Stage 3: Critic + Summary
        self.scene_critic = dspy.Predict(SceneCritic)
        # Limit summary to ~512 tokens to stay within target length
        self.scene_summary = dspy.Predict(SceneSummary, max_tokens=512)

    def _extract_output_tokens(self, result) -> int:
        """Extract output token count from DSPy prediction result."""
        try:
            usage = result.get_lm_usage()
            if not usage:
                return 0
            # Try to find token count - structure varies by model/version
            for model_key, model_usage in usage.items():
                if isinstance(model_usage, dict):
                    # Try various possible keys
                    for key in ["output_tokens", "completion_tokens", "total_tokens"]:
                        if key in model_usage and model_usage[key]:
                            return model_usage[key]
        except Exception:
            pass
        return 0

    def forward(
        self,
        prompt: str,
        image: dspy.Image | None = None,
        video: dspy.Image | None = None,
        audio: dspy.Audio | None = None,
        verbose: bool = True,
    ) -> ScenePipelineOutput:
        """Synchronous execution of the pipeline with multimodal inputs.

        Uses Best-of-N approach: generates BEST_OF_N candidates in parallel,
        picks the one with highest critic score. Retries if best score < threshold.

        Args:
            prompt: Text description of the scene (can be empty if image/video provided)
            image: Optional reference image for visual analysis
            video: Optional reference video (as frames) for motion/dynamics analysis
            audio: Optional reference audio for mood/pacing cues
            verbose: Whether to print progress information
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        best_result: ScenePipelineOutput | None = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0 and verbose:
                print(f"\n{'=' * 60}")
                print(f"=== RETRY {attempt}/{MAX_RETRIES} (best critic score was {best_result.critic_score:.2f}) ===")
                print(f"{'=' * 60}")

            if verbose:
                print(f"\n[Best-of-{BEST_OF_N}] Generating {BEST_OF_N} candidates in parallel...")

            # Generate BEST_OF_N candidates in parallel
            candidates: list[ScenePipelineOutput] = []
            with ThreadPoolExecutor(max_workers=BEST_OF_N) as executor:
                futures = [
                    executor.submit(self._forward_once, prompt, image, video, audio, verbose=(verbose and i == 0))
                    for i in range(BEST_OF_N)
                ]
                errors: list[Exception] = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        candidates.append(result)
                    except Exception as e:
                        errors.append(e)
                        if verbose:
                            print(f"[Best-of-{BEST_OF_N}] Candidate failed: {e}")

            if not candidates:
                # Include the first error in the message for debugging
                first_error = errors[0] if errors else "Unknown error"
                raise RuntimeError(f"All candidates failed to generate. First error: {first_error}")

            # Pick the candidate with highest critic score
            best_candidate = max(candidates, key=lambda x: x.critic_score)

            if verbose:
                scores = [f"{c.critic_score:.2f}" for c in candidates]
                print(f"\n[Best-of-{BEST_OF_N}] Candidate scores: {', '.join(scores)}")
                print(f"[Best-of-{BEST_OF_N}] Selected best: {best_candidate.critic_score:.2f}")

            # Track overall best result across retries
            if best_result is None or best_candidate.critic_score > best_result.critic_score:
                best_result = best_candidate

            # If critic score meets threshold, return immediately
            if best_candidate.critic_score >= CRITIC_THRESHOLD:
                if verbose and attempt > 0:
                    print(f"\n[Retry] Success! Critic score {best_candidate.critic_score:.2f} >= {CRITIC_THRESHOLD}")
                best_candidate.retry_count = attempt
                return best_candidate

            if verbose:
                print(f"\n[Critic] Best score {best_candidate.critic_score:.2f} < {CRITIC_THRESHOLD} threshold")
                if attempt < MAX_RETRIES:
                    print(f"[Critic] Retrying with another Best-of-{BEST_OF_N}...")

        # Return best result after all retries
        if verbose:
            print(f"\n[Retry] Max retries reached. Returning best result (score: {best_result.critic_score:.2f})")
        best_result.retry_count = MAX_RETRIES
        return best_result

    def _forward_once(
        self,
        prompt: str,
        image: dspy.Image | None = None,
        video: dspy.Image | None = None,
        audio: dspy.Audio | None = None,
        verbose: bool = True,
    ) -> ScenePipelineOutput:
        """Single execution of the pipeline (no retry)."""
        timing = StageTiming()
        total_start = time.perf_counter()

        if verbose:
            print("=" * 60)
            print("=== INPUT PROMPT ===")
            print(prompt)
            if image:
                print("=== IMAGE PROVIDED ===")
            if video:
                print("=== VIDEO PROVIDED ===")
            if audio:
                print("=== AUDIO PROVIDED ===")
            print("=" * 60)

        # -------- Stage 1: Prompt -> Elements --------
        if verbose:
            print("\n[Stage 1] Extracting elements...")
        stage1_start = time.perf_counter()

        # Use multimodal model if image/video/audio provided (only Gemini supports audio)
        has_multimodal = image is not None or video is not None or audio is not None
        multimodal_lm = dspy.LM(MODEL_MULTIMODAL, cache=True) if has_multimodal else None

        if multimodal_lm:
            with dspy.settings.context(lm=multimodal_lm):
                elements_result = self.prompt_to_elements(prompt=prompt, image=image, video=video, audio=audio)
        else:
            elements_result = self.prompt_to_elements(prompt=prompt, image=image, video=video, audio=audio)
        elements: Elements = elements_result.elements

        timing.stage1_elements_sec = time.perf_counter() - stage1_start

        if verbose:
            print(f"[Stage 1] Complete ({timing.stage1_elements_sec:.2f}s)")
            print("\n=== ELEMENTS ===")
            for element in elements.elements:
                print(f"  - {element.element_id} ({element.role}): {element.entity_type}")

        # -------- Stage 2: Prompt + Elements -> three heads (sequential) --------
        # Use multimodal model if audio is present (only Gemini supports audio input)
        if verbose:
            print("\n[Stage 2] Generating objects, actions, cinematography (sequential)...")
        stage2_start = time.perf_counter()

        if multimodal_lm and audio is not None:
            # Use multimodal model for audio support
            with dspy.settings.context(lm=multimodal_lm):
                objects_result = self.scene_objects(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
                actions_result = self.scene_actions(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
                cin_result = self.scene_cinematography(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
        else:
            objects_result = self.scene_objects(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
            actions_result = self.scene_actions(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
            cin_result = self.scene_cinematography(prompt=prompt, image=image, video=video, audio=audio, elements=elements)

        objects_block: ObjectsBlock = objects_result.objects
        actions_block: ActionsBlock = actions_result.actions
        cinematography: Cinematography = cin_result.cinematography

        timing.stage2_parallel_sec = time.perf_counter() - stage2_start

        if verbose:
            print(f"[Stage 2] Complete ({timing.stage2_parallel_sec:.2f}s)")

            print("\n=== OBJECTS ===")
            for obj in objects_block.objects:
                deps = ", ".join(obj.dependencies) if obj.dependencies else ""
                print(f"\n  [{deps}] {obj.category} (relative_size: {obj.relative_size}, location: {obj.location})")
                print(f"    Description: {obj.description}")
                print(f"    Shape/Color: {obj.shape_and_color}")
                print(f"    Texture: {obj.texture}")
                print(f"    Appearance Details: {obj.appearance_details}")
                if obj.pose:
                    print(f"    Pose: {obj.pose.pose_class}, {obj.pose.body_orientation}")
                    if obj.pose.key_body_parts:
                        print(f"    Body parts: {', '.join(obj.pose.key_body_parts)}")
                    if obj.pose.gaze_direction or obj.pose.expressive_face:
                        print(f"    Gaze: {obj.pose.gaze_direction}, Expression: {obj.pose.expressive_face}")

            print("\n=== ACTIONS ===")
            for action in actions_block.actions:
                deps = ", ".join(action.dependencies) if action.dependencies else ""
                print(f"\n  [{deps}] {action.action_class} / {action.stage_class}")
                print(f"    {action.description}")
                tc = action.temporal_context
                print(f"    Frame Position in Event: {tc.frame_position_in_event}, is_highlight_frame={tc.is_highlight_frame}")

            print("\n=== CINEMATOGRAPHY ===")
            cam = cinematography.camera
            print(f"  Camera:")
            print(f"    Shot: {cam.shot_size}, {cam.shot_framing}, {cam.camera_angle}")
            print(f"    Lens: {cam.lens_size}, DoF: {cam.depth_of_field}, Focus: {cam.focus}")
            print(f"    Movement: {cam.movement}")

            light = cinematography.lighting
            print(f"  Lighting:")
            print(f"    Conditions: {light.conditions}")
            print(f"    Direction: {light.direction}, Shadows: {light.shadows}")
            print(f"    Type: {light.lighting_type}, Mood: {light.mood_tag}")

            comp = cinematography.composition
            print(f"  Composition:")
            print(f"    {comp.description}")
            print(f"    Layout: {comp.subject_layout}")

            look = cinematography.look
            print(f"  Look:")
            print(f"    Medium: {look.style_medium}, Style: {look.artistic_style}")
            print(f"    Colors: {look.color_scheme}")
            print(f"    Mood: {look.mood_atmosphere}")
            print(f"    Scores: preference={look.preference_score}, aesthetic={look.aesthetic_score}")

        # Assemble full scene
        scene = Scene(
            elements=elements,
            objects=objects_block.objects,
            actions=actions_block.actions,
            cinematography=cinematography,
        )

        # -------- Stage 3: Critic + Summary (sequential) --------
        if verbose:
            print("\n[Stage 3] Running critic and summary (sequential)...")
        stage3_start = time.perf_counter()

        # Use Gemini Flash Lite for critic (faster, cheaper)
        critic_lm = dspy.LM(MODEL_CRITIC, cache=False)
        with dspy.settings.context(lm=critic_lm):
            critic_result = self.scene_critic(scene=scene)

        # Use Gemini 3 Pro for summary (best at respecting token limits)
        summary_lm = dspy.LM(MODEL_SUMMARY, max_tokens=512, cache=False)
        with dspy.settings.context(lm=summary_lm):
            summary_predictor = dspy.BestOfN(
                module=dspy.Predict(SceneSummary),
                N=3,
                reward_fn=summary_not_truncated,
                threshold=1.0,
            )
            summary_result = summary_predictor(scene=scene)

        timing.stage3_parallel_sec = time.perf_counter() - stage3_start
        timing.total_sec = time.perf_counter() - total_start

        # Get token usage from DSPy's built-in tracking
        summary_tokens = self._extract_output_tokens(summary_result)

        if verbose:
            print(f"[Stage 3] Complete ({timing.stage3_parallel_sec:.2f}s)")
            print("\n" + "=" * 60)
            print("=== TIMING SUMMARY ===")
            print(timing)
            print("=" * 60)

        return ScenePipelineOutput(
            scene=scene,
            critic_issues=critic_result.issues,
            critic_score=critic_result.consistency_score,
            short_description=summary_result.short_description,
            timing=timing,
            summary_tokens=summary_tokens,
        )

    async def aforward(
        self,
        prompt: str,
        image: dspy.Image | None = None,
        video: dspy.Image | None = None,
        audio: dspy.Audio | None = None,
        verbose: bool = True,
    ) -> ScenePipelineOutput:
        """Async version with true parallel LLM calls using DSPy's native async support.

        Uses Best-of-N approach: generates BEST_OF_N candidates in parallel,
        picks the one with highest critic score. Retries if best score < threshold.

        Args:
            prompt: Text description of the scene (can be empty if image/video provided)
            image: Optional reference image for visual analysis
            video: Optional reference video (as frames) for motion/dynamics analysis
            audio: Optional reference audio for mood/pacing cues
            verbose: Whether to print progress information
        """
        best_result: ScenePipelineOutput | None = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0 and verbose:
                print(f"\n{'=' * 60}")
                print(f"=== RETRY {attempt}/{MAX_RETRIES} (best critic score was {best_result.critic_score:.2f}) ===")
                print(f"{'=' * 60}")

            if verbose:
                print(f"\n[Best-of-{BEST_OF_N}] Generating {BEST_OF_N} candidates in parallel (async)...")

            # Generate BEST_OF_N candidates in parallel using asyncio.gather
            tasks = [
                self._aforward_once(prompt, image, video, audio, verbose=(verbose and i == 0))
                for i in range(BEST_OF_N)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and collect successful candidates
            candidates: list[ScenePipelineOutput] = []
            errors: list[Exception] = []
            for r in results:
                if isinstance(r, Exception):
                    errors.append(r)
                    if verbose:
                        print(f"[Best-of-{BEST_OF_N}] Candidate failed: {r}")
                else:
                    candidates.append(r)

            if not candidates:
                # Include the first error in the message for debugging
                first_error = errors[0] if errors else "Unknown error"
                raise RuntimeError(f"All candidates failed to generate. First error: {first_error}")

            # Pick the candidate with highest critic score
            best_candidate = max(candidates, key=lambda x: x.critic_score)

            if verbose:
                scores = [f"{c.critic_score:.2f}" for c in candidates]
                print(f"\n[Best-of-{BEST_OF_N}] Candidate scores: {', '.join(scores)}")
                print(f"[Best-of-{BEST_OF_N}] Selected best: {best_candidate.critic_score:.2f}")

            # Track overall best result across retries
            if best_result is None or best_candidate.critic_score > best_result.critic_score:
                best_result = best_candidate

            # If critic score meets threshold, return immediately
            if best_candidate.critic_score >= CRITIC_THRESHOLD:
                if verbose and attempt > 0:
                    print(f"\n[Retry] Success! Critic score {best_candidate.critic_score:.2f} >= {CRITIC_THRESHOLD}")
                best_candidate.retry_count = attempt
                return best_candidate

            if verbose:
                print(f"\n[Critic] Best score {best_candidate.critic_score:.2f} < {CRITIC_THRESHOLD} threshold")
                if attempt < MAX_RETRIES:
                    print(f"[Critic] Retrying with another Best-of-{BEST_OF_N}...")

        # Return best result after all retries
        if verbose:
            print(f"\n[Retry] Max retries reached. Returning best result (score: {best_result.critic_score:.2f})")
        best_result.retry_count = MAX_RETRIES
        return best_result

    async def _aforward_once(
        self,
        prompt: str,
        image: dspy.Image | None = None,
        video: dspy.Image | None = None,
        audio: dspy.Audio | None = None,
        verbose: bool = True,
    ) -> ScenePipelineOutput:
        """Single async execution of the pipeline (no retry)."""
        timing = StageTiming()
        total_start = time.perf_counter()

        if verbose:
            print("=" * 60)
            print("=== INPUT PROMPT ===")
            print(prompt)
            if image:
                print("=== IMAGE PROVIDED ===")
            if video:
                print("=== VIDEO PROVIDED ===")
            if audio:
                print("=== AUDIO PROVIDED ===")
            print("=" * 60)

        # -------- Stage 1: Prompt -> Elements (sequential) --------
        if verbose:
            print("\n[Stage 1] Extracting elements...")
        stage1_start = time.perf_counter()

        # Use multimodal model if image/video/audio provided (only Gemini supports audio)
        has_multimodal = image is not None or video is not None or audio is not None
        multimodal_lm = dspy.LM(MODEL_MULTIMODAL, cache=True) if has_multimodal else None

        if multimodal_lm:
            with dspy.settings.context(lm=multimodal_lm):
                elements_result = await self.prompt_to_elements.acall(prompt=prompt, image=image, video=video, audio=audio)
        else:
            elements_result = await self.prompt_to_elements.acall(prompt=prompt, image=image, video=video, audio=audio)
        elements: Elements = elements_result.elements

        timing.stage1_elements_sec = time.perf_counter() - stage1_start

        if verbose:
            print(f"[Stage 1] Complete ({timing.stage1_elements_sec:.2f}s)")
            print("\n=== ELEMENTS ===")
            for element in elements.elements:
                print(f"  - {element.element_id} ({element.role}): {element.entity_type}")

        # -------- Stage 2: Prompt + Elements -> three heads (parallel) --------
        # Using DSPy's native acall() for true async HTTP requests via litellm.acompletion
        # This allows concurrent LLM API calls without blocking threads.
        # Total time ≈ max(objects, actions, cinematography) with better resource utilization.
        # Use multimodal model if audio is present (only Gemini supports audio input)
        if verbose:
            print("\n[Stage 2] Generating objects, actions, cinematography (parallel async)...")
        stage2_start = time.perf_counter()

        # True async parallel calls - no thread pool needed
        if multimodal_lm and audio is not None:
            # Use multimodal model for audio support
            with dspy.settings.context(lm=multimodal_lm):
                objects_task = self.scene_objects.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
                actions_task = self.scene_actions.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
                cinematography_task = self.scene_cinematography.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)

                objects_result, actions_result, cin_result = await asyncio.gather(
                    objects_task, actions_task, cinematography_task
                )
        else:
            objects_task = self.scene_objects.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
            actions_task = self.scene_actions.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)
            cinematography_task = self.scene_cinematography.acall(prompt=prompt, image=image, video=video, audio=audio, elements=elements)

            objects_result, actions_result, cin_result = await asyncio.gather(
                objects_task, actions_task, cinematography_task
            )

        objects_block: ObjectsBlock = objects_result.objects
        actions_block: ActionsBlock = actions_result.actions
        cinematography: Cinematography = cin_result.cinematography

        timing.stage2_parallel_sec = time.perf_counter() - stage2_start

        if verbose:
            print(f"[Stage 2] Complete ({timing.stage2_parallel_sec:.2f}s)")

            print("\n=== OBJECTS ===")
            for obj in objects_block.objects:
                deps = ", ".join(obj.dependencies) if obj.dependencies else ""
                print(f"\n  [{deps}] {obj.category} (relative_size: {obj.relative_size}, location: {obj.location})")
                print(f"    Description: {obj.description}")
                print(f"    Shape/Color: {obj.shape_and_color}")
                print(f"    Texture: {obj.texture}")
                print(f"    Appearance Details: {obj.appearance_details}")
                if obj.pose:
                    print(f"    Pose: {obj.pose.pose_class}, {obj.pose.body_orientation}")
                    if obj.pose.key_body_parts:
                        print(f"    Body parts: {', '.join(obj.pose.key_body_parts)}")
                    if obj.pose.gaze_direction or obj.pose.expressive_face:
                        print(f"    Gaze: {obj.pose.gaze_direction}, Expression: {obj.pose.expressive_face}")

            print("\n=== ACTIONS ===")
            for action in actions_block.actions:
                deps = ", ".join(action.dependencies) if action.dependencies else ""
                print(f"\n  [{deps}] {action.action_class} / {action.stage_class}")
                print(f"    {action.description}")
                tc = action.temporal_context
                print(f"    Frame Position in Event: {tc.frame_position_in_event}, is_highlight_frame={tc.is_highlight_frame}")

            print("\n=== CINEMATOGRAPHY ===")
            cam = cinematography.camera
            print(f"  Camera:")
            print(f"    Shot: {cam.shot_size}, {cam.shot_framing}, {cam.camera_angle}")
            print(f"    Lens: {cam.lens_size}, DoF: {cam.depth_of_field}, Focus: {cam.focus}")
            print(f"    Movement: {cam.movement}")

            light = cinematography.lighting
            print(f"  Lighting:")
            print(f"    Conditions: {light.conditions}")
            print(f"    Direction: {light.direction}, Shadows: {light.shadows}")
            print(f"    Type: {light.lighting_type}, Mood: {light.mood_tag}")

            comp = cinematography.composition
            print(f"  Composition:")
            print(f"    {comp.description}")
            print(f"    Layout: {comp.subject_layout}")

            look = cinematography.look
            print(f"  Look:")
            print(f"    Medium: {look.style_medium}, Style: {look.artistic_style}")
            print(f"    Colors: {look.color_scheme}")
            print(f"    Mood: {look.mood_atmosphere}")
            print(f"    Scores: preference={look.preference_score}, aesthetic={look.aesthetic_score}")

        # Assemble full scene
        scene = Scene(
            elements=elements,
            objects=objects_block.objects,
            actions=actions_block.actions,
            cinematography=cinematography,
        )

        # -------- Stage 3: Critic + Summary (parallel async) --------
        if verbose:
            print("\n[Stage 3] Running critic and summary (parallel async)...")
        stage3_start = time.perf_counter()

        # Use Gemini Flash Lite for critic (faster, cheaper)
        critic_lm = dspy.LM(MODEL_CRITIC, cache=False)
        with dspy.settings.context(lm=critic_lm):
            critic_result = await self.scene_critic.acall(scene=scene)

        # Use Gemini 3 Pro for summary (best at respecting token limits)
        summary_lm = dspy.LM(MODEL_SUMMARY, max_tokens=512, cache=False)
        with dspy.settings.context(lm=summary_lm):
            summary_predictor = dspy.BestOfN(
                module=dspy.Predict(SceneSummary),
                N=3,
                reward_fn=summary_not_truncated,
                threshold=1.0,
            )
            summary_result = await asyncio.to_thread(summary_predictor, scene=scene)

        timing.stage3_parallel_sec = time.perf_counter() - stage3_start
        timing.total_sec = time.perf_counter() - total_start

        # Get token usage from DSPy's built-in tracking
        summary_tokens = self._extract_output_tokens(summary_result)

        if verbose:
            print(f"[Stage 3] Complete ({timing.stage3_parallel_sec:.2f}s)")
            print("\n" + "=" * 60)
            print("=== TIMING SUMMARY ===")
            print(timing)
            print("=" * 60)

        return ScenePipelineOutput(
            scene=scene,
            critic_issues=critic_result.issues,
            critic_score=critic_result.consistency_score,
            short_description=summary_result.short_description,
            timing=timing,
            summary_tokens=summary_tokens,
        )


def configure_lm(
    model: str = MODEL_DEFAULT,
    cache: bool = False,
    track_usage: bool = True,
) -> dspy.LM:
    """Configure the DSPy language model.

    Default model is Gemini Flash for fast text-only operations.
    Multimodal inputs (image/video/audio) automatically use Gemini 3 Pro.
    """
    lm = dspy.LM(model, cache=cache)
    dspy.configure(lm=lm, track_usage=track_usage)
    return lm


# Example usage
if __name__ == "__main__":
    # Configure LM - defaults to Gemini 3 Pro Preview for multimodal support
    configure_lm()  # Uses default: gemini/gemini-3-pro-preview

    # Instantiate the pipeline module
    pipeline = SceneGenerationPipeline()

    async def main():
        while True:
            print("\n" + "=" * 60)
            prompt = input("Enter prompt (or 'quit' to exit): ").strip()

            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not prompt:
                print("Please enter a valid prompt.")
                continue

            # Example: multimodal input
            # image = dspy.Image("path/to/image.jpg")
            # video = dspy.Image("path/to/video.mp4")  # Treated as frames
            # audio = dspy.Audio("path/to/audio.mp3")
            # result = await pipeline.aforward(prompt, image=image, video=video, audio=audio)

            result = await pipeline.aforward(prompt, verbose=True)

            # Final outputs
            token_info = f"{result.summary_tokens} tokens" if result.summary_tokens > 0 else "tokens unavailable"
            print(f"\n=== SHORT DESCRIPTION ({token_info}) ===")
            print(result.short_description)
            print(f"\n=== CRITIC SCORE === {result.critic_score}")
            print("=== ISSUES ===")
            for issue in result.critic_issues:
                print(f"  - {issue}")

    asyncio.run(main())
