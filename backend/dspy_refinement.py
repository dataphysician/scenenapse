"""
Scene Refinement Pipeline - Modify existing scenes with natural language instructions.

Routes refinements based on scope:
  - Single head affected (non-elements) → targeted head refinement (cheaper, precise)
  - ELEMENTS affected OR multiple heads → all-in-one scene refinement (coherent)

CRITICAL CASCADE RULE:
  If ELEMENTS change, ALL dependent heads (OBJECTS, ACTIONS) MUST be regenerated.
  This ensures consistency: woman→man requires new object descriptions and poses.

Examples:
  - "Make the dress blue" → OBJECTS only → RefineObjects
  - "Have her run instead of walk" → ACTIONS only → RefineActions
  - "Make the woman into a man" → ELEMENTS changed → RefineScene (all heads)
  - "Replace the walking woman with a man racing" → RefineScene (all heads)
"""

from enum import Flag, auto

import dspy
from pydantic import BaseModel, Field, ConfigDict

from .dspy_signatures import (
    Scene,
    Elements,
    ObjectsBlock,
    ActionsBlock,
    Cinematography,
    SceneSummary,
    SceneCritic,
)
from .dspy_pipeline import (
    MODEL_CRITIC,
    MODEL_SUMMARY,
    summary_not_truncated,
    CRITIC_THRESHOLD,
    MAX_RETRIES,
    BEST_OF_N,
)


# ============================================================
# Affected Heads Flag
# ============================================================

class AffectedHeads(Flag):
    """Bitflag for affected scene components."""
    NONE = 0
    ELEMENTS = auto()       # 1 - who/what is in scene
    OBJECTS = auto()        # 2 - visual appearance
    ACTIONS = auto()        # 4 - motion/behavior
    CINEMATOGRAPHY = auto() # 8 - camera/lighting/mood


# ============================================================
# Classification Signature
# ============================================================

class ClassifyRefinement(dspy.Signature):
    """
    Identify which scene head is affected by the refinement instruction.

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  CRITICAL: PREFER SINGLE-HEAD REFINEMENTS                                    ║
    ║                                                                              ║
    ║  Most refinements affect ONLY ONE head. Multi-head is RARE.                 ║
    ║  When in doubt, pick the ONE most relevant head and set others to False.    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    HEAD DEFINITIONS (pick ONE unless absolutely necessary):
      ELEMENTS = ONLY when adding/removing/replacing WHO or WHAT exists
                 (add a dog, remove the car, change person to animal)
      OBJECTS = visual appearance changes (colors, clothing, features, pose, materials)
      ACTIONS = motion/behavior changes (walk→run, add gesture, change speed)
      CINEMATOGRAPHY = camera/lighting/mood (angle, time of day, atmosphere, style)

    SINGLE-HEAD EXAMPLES (the common case - 90% of refinements):
      "Make the dress blue" → OBJECTS only (color change)
      "Add glasses" → OBJECTS only (appearance detail)
      "Change to running pose" → OBJECTS only (pose is in Objects)
      "Have them run instead of walk" → ACTIONS only (motion change)
      "Speed up the movement" → ACTIONS only (action intensity)
      "Make it night time" → CINEMATOGRAPHY only (lighting)
      "Close-up shot" → CINEMATOGRAPHY only (camera)
      "More dramatic lighting" → CINEMATOGRAPHY only (lighting mood)
      "Change woman to man" → ELEMENTS only (entity_type change, objects use neutral terms)
      "Make the subject older" → ELEMENTS only (entity_type change)
      "Add a dog" → ELEMENTS only (new entity, objects will be generated separately)

    MULTI-HEAD EXAMPLES (RARE - only for complete rewrites):
      "Replace everything with an underwater scene" → ALL heads (complete rewrite)
      "Change to a noir detective scene" → multiple heads (genre change)

    DEFAULT TO SINGLE HEAD. Only mark multiple heads True if the instruction
    EXPLICITLY requires changes to multiple distinct aspects.
    """

    instruction: str = dspy.InputField(desc="Refinement instruction from user.")

    affects_elements: bool = dspy.OutputField(
        desc=(
            "True ONLY if adding/removing/replacing entities (who/what EXISTS changes). "
            "False for appearance, motion, or camera changes. "
            "Changing gender/age/species = ELEMENTS. Changing clothes/pose = NOT ELEMENTS."
        )
    )
    affects_objects: bool = dspy.OutputField(
        desc=(
            "True ONLY if changing visual appearance (clothing, colors, features, pose, materials). "
            "False for entity changes, motion changes, or camera changes. "
            "Pose changes go here, NOT in Actions."
        )
    )
    affects_actions: bool = dspy.OutputField(
        desc=(
            "True ONLY if changing motion/behavior (walk→run, add gesture, change activity). "
            "False for appearance changes or camera changes. "
            "Pose changes go in OBJECTS, not here."
        )
    )
    affects_cinematography: bool = dspy.OutputField(
        desc=(
            "True ONLY if changing camera/lighting/mood/style. "
            "False for entity, appearance, or motion changes."
        )
    )


# ============================================================
# Targeted Refinement Signatures (Single Head - Truncated Input)
# ============================================================
# These signatures receive ONLY the relevant head data, not the full scene.
# This reduces input tokens by 40-70% for spot edits.

class RefineElements(dspy.Signature):
    """
    Modify scene elements (the scene ontology) based on refinement instruction.

    ELEMENTS LANE - What this head owns:
      - element_id: stable identifier (e.g., 'main_subject', 'background_env')
      - role: narrative function (e.g., 'protagonist', 'environment')
      - entity_type: semantic description (e.g., 'adult human woman' → 'tall athletic man')
      - importance: relative prominence (primary/secondary/background)
      - rough_description: ROLE/TYPE only, NO appearance details

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  rough_description must NOT contain appearance details (those are in Objects)║
    ║                                                                              ║
    ║  GOOD: 'walking figure', 'garden path', 'handbag held by subject'           ║
    ║  BAD: 'woman in red dress', 'man with wavy hair' (clothing/hair = Objects)  ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    STAY IN YOUR LANE - Do NOT:
      - Add clothing, colors, materials, textures (that's OBJECTS)
      - Add camera/lighting/stylistic details (that's CINEMATOGRAPHY)
      - Describe motion/actions (that's ACTIONS)

    Examples:
      - "Make the woman into a tall athletic man" → update entity_type, rough_description='walking figure'
      - "Add a dog to the scene" → add new element with element_id='companion_animal'
      - "Remove the background people" → remove those elements

    PRESERVE: Elements not mentioned in the instruction stay EXACTLY as they are.
    """
    elements: Elements = dspy.InputField(
        desc="Current elements (scene ontology) to refine. Each has element_id, role, entity_type, importance, rough_description."
    )
    instruction: str = dspy.InputField(
        desc="What to change about entities. E.g., 'make the woman a man', 'add a dog', 'remove the car'."
    )

    refined_elements: Elements = dspy.OutputField(
        desc=(
            "Updated elements. Modify entity_type/role as instructed. "
            "rough_description must be ROLE/TYPE only (e.g., 'walking figure'), NOT appearance details. "
            "Keep element_ids stable when possible (changing woman→man keeps element_id='main_subject'). "
            "PRESERVE unmentioned elements exactly as they were."
        )
    )


class RefineObjects(dspy.Signature):
    """
    Modify object visual appearances based on refinement instruction.

    Write from a NEUTRAL THIRD-PERSON OBSERVER perspective.

    OBJECTS LANE - What this head owns:
      - description: rich visual appearance (clothing, features, materials)
      - shape_and_color: silhouette, color palette, contrasts
      - texture: material quality, surface feel, how light interacts
      - appearance_details: fine-grained unique visual details
      - pose: for humans/animals (pose_class, body_orientation, key_body_parts, gaze_direction, expressive_face)
      - location, relative_size, orientation in frame

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  FORBIDDEN (gendered terms):                                                 ║
    ║  woman, man, girl, boy, lady, gentleman, she, he, her, his, female, male    ║
    ║                                                                              ║
    ║  ALLOWED (neutral terms):                                                    ║
    ║  the subject, the figure, the individual, the person, [element_id]          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    WRONG: "The woman wears a red dress..." / "She has long hair..."
    CORRECT: "The subject wears a red dress..." / "The figure has long hair..."

    STAY IN YOUR LANE - Do NOT:
      - Change WHO/WHAT exists (that's ELEMENTS)
      - Describe motion/actions (that's ACTIONS)
      - Describe camera angles, lighting, mood (that's CINEMATOGRAPHY)
      - Include fragrance, sound, or mood statements (VISUAL ONLY)

    Examples:
      - "Make the dress blue" → update description, shape_and_color for that object
      - "Add glasses" → add appearance_details
      - "Change to a running pose" → update pose fields

    PRESERVE: Objects not mentioned in the instruction stay EXACTLY as they are.
    """
    elements: Elements = dspy.InputField(
        desc="Element layer for [element_id] references. Objects.dependencies must reference these."
    )
    objects: ObjectsBlock = dspy.InputField(
        desc="Current objects with visual descriptions, pose, appearance details to refine."
    )
    instruction: str = dspy.InputField(
        desc="What visual appearance to change. E.g., 'make the dress blue', 'add glasses', 'change to running pose'."
    )

    refined_objects: ObjectsBlock = dspy.OutputField(
        desc=(
            "Updated objects using NEUTRAL THIRD-PERSON perspective. "
            "FORBIDDEN: woman, man, she, he, her, his (gendered). "
            "ALLOWED: the subject, the figure, the person, [element_id]. "
            "Modify ONLY visual appearance as instructed. "
            "PRESERVE unmentioned objects exactly. Keep dependencies valid."
        )
    )


class RefineActions(dspy.Signature):
    """
    Modify actions/motion based on refinement instruction.

    ACTIONS LANE - What this head owns:
      - action_class: type of action (walking, running, falling, holding, reaching)
      - stage_class: phase/stage (ongoing_casual, take_off, peak, mid_stride)
      - description: physical motion detail using [element_id] references
      - temporal_context: is_highlight_frame, frame_position_in_event

    CRITICAL - USE [element_id] REFERENCES:
      WRONG: "She walks gracefully..."
      CORRECT: "[main_subject] walks gracefully..."
      NO gendered pronouns (he/she/his/her) - use [element_id] instead.

    STAY IN YOUR LANE - Do NOT:
      - Change WHO/WHAT exists (that's ELEMENTS)
      - Change visual appearance (that's OBJECTS)
      - Describe camera/lighting (that's CINEMATOGRAPHY)
      - Include mood/atmosphere statements (describe PHYSICAL MOTION only)

    Examples:
      - "Make her run instead of walk" → update action_class, stage_class, description
      - "Add a wave gesture" → add new action with dependencies
      - "Speed up the movement" → update stage_class and description

    PRESERVE: Actions not mentioned in the instruction stay EXACTLY as they are.
    """
    elements: Elements = dspy.InputField(
        desc="Element layer for [element_id] references. Action.dependencies must reference these."
    )
    actions: ActionsBlock = dspy.InputField(
        desc="Current actions with motion descriptions and temporal context to refine."
    )
    instruction: str = dspy.InputField(
        desc="What motion to change. E.g., 'make her run', 'add a wave gesture', 'slow down the movement'."
    )

    refined_actions: ActionsBlock = dspy.OutputField(
        desc=(
            "Updated actions. Use [element_id] references in descriptions (NOT pronouns). "
            "Modify ONLY motion/behavior as instructed. "
            "PRESERVE unmentioned actions exactly. Keep dependencies valid."
        )
    )


class RefineCinematography(dspy.Signature):
    """
    Modify cinematography based on refinement instruction.

    CINEMATOGRAPHY LANE - What this head owns:
      CAMERA: shot_size, shot_framing, camera_angle, lens_size, movement, depth_of_field, focus
      LIGHTING: conditions, direction, shadows, lighting_type, mood_tag
      COMPOSITION: description, subject_layout
      LOOK: style_medium, artistic_style, color_scheme, mood_atmosphere, preference_score, aesthetic_score

    STAY IN YOUR LANE - Do NOT:
      - Change WHO/WHAT exists (that's ELEMENTS)
      - Change visual appearance of entities (that's OBJECTS)
      - Change motion/actions (that's ACTIONS)
      - This head owns HOW the scene is filmed, not WHAT is in it

    Examples:
      - "Make it night time" → update lighting.conditions, lighting.lighting_type, lighting.mood_tag
      - "Close-up shot" → update camera.shot_size
      - "More dramatic" → update lighting.shadows, look.mood_atmosphere
      - "Dutch angle" → update camera.camera_angle

    PRESERVE: Settings not mentioned in the instruction stay EXACTLY as they are.
    """
    elements: Elements = dspy.InputField(
        desc="Element layer for dependency references in cinematography."
    )
    cinematography: Cinematography = dspy.InputField(
        desc="Current cinematography (camera, lighting, composition, look) to refine."
    )
    instruction: str = dspy.InputField(
        desc="What film craft to change. E.g., 'make it night', 'close-up shot', 'more dramatic lighting'."
    )

    refined_cinematography: Cinematography = dspy.OutputField(
        desc=(
            "Updated cinematography. Modify ONLY camera/lighting/composition/look as instructed. "
            "PRESERVE unmentioned settings exactly. Keep dependencies valid."
        )
    )


# ============================================================
# All-in-One Refinement Signature (Multiple Heads - RARE)
# ============================================================

class RefineScene(dspy.Signature):
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  WARNING: This is for RARE multi-head refinements only!                      ║
    ║                                                                              ║
    ║  Most refinements should use targeted single-head signatures:               ║
    ║  - RefineElements, RefineObjects, RefineActions, RefineCinematography       ║
    ║                                                                              ║
    ║  Only use RefineScene for COMPLETE SCENE REWRITES like:                     ║
    ║  - "Change everything to an underwater scene"                               ║
    ║  - "Make this a noir detective film"                                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    === HEAD LANES (each head stays in its lane) ===

    ELEMENTS LANE:
      - element_id, role, entity_type, importance, rough_description
      - WHO/WHAT exists in the scene (not how it looks or moves)

    OBJECTS LANE (NEUTRAL THIRD-PERSON):
      - description, shape_and_color, texture, appearance_details
      - FORBIDDEN: woman, man, she, he, her, his (gendered terms)
      - ALLOWED: the subject, the figure, the person, [element_id]
      - pose (for humans/animals): pose_class, body_orientation, key_body_parts

    ACTIONS LANE:
      - action_class, stage_class, description using [element_id], temporal_context
      - PHYSICAL MOTION only (no mood/atmosphere)

    CINEMATOGRAPHY LANE:
      - camera, lighting, composition, look
      - This is where mood/atmosphere belongs

    === CRITICAL RULES ===

    1. PRESERVE what is NOT mentioned in the instruction
    2. Use neutral third-person terms in Objects (the subject, the figure, NOT woman/man)
    3. Ensure cross-head consistency (poses match actions)
    """

    scene: Scene = dspy.InputField(
        desc="Current complete scene (elements, objects, actions, cinematography) to refine."
    )
    instruction: str = dspy.InputField(
        desc="Multi-head refinement instruction. E.g., 'make it a noir scene', 'change to underwater'."
    )

    refined_scene: Scene = dspy.OutputField(
        desc=(
            "Complete refined scene with affected heads updated coherently. "
            "Use [element_id] references in ALL descriptions (objects.description, actions.description). "
            "Ensure poses match actions. Ensure dependencies are valid. "
            "PRESERVE any head/field not mentioned in the instruction."
        )
    )


# ============================================================
# Smart Refinement Pipeline
# ============================================================

class RefinementResult(BaseModel):
    """Result from the refinement pipeline."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scene: Scene
    short_description: str | None = None
    critic_score: float = 0.0
    critic_issues: list[str] = Field(default_factory=list)
    affected_heads: list[str] = Field(default_factory=list)
    retry_count: int = 0  # Number of retries due to low critic score


class SmartRefinementPipeline(dspy.Module):
    """
    Route refinements based on scope:
      - Single head affected → targeted head refinement (cheaper, preserves other heads)
      - Multiple heads affected → all-in-one scene refinement (ensures consistency)

    IMPORTANT: Objects/Actions/Cinematography use [element_id] references (e.g., [main_subject]).
    When ELEMENTS change (woman→man), only the element's entity_type changes.
    The [element_id] references in other heads automatically point to the new entity.

    Usage:
        pipeline = SmartRefinementPipeline()
        result = pipeline(scene, "Make the dress blue")  # OBJECTS only
        result = pipeline(scene, "Make the woman into a tall athletic man")  # ELEMENTS only
    """

    def __init__(self) -> None:
        super().__init__()
        # Classifier
        self.classifier = dspy.Predict(ClassifyRefinement)

        # All-in-one (for multi-head changes)
        self.refine_scene = dspy.Predict(RefineScene)

        # Targeted refiners (for single-head changes)
        self.refine_elements = dspy.Predict(RefineElements)
        self.refine_objects = dspy.Predict(RefineObjects)
        self.refine_actions = dspy.Predict(RefineActions)
        self.refine_cinematography = dspy.Predict(RefineCinematography)

        # Critic validation
        self.critic = dspy.Predict(SceneCritic)

        # Summary regeneration with BestOfN
        self.summarize = dspy.BestOfN(
            module=dspy.Predict(SceneSummary),
            N=3,
            reward_fn=summary_not_truncated,
            threshold=1.0,
        )

    def _classify(self, instruction: str) -> AffectedHeads:
        """Classify instruction and return affected heads as flags."""
        result = self.classifier(instruction=instruction)

        flags = AffectedHeads.NONE
        if result.affects_elements:
            flags |= AffectedHeads.ELEMENTS
        if result.affects_objects:
            flags |= AffectedHeads.OBJECTS
        if result.affects_actions:
            flags |= AffectedHeads.ACTIONS
        if result.affects_cinematography:
            flags |= AffectedHeads.CINEMATOGRAPHY

        return flags

    def _count_heads(self, flags: AffectedHeads) -> int:
        """Count how many heads are affected."""
        count = 0
        for head in [AffectedHeads.ELEMENTS, AffectedHeads.OBJECTS,
                     AffectedHeads.ACTIONS, AffectedHeads.CINEMATOGRAPHY]:
            if head in flags:
                count += 1
        return count

    def _targeted_refine(
        self,
        scene: Scene,
        instruction: str,
        affected: AffectedHeads,
        verbose: bool,
    ) -> Scene:
        """
        Apply single-head refinement with TRUNCATED inputs, preserving other heads exactly.

        Objects/Actions/Cinematography use [element_id] references, so when ELEMENTS change,
        only the element's entity_type/description changes - other heads stay the same.
        """

        if AffectedHeads.ELEMENTS in affected:
            if verbose:
                print("  [Refining ELEMENTS] (input: elements only)")
            result = self.refine_elements(
                elements=scene.elements,
                instruction=instruction
            )
            return Scene(
                elements=result.refined_elements,
                objects=scene.objects,  # Unchanged - uses [element_id] references
                actions=scene.actions,  # Unchanged - uses [element_id] references
                cinematography=scene.cinematography,  # Unchanged
            )

        if AffectedHeads.OBJECTS in affected:
            if verbose:
                print("  [Refining OBJECTS] (input: elements + objects)")
            result = self.refine_objects(
                elements=scene.elements,
                objects=ObjectsBlock(objects=scene.objects),
                instruction=instruction
            )
            return Scene(
                elements=scene.elements,
                objects=result.refined_objects.objects,
                actions=scene.actions,
                cinematography=scene.cinematography,
            )

        if AffectedHeads.ACTIONS in affected:
            if verbose:
                print("  [Refining ACTIONS] (input: elements + actions)")
            result = self.refine_actions(
                elements=scene.elements,
                actions=ActionsBlock(actions=scene.actions),
                instruction=instruction
            )
            return Scene(
                elements=scene.elements,
                objects=scene.objects,
                actions=result.refined_actions.actions,
                cinematography=scene.cinematography,
            )

        if AffectedHeads.CINEMATOGRAPHY in affected:
            if verbose:
                print("  [Refining CINEMATOGRAPHY] (input: elements + cinematography)")
            result = self.refine_cinematography(
                elements=scene.elements,
                cinematography=scene.cinematography,
                instruction=instruction
            )
            return Scene(
                elements=scene.elements,
                objects=scene.objects,
                actions=scene.actions,
                cinematography=result.refined_cinematography,
            )

        return scene

    def _get_affected_list(self, affected: AffectedHeads) -> list[str]:
        """Convert flags to list of head names."""
        heads = []
        if AffectedHeads.ELEMENTS in affected:
            heads.append("ELEMENTS")
        if AffectedHeads.OBJECTS in affected:
            heads.append("OBJECTS")
        if AffectedHeads.ACTIONS in affected:
            heads.append("ACTIONS")
        if AffectedHeads.CINEMATOGRAPHY in affected:
            heads.append("CINEMATOGRAPHY")
        return heads

    def forward(
        self,
        scene: Scene,
        instruction: str,
        regenerate_summary: bool = True,
        verbose: bool = False,
    ) -> RefinementResult:
        """
        Refine a scene based on natural language instruction.

        Uses Best-of-N approach: generates BEST_OF_N candidates in parallel,
        picks the one with highest critic score. Retries if best score < threshold.

        Args:
            scene: Existing scene to modify
            instruction: Natural language refinement instruction
            regenerate_summary: Whether to regenerate summary after refinement
            verbose: Print routing and progress information

        Returns:
            RefinementResult with scene, critic validation, and summary
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        best_result: RefinementResult | None = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0 and verbose:
                print(f"\n{'=' * 60}")
                print(f"=== RETRY {attempt}/{MAX_RETRIES} (best critic score was {best_result.critic_score:.2f}) ===")
                print(f"{'=' * 60}")

            if verbose:
                print(f"\n[Best-of-{BEST_OF_N}] Generating {BEST_OF_N} refinement candidates in parallel...")

            # Generate BEST_OF_N candidates in parallel
            candidates: list[RefinementResult] = []
            with ThreadPoolExecutor(max_workers=BEST_OF_N) as executor:
                futures = [
                    executor.submit(
                        self._forward_once,
                        scene, instruction, regenerate_summary,
                        verbose=(verbose and i == 0)
                    )
                    for i in range(BEST_OF_N)
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        candidates.append(result)
                    except Exception as e:
                        if verbose:
                            print(f"[Best-of-{BEST_OF_N}] Candidate failed: {e}")

            if not candidates:
                raise RuntimeError("All refinement candidates failed to generate")

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
        scene: Scene,
        instruction: str,
        regenerate_summary: bool = True,
        verbose: bool = False,
    ) -> RefinementResult:
        """Single execution of the refinement pipeline (no retry)."""
        # Store affected heads for tracking
        self._last_affected_heads = AffectedHeads.NONE

        # Step 1: Classify affected heads
        affected = self._classify(instruction)
        head_count = self._count_heads(affected)
        self._last_affected_heads = affected

        if verbose:
            print(f"\n{'='*60}")
            print(f"[Classifier] Instruction: '{instruction}'")
            print(f"[Classifier] Affected: {affected.name if affected else 'NONE'} (count: {head_count})")

        # Step 2: Route based on head count
        # Single head → targeted refinement (cheaper, precise)
        # Multiple heads → all-in-one RefineScene (ensures consistency)
        #
        # NOTE: Objects/Actions/Cinematography use [element_id] references.
        # When ELEMENTS change, only the element definition changes - other heads stay the same.

        if head_count == 0:
            if verbose:
                print("[Router] No heads affected → returning unchanged")
            refined = scene

        elif head_count == 1:
            # Single head → targeted refinement
            if verbose:
                print(f"[Router] Single head → targeted refinement")
            refined = self._targeted_refine(scene, instruction, affected, verbose)

        else:
            # Multiple heads → all-in-one refinement for consistency
            if verbose:
                print(f"[Router] Multiple heads ({head_count}) → all-in-one RefineScene")
            result = self.refine_scene(scene=scene, instruction=instruction)
            refined = result.refined_scene

        # Step 3: Critic validation
        if verbose:
            print("[Critic] Validating refined scene...")

        critic_lm = dspy.LM(MODEL_CRITIC, cache=False)
        with dspy.settings.context(lm=critic_lm):
            critic_result = self.critic(scene=refined)

        critic_score = critic_result.consistency_score
        critic_issues = critic_result.issues

        if verbose:
            print(f"[Critic] Score: {critic_score:.2f}")
            if critic_issues:
                print(f"[Critic] Issues: {critic_issues}")

        # Step 4: Regenerate summary if requested (using MODEL_SUMMARY like generation pipeline)
        summary = None
        if regenerate_summary:
            if verbose:
                print("[Summary] Regenerating with BestOfN...")
            summary_lm = dspy.LM(MODEL_SUMMARY, max_tokens=512, cache=False)
            with dspy.settings.context(lm=summary_lm):
                summary = self.summarize(scene=refined).short_description

        if verbose:
            print(f"{'='*60}\n")

        return RefinementResult(
            scene=refined,
            short_description=summary,
            critic_score=critic_score,
            critic_issues=critic_issues,
            affected_heads=self._get_affected_list(affected),
        )


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    import asyncio
    from dspy_pipeline import SceneGenerationPipeline, MODEL_DEFAULT

    # Configure DSPy
    lm = dspy.LM(MODEL_DEFAULT)
    dspy.configure(lm=lm)

    # -------- Step 1: Generate initial scene --------
    print("=" * 70)
    print("STEP 1: Generate initial scene")
    print("=" * 70)

    gen_pipeline = SceneGenerationPipeline()
    initial_result = gen_pipeline.forward(
        prompt="Woman in a red dress walking through a garden",
        verbose=True
    )
    initial_scene = initial_result.scene

    print("\n[Initial Scene Summary]")
    print(f"Elements: {[e.element_id for e in initial_scene.elements.elements]}")
    print(f"Objects: {len(initial_scene.objects)}")
    print(f"Actions: {[a.action_class for a in initial_scene.actions]}")

    # -------- Step 2: Apply refinements --------
    print("\n" + "=" * 70)
    print("STEP 2: Apply refinements")
    print("=" * 70)

    refine_pipeline = SmartRefinementPipeline()

    # Test refinements with different scope
    refinements = [
        # Single head: OBJECTS only
        "Make the dress deep blue velvet",

        # Single head: CINEMATOGRAPHY only
        "Change to dramatic sunset lighting",

        # Multiple heads: ELEMENTS + OBJECTS + ACTIONS
        "Replace the walking woman with a man racing",
    ]

    current_scene = initial_scene
    for instruction in refinements:
        current_scene, summary = refine_pipeline.forward(
            scene=current_scene,
            instruction=instruction,
            verbose=True
        )

        print(f"[After refinement]")
        print(f"  Elements: {[e.element_id for e in current_scene.elements.elements]}")
        print(f"  Actions: {[a.action_class for a in current_scene.actions]}")
        if summary:
            print(f"  Summary: {summary[:150]}...")

    # -------- Final result --------
    print("\n" + "=" * 70)
    print("FINAL REFINED SCENE")
    print("=" * 70)
    print(f"Elements: {[e.element_id for e in current_scene.elements.elements]}")
    print(f"Entity types: {[e.entity_type for e in current_scene.elements.elements]}")
    print(f"Actions: {[a.action_class for a in current_scene.actions]}")
    print(f"Lighting: {current_scene.cinematography.lighting.conditions}")
    print(f"Mood: {current_scene.cinematography.look.mood_atmosphere}")
