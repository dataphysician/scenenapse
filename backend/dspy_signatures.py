"""
Scene Generation Signatures.

This module defines the Pydantic models and DSPy Signatures for the
multi-head (Objects, Actions, Cinematography) scene generation pipeline. It contains:

Pydantic Models (Data Schemas):
  - Element, Elements: Entity/role bindings for the scene
  - Pose: Animate entity pose description
  - SceneObject, ObjectsBlock: Visual appearance of entities
  - Action, TemporalContext, ActionsBlock: Dynamic actions and timing
  - CinematographyCamera, CinematographyLighting, CinematographyComposition,
    CinematographyLook, Cinematography: Film craft specifications
  - Scene: Complete scene combining all heads

DSPy Signatures:
  - PromptToElements: Extract entity elements from prompt
  - SceneObjects: Generate rich object descriptions
  - SceneActions: Generate action dynamics
  - SceneCinematography: Generate camera/lighting/composition
  - SceneCritic: Evaluate scene consistency
  - SceneSummary: Generate short scene description (Max 512 tokens)

For the pipeline implementation, see dspy_pipeline.py.
"""

from pydantic import BaseModel, Field, ConfigDict
import dspy


# ============================================================
# Core schema: Elements
# ============================================================

class Element(BaseModel):
    """Scene Cinematic Binding layer: who/what is in the scene at a coarse semantic level."""
    model_config = ConfigDict(extra="ignore")

    element_id: str = Field(
        description="Stable identifier for this element, e.g. 'main_subject', 'background_env', 'handbag'."
    )
    role: str = Field(
        description="Narrative role, e.g. 'protagonist', 'runner in background'."
    )
    entity_type: str = Field(
        description=(
            "Short and precise semantic description of the entity, e.g. "
            "'adult human woman', 'urban street environment', 'small leather handbag'."
        )
    )
    importance: str | None = Field(
        default=None,
        description="Relative importance, e.g. 'primary', 'secondary', 'background'."
    )
    rough_description: str | None = Field(
        default=None,
        description=(
            "Short phrase describing the entity's ROLE and TYPE only - NO appearance details. "
            "This field must NOT include clothing, colors, materials, or visual features (those belong in Objects). "
            "\n\n*** FORBIDDEN (appearance details → Objects head): ***"
            "\nclothing, dress, suit, shirt, hair color, eye color, materials, textures"
            "\n\n*** GOOD (entity role/type only): ***"
            "\n'walking figure', 'seated person', 'handbag held by subject', 'garden path', 'falling petals'"
            "\n\n*** BAD (leaks into Objects): ***"
            "\n'woman in red dress' (clothing), 'man with wavy red hair' (hair), 'vintage leather handbag' (material)"
        )
    )


class Elements(BaseModel):
    model_config = ConfigDict(extra="ignore")

    elements: list[Element] = Field(
        default_factory=list,
        description="All elements (roles) involved in this shot."
    )


# ============================================================
# Objects head: Aligned to Cinematic terms
# ============================================================

class Pose(BaseModel):
    """
    Cinematic-style pose axis for animate entities (humans, animals, humanoid, creatures, personified entities).
    ALL fields are REQUIRED when Pose is used - do not leave any as null.
    """
    model_config = ConfigDict(extra="ignore")

    pose_class: str = Field(
        description=(
            "REQUIRED: Discrete pose label describing the overall body position. "
            "E.g. 'walking_casual', 'running_sprint', 'standing_relaxed', 'seated_formal', "
            "'jumping_peak', 'dancing_twirl', 'crouching', 'lying_down'."
        )
    )
    body_orientation: str = Field(
        description=(
            "REQUIRED: Body orientation relative to camera. "
            "E.g. 'facing_camera', 'three-quarter_left', 'profile_right', 'back_to_camera', "
            "'three-quarter_right', 'slight_turn_left'."
        )
    )
    key_body_parts: list[str] = Field(
        default_factory=list,
        description=(
            "REQUIRED: List of key body-part configurations that define the pose. "
            "E.g. ['left_arm_bent_at_elbow', 'right_hand_holding_object', "
            "'legs_mid_stride', 'shoulders_relaxed', 'head_tilted_slightly']."
        )
    )
    gaze_direction: str = Field(
        description=(
            "REQUIRED: Where the subject is looking. "
            "E.g. 'towards_camera', 'looking_down', 'gazing_left', 'eyes_closed', "
            "'looking_up', 'distant_stare', 'focused_on_object'."
        )
    )
    expressive_face: str = Field(
        description=(
            "REQUIRED: Facial expression conveying emotion/mood. "
            "E.g. 'serene_smile', 'intense_concentration', 'joyful_laugh', "
            "'neutral_calm', 'wistful_longing', 'confident_smirk'."
        )
    )


class SceneObject(BaseModel):
    """
    Scene object enriched with cinematic pose axis.
    Scene object's identity is determined via element dependencies, not per-object IDs.
    VISUAL ONLY: describe what can be SEEN, not smelled/heard/felt
    NO fragrance, scent, aroma, sound, or mood/atmosphere statements
    """
    model_config = ConfigDict(extra="ignore")

    dependencies: list[str] = Field(
        min_length=1,
        description=(
            "REQUIRED: Element ids this object represents. MUST have at least one. "
            "E.g. ['main_subject'] for woman, ['background_env'] for garden."
        )
    )

    category: str = Field(
        description="High-level category, e.g. 'human', 'animal', 'object', 'environment'."
    )
    is_primary_subject: bool | None = Field(
        default=None,
        description="True if this is the primary subject in the frame."
    )

    # Cinematic-style appearance + placement (REQUIRED fields for rich output)
    # Use neutral third-person perspective - avoid gendered terms
    description: str = Field(
        description=(
            "REQUIRED: Rich, detailed description of the entity's visual appearance. "
            "Write from a NEUTRAL THIRD-PERSON OBSERVER perspective."
            "\n\n*** FORBIDDEN (gendered terms): ***"
            "\nwoman, man, girl, boy, lady, gentleman, she, he, her, his, female, male"
            "\n\n*** ALLOWED (neutral terms): ***"
            "\nthe subject, the figure, the individual, the person, the character, [element_id]"
            "\n\n*** CORRECT: ***"
            "\n'The subject wears a flowing red silk gown with delicate lace trim...'"
            "\n'[main_subject] has long auburn hair cascading down the shoulders...'"
            "\n'The figure stands gracefully, posture elegant and poised...'"
            "\n\n*** WRONG: ***"
            "\n'The woman wears a red dress...' (gendered)"
            "\n'She has long flowing hair...' (gendered pronoun)"
            "\n\nFor humans: clothing style, colors, fabrics, hair color/style, skin tone, accessories. "
            "For objects: materials, condition, distinctive features. "
            "For environments: vegetation, architectural elements. "
            "Must be at least 2-3 sentences with specific visual details."
        )
    )
    location: str = Field(
        default="center",
        description="Frame-relative location, e.g. 'center', 'left third', 'background right'."
    )
    relationship: str | None = Field(
        default=None,
        description="Relationship to other entities, e.g. 'Primary subject', 'held_by main_subject'."
    )
    relative_size: str = Field(
        default="medium",
        description="Relative size in the frame, e.g. 'large in frame', 'small in background', 'fills frame'."
    )
    shape_and_color: str = Field(
        description=(
            "REQUIRED: Shape silhouette and dominant colors. "
            "E.g. 'Tall, slender silhouette in deep crimson red with gold accents'."
        )
    )
    texture: str = Field(
        description=(
            "REQUIRED: Perceived texture and material quality. "
            "E.g. 'Smooth silk with subtle sheen', 'Rough weathered leather', 'Soft velvety petals'."
        )
    )
    appearance_details: str = Field(
        description=(
            "REQUIRED: Fine-grained visual details that make this entity unique. "
            "E.g. 'Intricate lace trim at the neckline, delicate pearl earrings, "
            "a few loose strands of hair framing the face'."
        )
    )
    orientation: str | None = Field(
        default=None,
        description="Orientation in the frame, e.g. 'Facing directly forward', 'Profile view to the right'."
    )

    pose: Pose | None = Field(
        default=None,
        description=(
            "Pose description - REQUIRED for animate entities (category='human' or 'animal'), "
            "must be None/omitted for inanimate entities (category='object', 'environment', "
            "'natural_element'). Cherry blossoms, handbags, gardens do NOT have poses."
        )
    )


class ObjectsBlock(BaseModel):
    """Output of the objects head: list of objects tied to elements via dependencies."""
    model_config = ConfigDict(extra="ignore")

    objects: list[SceneObject] = Field(
        default_factory=list,
        description="scene objects with cinematic pose axis."
    )


# ============================================================
# Actions head: Cinematic action axis
# ============================================================

class TemporalContext(BaseModel):
    """
    Temporal positioning of this frame within the action event.
    ALL fields are REQUIRED - always specify temporal context.
    """
    model_config = ConfigDict(extra="ignore")

    is_highlight_frame: bool = Field(
        description=(
            "REQUIRED: Is this frame a highlight/key moment of the action? "
            "True for peak moments (mid-jump, decisive gesture, emotional climax). "
            "False for transitional or ongoing moments."
        )
    )
    frame_position_in_event: str = Field(
        description=(
            "REQUIRED: Where this frame sits in the action's timeline. "
            "E.g. 'early' (action starting), 'peak' (climax/apex), 'late' (winding down), "
            "'ongoing' (continuous action), 'anticipation' (before action starts)."
        )
    )


class Action(BaseModel):
    """
    Motion-focused action entry. ALL fields are REQUIRED.
    dependencies: which element(s) participate in this action.
    Describe PHYSICAL MOTION only (no camera terms, no appearance details, NO MOOD/ATMOSPHERE)
    To describe the agent affected by the action, use [element_id] references instead of pronouns like he/she/her/it/they.
    """
    model_config = ConfigDict(extra="ignore")

    dependencies: list[str] = Field(
        min_length=1,
        description=(
            "REQUIRED: Element ids participating in this action. MUST have at least one. "
            "E.g. ['main_subject'] for woman walking, ['cherry_blossoms'] for petals falling."
        )
    )

    action_class: str = Field(
        description=(
            "REQUIRED: Type/category of action. "
            "E.g. 'walking', 'running', 'jumping', 'falling', 'sitting', 'dancing', "
            "'reaching', 'holding', 'looking', 'floating', 'swaying'."
        )
    )
    stage_class: str = Field(
        description=(
            "REQUIRED: Fine-grained phase/stage of the action. "
            "E.g. 'ongoing_casual' (continuous), 'take_off' (starting), 'peak' (apex), "
            "'landing' (ending), 'mid_stride', 'arm_extended', 'descent', 'at_rest'."
        )
    )
    description: str = Field(
        description=(
            "REQUIRED: Description using [element_id] references (1-2 sentences). "
            "Use [main_subject], [natural_element], etc. instead of pronouns (he/she/his/her). "
            "Example: '[main_subject] walks gracefully, dress swaying with each step.'"
        )
    )
    temporal_context: TemporalContext = Field(
        description="Context of where this frame lies in the event."
    )


class ActionsBlock(BaseModel):
    """Output of the actions head: list of actions tied to elements via dependencies."""
    model_config = ConfigDict(extra="ignore")

    actions: list[Action] = Field(
        default_factory=list,
        description="Cinematic-style action entries."
    )


# ============================================================
# Cinematography head: unified shot grammar + look
# ============================================================

class CinematographyCamera(BaseModel):
    """Camera setup - ALL fields REQUIRED."""
    model_config = ConfigDict(extra="ignore")

    shot_size: str = Field(
        description=(
            "REQUIRED: Shot size/scale. "
            "E.g. 'extreme_close_up', 'close_up', 'medium_close_up', 'medium', "
            "'medium_long', 'long', 'extreme_long'."
        )
    )
    shot_framing: str = Field(
        description=(
            "REQUIRED: How subjects are framed. "
            "E.g. 'centered_single', 'rule_of_thirds', 'two_shot', 'over_the_shoulder', "
            "'point_of_view', 'wide_establishing'."
        )
    )
    camera_angle: str = Field(
        description=(
            "REQUIRED: Camera angle relative to subject. "
            "E.g. 'eye_level', 'low_angle', 'high_angle', 'dutch_angle', "
            "'birds_eye', 'worms_eye'."
        )
    )
    lens_size: str = Field(
        description=(
            "REQUIRED: Focal length / lens type. "
            "E.g. '24mm_wide', '35mm', '50mm_standard', '85mm_portrait', "
            "'135mm_telephoto', '200mm_long_telephoto'."
        )
    )
    movement: str = Field(
        description=(
            "REQUIRED: Camera movement during shot. "
            "E.g. 'static', 'dolly_in', 'dolly_out', 'pan_left', 'pan_right', "
            "'tilt_up', 'tilt_down', 'tracking', 'handheld', 'crane_up', 'steadicam'."
        )
    )
    depth_of_field: str = Field(
        description=(
            "REQUIRED: Depth of field / bokeh level. "
            "E.g. 'very_shallow' (f/1.4), 'shallow' (f/2.8), 'moderate' (f/5.6), "
            "'deep' (f/11), 'very_deep' (f/16+)."
        )
    )
    focus: str = Field(
        description=(
            "REQUIRED: What is in sharp focus. "
            "E.g. 'sharp_on_eyes', 'sharp_on_hands', 'sharp_on_foreground_subject', "
            "'split_focus', 'soft_focus_overall', 'rack_focus_to_background'."
        )
    )


class CinematographyLighting(BaseModel):
    """Lighting design - ALL fields REQUIRED."""
    model_config = ConfigDict(extra="ignore")

    conditions: str = Field(
        description=(
            "REQUIRED: Overall lighting conditions. "
            "E.g. 'soft natural daylight', 'harsh midday sun', 'golden hour warmth', "
            "'cool moonlight', 'neon city lights', 'candlelit interior', 'overcast diffused'."
        )
    )
    direction: str = Field(
        description=(
            "REQUIRED: Primary light direction relative to subject. "
            "E.g. 'frontal', 'side_left_45deg', 'side_right_45deg', 'backlit', "
            "'overhead', 'under_lighting', 'rim_light_from_behind'."
        )
    )
    shadows: str = Field(
        description=(
            "REQUIRED: Shadow quality and intensity. "
            "E.g. 'soft_diffused', 'hard_defined', 'minimal_flat', 'deep_dramatic', "
            "'dappled_through_leaves', 'long_evening_shadows'."
        )
    )
    lighting_type: str = Field(
        description=(
            "REQUIRED: Normalized lighting type tag. "
            "E.g. 'natural_daylight', 'natural_moonlight', 'artificial_tungsten', "
            "'artificial_fluorescent', 'mixed_practical', 'studio_softbox', 'neon_colored'."
        )
    )
    mood_tag: str = Field(
        description=(
            "REQUIRED: Short mood/style tag for the lighting. "
            "E.g. 'high_key_bright', 'low_key_moody', 'chiaroscuro', 'film_noir', "
            "'romantic_soft', 'harsh_gritty', 'ethereal_glowing'."
        )
    )


class CinematographyComposition(BaseModel):
    """Composition/framing design - ALL fields REQUIRED."""
    model_config = ConfigDict(extra="ignore")

    description: str = Field(
        description=(
            "REQUIRED: Overall composition description. "
            "E.g. 'Balanced symmetrical frame with subject centered', "
            "'Dynamic diagonal leading lines drawing eye to subject', "
            "'Intimate tight framing isolating the face'."
        )
    )
    subject_layout: str = Field(
        description=(
            "REQUIRED: Spatial arrangement of subjects in frame. "
            "E.g. 'primary_subject_centered', 'subject_left_third_negative_space_right', "
            "'foreground_subject_background_environment', 'layered_depth_multiple_planes'."
        )
    )


class CinematographyLook(BaseModel):
    """Visual style and aesthetic - ALL fields REQUIRED."""
    model_config = ConfigDict(extra="ignore")

    style_medium: str = Field(
        description=(
            "REQUIRED: Visual medium/format. "
            "E.g. 'photograph', 'film_35mm', 'digital_cinema', 'oil_painting', "
            "'watercolor', '3d_render', 'anime_cel', 'pencil_sketch'."
        )
    )
    artistic_style: str = Field(
        description=(
            "REQUIRED: Artistic/visual style. "
            "E.g. 'photorealistic', 'hyperrealistic', 'impressionistic', 'noir', "
            "'anime', 'surrealist', 'minimalist', 'baroque', 'pop_art'."
        )
    )
    color_scheme: str = Field(
        description=(
            "REQUIRED: Dominant color palette description. "
            "E.g. 'warm_earth_tones', 'cool_blues_and_silvers', 'vibrant_saturated', "
            "'muted_desaturated', 'monochromatic_sepia', 'complementary_orange_teal'."
        )
    )
    mood_atmosphere: str = Field(
        description=(
            "REQUIRED: Emotional mood/atmosphere of the image. "
            "E.g. 'romantic_and_dreamy', 'tense_and_suspenseful', 'peaceful_serene', "
            "'energetic_dynamic', 'melancholic_wistful', 'mysterious_ethereal'."
        )
    )
    preference_score: str = Field(
        description=(
            "REQUIRED: Predicted viewer preference level. "
            "E.g. 'very_high', 'high', 'medium', 'low'. "
            "Based on composition quality, visual appeal, emotional impact."
        )
    )
    aesthetic_score: str = Field(
        description=(
            "REQUIRED: Technical aesthetic quality level. "
            "E.g. 'very_high', 'high', 'medium', 'low'. "
            "Based on lighting, color harmony, technical execution."
        )
    )


class Cinematography(BaseModel):
    """
    Unified shot grammar + look for a single scene.
    ALL sub-components are REQUIRED - camera, lighting, composition, and look.
    """
    model_config = ConfigDict(extra="ignore")

    dependencies: list[str] = Field(
        default_factory=list,
        description="REQUIRED: Element ids this shot is composed around, e.g. ['main_subject', 'background_env']."
    )

    camera: CinematographyCamera = Field(
        description="REQUIRED: Full camera setup specification."
    )
    lighting: CinematographyLighting = Field(
        description="REQUIRED: Full lighting design specification."
    )
    composition: CinematographyComposition = Field(
        description="REQUIRED: Full composition/framing specification."
    )
    look: CinematographyLook = Field(
        description="REQUIRED: Full visual style and aesthetic specification."
    )


# ============================================================
# DSPy Signatures (no Module classes yet)
# ============================================================

import dspy


# 1) Prompt -> Elements (Multimodal: text + optional image/video/audio)
class PromptToElements(dspy.Signature):
    """
    Infer a compact, role-centric element layer (the scene ontology) from multimodal inputs.

    INPUTS: Can be any combination of:
      - prompt (text): Natural language description of the desired shot/scene
      - image: Reference image to analyze for entities and composition
      - video: Reference video to analyze for entities, motion, and dynamics
      - audio: Reference audio to inform mood, pacing, or narrative elements

    The goal is to:
      - Identify a SMALL set of stable entities (elements) that the rest of the pipeline
        (objects, actions, cinematography, edits) can reference.
      - Assign each element:
          * element_id: a concise, stable identifier (e.g. 'main_subject', 'background_env',
            'accessory', 'natural_element').
          * role: narrative function (e.g. 'protagonist', 'environment', 'object carried').
          * entity_type: short semantic description (e.g. 'adult human woman',
            'moonlit garden', 'leather handbag').
          * importance: rough visual/narrative prominence (primary / secondary / background).
          * rough_description: ROLE/TYPE only, NO appearance details (e.g. 'walking figure',
            'garden path', 'handbag held by subject'). NOT 'woman in red dress' (clothing = Objects).

    STAY IN YOUR LANE - Elements defines WHO/WHAT exists:
      - Do NOT include clothing, colors, materials, textures (that's Objects)
      - Do NOT invent camera, lighting, or stylistic details (that's Cinematography)
      - Do NOT over-explain actions or motion (that's Actions)
      - rough_description must be ENTITY-ROLE focused, not appearance-focused

    Every element_id must be unique and simple; prefer snake_case IDs that can be
    reused across shots and edits. Prefer a small, coherent set of elements over
    many micro-entities; elements are edit handles and dependency anchors.
    """

    prompt: str = dspy.InputField(
        desc=(
            "User's natural-language description of the desired shot/scene. May mix "
            "content, mood, and light story context but typically under-specifies "
            "visual detail and film craft. Can be empty if image/video is provided."
        )
    )
    image: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference image to analyze for entities, appearance, and composition."
    )
    video: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference video (as frames) to analyze for entities, motion, and dynamics."
    )
    audio: dspy.Audio | None = dspy.InputField(
        default=None,
        desc="Optional reference audio to analyze for entities, and inform the mood, pacing, or narrative elements."
    )
    elements: Elements = dspy.OutputField(
        desc=(
            "Inferred element layer (roles): a compact ontology of entities in the scene. "
            "Each element_id must be unique and reusable; later heads will ONLY refer "
            "to these element_ids via dependencies."
        )
    )


# 2) Prompt + Elements -> ObjectsBlock (SceneObjects) - Multimodal
class SceneObjects(dspy.Signature):
    """
    MAXIMIZE VISUAL DETAIL: Transform multimodal inputs into RICHLY DETAILED visual entities.

    Your job is to EXPAND and ENRICH every element with vivid, cinematic visual detail.
    Use provided image/video as reference for accurate visual descriptions.
    DO NOT be conservative - be EXPANSIVE and IMAGINATIVE.

    Write from a NEUTRAL THIRD-PERSON OBSERVER perspective.

    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  FORBIDDEN (gendered terms):                                                 ║
    ║  woman, man, girl, boy, lady, gentleman, she, he, her, his, female, male    ║
    ║                                                                              ║
    ║  ALLOWED (neutral terms):                                                    ║
    ║  the subject, the figure, the individual, the person, [element_id]          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    WRONG EXAMPLES:
      ✗ "The woman wears a red dress with flowing fabric..."
      ✗ "She has long auburn hair cascading down her shoulders..."
      ✗ "His muscular frame is clad in a dark suit..."

    CORRECT EXAMPLES:
      ✓ "The subject wears a flowing crimson silk gown with delicate lace trim..."
      ✓ "The figure has long auburn hair cascading down the shoulders..."
      ✓ "[main_subject] stands gracefully in the garden, posture elegant..."
      ✓ "The individual's frame is clad in a tailored dark suit..."

    For EVERY field, provide RICH descriptions:
      - description: 3-4 sentences of detailed VISUAL appearance (clothing, features, materials)
      - shape_and_color: specific silhouette, color palette, gradients, contrasts
      - texture: material quality, surface feel, how light interacts with it
      - appearance_details: fine-grained unique details that make this entity distinctive

    For humans/animals, provide COMPLETE pose:
      - pose_class, body_orientation, key_body_parts (list 4-6 parts), gaze_direction, expressive_face

    Constraints (only these):
      - VISUAL ONLY: what can be SEEN (no fragrance, sound, mood statements)
      - NO camera/lighting terms (those go in Cinematography)
      - Use neutral third-person terms, NOT gendered terms
    """

    prompt: str = dspy.InputField(
        desc="Natural-language scene description to expand into rich visual detail."
    )
    image: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference image to extract visual details from."
    )
    video: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference video frames to extract visual details from."
    )
    audio: dspy.Audio | None = dspy.InputField(
        default=None,
        desc="Optional reference audio (may inform entity mood/expression)."
    )
    elements: Elements = dspy.InputField(
        desc="Element layer defining who/what exists. Create one SceneObject per element."
    )

    objects: ObjectsBlock = dspy.OutputField(
        desc=(
            "RICHLY DETAILED objects using NEUTRAL THIRD-PERSON perspective. "
            "FORBIDDEN: woman, man, girl, boy, she, he, her, his (gendered terms). "
            "ALLOWED: the subject, the figure, the person, [element_id]. "
            "Each object needs: dependencies=['element_id'], "
            "description (3-4 sentences), shape_and_color, texture, appearance_details. "
            "For humans/animals: MUST include pose. "
            "WRONG: 'The woman wears...' CORRECT: 'The subject wears...' or '[main_subject] wears...'"
        )
    )


# 3) Prompt + Elements -> ActionsBlock (SceneActions) - Multimodal
class SceneActions(dspy.Signature):
    """
    MAXIMIZE MOTION DETAIL: Infer RICH temporal dynamics and action semantics from multimodal inputs.

    Your job is to EXPAND motion descriptions with vivid physical detail.
    Use provided video for motion analysis; use audio for timing/rhythm cues.
    DO NOT be conservative - describe motion with CINEMATIC richness.

    For each action, provide:
      - dependencies: element_ids performing this action (REQUIRED, at least one)
      - action_class: type of action (walking, running, falling, holding, reaching, etc.)
      - stage_class: phase/stage (ongoing_casual, take_off, peak, landing, mid_stride, etc.)
      - description: 2-3 sentences of RICH physical motion detail
      - temporal_context: is_highlight_frame + frame_position_in_event

    === DESCRIPTION RULES ===
    Use [element_id] references, describe motion RICHLY with physical detail or adverbs further describing actions:
      - "[main_subject] strides gracefully along the path, each step measured and elegant,
         dress fabric rippling softly with the movement, one hand trailing lightly at side."
      - "[natural_element] cascades gently through the air in spiraling patterns, petals
         catching the light as they tumble, some brushing past [main_subject] before settling."
      - "[main_subject] pushes forward with powerful momentum, arms pumping rhythmically,
         feet striking the ground in rapid succession, body leaning into the sprint."

    ✗ FORBIDDEN: Gendered pronouns (he/she/his/her), mood/atmosphere statements

    Notes:
      - Use [element_id] references instead of pronouns
      - Describe PHYSICAL MOTION only (descriptions of mood/atmosphere go in Cinematography)
      - Material motion and material physics are valid action descriptors
    """

    prompt: str = dspy.InputField(
        desc="Scene description with implicit or explicit action hints."
    )
    image: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference image to infer pose/action from."
    )
    video: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference video frames to analyze motion and dynamics."
    )
    audio: dspy.Audio | None = dspy.InputField(
        default=None,
        desc="Optional reference audio for timing, rhythm, and pacing cues."
    )
    elements: Elements = dspy.InputField(
        desc="Element layer - Action.dependencies must reference these element_ids."
    )

    actions: ActionsBlock = dspy.OutputField(
        desc=(
            "RICHLY DETAILED actions. Each must have: dependencies=['element_id'], action_class, "
            "stage_class, description (2-3 sentences of vivid physical motion using [element_id]), "
            "temporal_context. BE EXPANSIVE with motion detail."
        )
    )


# 4) Prompt + Elements -> Cinematography (SceneCinematography) - Multimodal
class SceneCinematography(dspy.Signature):
    """
    MAXIMIZE CINEMATIC CRAFT: Derive RICH professional cinematography from multimodal inputs.

    Your job is to create DETAILED, PROFESSIONAL cinematography specifications.
    Use provided image/video as reference for camera setup and lighting analysis.
    Use audio for mood and pacing cues.
    DO NOT be conservative - craft a visually compelling cinematic treatment.

    CAMERA - make specific, intentional choices:
      - shot_size: extreme_close_up/close_up/medium_close_up/medium/medium_long/long/extreme_long
      - shot_framing: centered_single/rule_of_thirds/two_shot/over_the_shoulder/point_of_view
      - camera_angle: eye_level/low_angle/high_angle/dutch_angle/birds_eye/worms_eye
      - lens_size: 24mm_wide/35mm_standard/50mm_standard/85mm_portrait/135mm_telephoto
      - movement: static/dolly_in/dolly_out/pan_left/pan_right/tracking/crane_up/handheld/steadicam
      - depth_of_field: very_shallow/shallow/moderate/deep/infinite
      - focus: sharp_on_eyes/sharp_on_hands/rack_focus/foreground_subject/split_diopter

    LIGHTING - create atmosphere through light:
      - conditions: soft natural daylight/golden hour warmth/cool moonlight/neon city glow/candlelit
      - direction: frontal/side_left_45deg/side_right_45deg/backlit/rim_light/overhead/under_lighting
      - shadows: soft_diffused/hard_defined/deep_dramatic/minimal_flat/dappled
      - lighting_type: natural_daylight/natural_moonlight/artificial_tungsten/neon_colored/mixed_practical
      - mood_tag: romantic_soft/film_noir/high_key_bright/low_key_moody/chiaroscuro/ethereal

    COMPOSITION - describe the visual arrangement in 2-3 sentences:
      - description: "Subject positioned using rule of thirds, negative space on left drawing eye
         to the flowing dress, cherry blossoms creating natural frame at top of image."
      - subject_layout: primary_subject_centered/rule_of_thirds_left/rule_of_thirds_right/etc.

    LOOK - define the visual style and mood:
      - style_medium: photograph/film_35mm/digital_cinema/vintage_film/instant_polaroid
      - artistic_style: photorealistic/noir/impressionistic/surreal/documentary/cinematic
      - color_scheme: warm_earth_tones/cool_blues_and_silvers/complementary_orange_teal/monochromatic
      - mood_atmosphere: romantic_and_dreamy/tense_suspenseful/peaceful_serene/melancholic/euphoric
      - preference_score/aesthetic_score: very_high/high/medium/low

    Constraints (only these):
      - FILM CRAFT only (no clothing/appearance details - those go in Objects)
    """

    prompt: str = dspy.InputField(
        desc="Scene description - infer appropriate cinematography from mood and content."
    )
    image: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference image to analyze camera setup, lighting, and composition."
    )
    video: dspy.Image | None = dspy.InputField(
        default=None,
        desc="Optional reference video frames for camera movement and lighting analysis."
    )
    audio: dspy.Audio | None = dspy.InputField(
        default=None,
        desc="Optional reference audio for mood and pacing cues."
    )
    elements: Elements = dspy.InputField(
        desc="Element layer - Cinematography.dependencies must reference these element_ids."
    )

    cinematography: Cinematography = dspy.OutputField(
        desc=(
            "RICHLY DETAILED cinematography. All sub-components (camera, lighting, composition, look) "
            "must be fully specified. Composition.description should be 2-3 sentences. "
            "BE EXPANSIVE - craft professional-grade cinematic specifications."
        )
    )


####################
## Scene Critic   ##
####################
from pydantic import BaseModel, Field, ConfigDict


class Scene(BaseModel):
    """Full structured shot: elements + objects + actions + cinematography. ALL fields REQUIRED."""
    model_config = ConfigDict(extra="ignore")

    elements: Elements = Field(description="REQUIRED: The element/entity layer.")
    objects: list[SceneObject] = Field(
        default_factory=list,
        description="REQUIRED: List of fully-enriched scene objects."
    )
    actions: list[Action] = Field(
        default_factory=list,
        description="REQUIRED: List of actions with temporal context."
    )
    cinematography: Cinematography = Field(
        description="REQUIRED: Full cinematography specification."
    )


class SceneCritic(dspy.Signature):
    """
    Validate multi-head scene architecture for CROSS-HEAD lane separation and consistency.

    CRITICAL RULE: NEVER flag anything in the Cinematography head as LANE_LEAKAGE.
    Mood, atmosphere, romantic_soft, peaceful_serene - ALL belong in Cinematography.

    LANES:
      Elements = WHO/WHAT exists (entity_type, role, importance, rough_description=ROLE/TYPE only)
      Objects  = VISUAL appearance only (clothing, textures, pose, material behavior)
      Actions  = entity motion + adverbs describing HOW motion happens
      Cinematography = camera, lighting, mood, atmosphere (EVERYTHING mood-related goes here)

    LANE_LEAKAGE = content in the WRONG HEAD:
      - Elements.rough_description with appearance details: "woman in red dress", "man with wavy hair"
        (clothing, colors, materials, hair style = Objects, NOT Elements)
      - Camera terms in Objects/Actions: "medium shot", "fills the frame", "camera tracks"
      - Mood STATEMENTS in Objects/Actions: "enhancing the serene atmosphere", "creating a romantic feel"
      - Non-visual sensory in Objects: "fragrance", "scent", "sound of"
      - Gendered terms in Objects/Actions: "woman", "man", "she", "he" (use neutral: subject, figure, person)

    NEVER FLAG (these are CORRECT):
      - Elements.rough_description with role/type only: "walking figure", "garden path", "handbag held by subject"
      - Entity motion + adverbs: "walks gracefully", "drift softly", "falls gently"
      - Material/object physics: "dress flowing gently", "hair blowing in the wind", "fabric swaying"
      - Motion consequences: "dress flowing with each step", "petals landing on ground"
      - ANY content in Cinematography head

    OTHER ISSUES: CONTRADICTION (pose vs action, rough_description vs objects), INVALID_DEPENDENCY, MISSING_DATA
    """

    scene: Scene = dspy.InputField(desc="Full scene to validate.")

    issues: list[str] = dspy.OutputField(
        desc=(
            "Format: [TYPE] Head.element_id.field: 'quoted content' - reason. "
            "Examples:\n"
            "- '[LANE_LEAKAGE] Elements.main_subject.rough_description: \"woman in red dress\" - clothing in Elements'\n"
            "- '[LANE_LEAKAGE] Objects.main_subject.description: \"The woman wears\" - gendered term'\n"
            "- '[LANE_LEAKAGE] Actions.natural_element.description: \"serene atmosphere\" - mood in Actions'\n"
            "- '[CONTRADICTION] Elements.rough_description vs Objects: rough_description says red, objects says gray'\n"
            "- '[CONTRADICTION] Objects.main_subject.pose vs Actions.main_subject: standing pose but running action'\n"
            "- '[MISSING_DATA] Objects.accessory.texture: empty field'\n"
            "Types: LANE_LEAKAGE, CONTRADICTION, INVALID_DEPENDENCY, MISSING_DATA. "
            "Empty list [] if no issues."
        )
    )
    consistency_score: float = dspy.OutputField(
        desc="1.0 = perfect. Deduct per issue: LANE_LEAKAGE -0.15, CONTRADICTION -0.25, INVALID_DEPENDENCY -0.30, MISSING_DATA -0.10."
    )


class SceneSummary(dspy.Signature):
    """
    Generate a FAITHFUL EXTRACTIVE SUMMARY grounded in the Scene JSON.

    This is a human-readable description - use natural language including gendered terms
    from entity_type (e.g., "woman", "man"). The neutral third-person restrictions
    (subject, figure, person) do NOT apply here.

    GROUNDING RULES:
    - Extract and faithfully represent what is in the Scene object
    - Use entity_type from Elements for subject descriptions (woman, man, etc.)
    - Pull visual details from Objects (clothing, appearance, pose)
    - Pull motion details from Actions
    - Pull camera/lighting/mood from Cinematography
    - Do NOT invent details not present in the Scene

    THREE PARAGRAPHS:
    1. Subject (3-4 sentences): appearance, clothing, pose, expression
    2. Action & Environment (3-4 sentences): what's happening, setting details
    3. Camera & Mood (2-3 sentences): shot type, lighting, atmosphere

    FORMAT RULES:
    - Target 400-500 tokens (do NOT exceed 512)
    - Direct, efficient sentences
    - No flowery or redundant adjectives
    - No preamble - start directly with the subject
    """

    scene: Scene = dspy.InputField(
        desc="Full structured scene with all details to faithfully summarize."
    )
    short_description: str = dspy.OutputField(
        desc=(
            "FAITHFUL extractive summary grounded in Scene JSON. "
            "Use natural language including gendered terms from entity_type. "
            "3 paragraphs, 400-500 tokens (max 512). "
            "Paragraph 1: subject appearance. Paragraph 2: action and environment. Paragraph 3: camera and mood. "
            "Start immediately with subject description."
        )
    )
