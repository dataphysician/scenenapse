"""
In-memory state management for the current scene.

Maintains Python objects (Elements, Objects, Actions, Cinematography, Scene)
in backend memory while the frontend displays JSON representations.
"""

from dataclasses import dataclass, field
from typing import Any

from backend.dspy_signatures import (
    Scene,
    Elements,
    Element,
    SceneObject,
    Action,
    Cinematography,
    CinematographyCamera,
    CinematographyLighting,
    CinematographyComposition,
    CinematographyLook,
)


# Default cinematography settings
DEFAULT_CINEMATOGRAPHY = Cinematography(
    dependencies=[],
    camera=CinematographyCamera(
        shot_size="medium",
        shot_framing="rule_of_thirds",
        camera_angle="eye_level",
        lens_size="35mm",
        movement="static",
        depth_of_field="moderate",
        focus="sharp_on_subject",
    ),
    lighting=CinematographyLighting(
        conditions="soft natural daylight",
        direction="frontal",
        shadows="soft_diffused",
        lighting_type="natural_daylight",
        mood_tag="neutral",
    ),
    composition=CinematographyComposition(
        description="Balanced composition.",
        subject_layout="centered",
    ),
    look=CinematographyLook(
        style_medium="digital_cinema",
        artistic_style="photorealistic",
        color_scheme="natural",
        mood_atmosphere="neutral",
        preference_score="medium",
        aesthetic_score="medium",
    ),
)


@dataclass
class ReferenceImage:
    """A reference image from Freepik search."""
    id: int
    url: str
    thumbnail: str | None
    title: str | None


@dataclass
class GeneratedImage:
    """A generated image from Fal AI."""
    index: int  # 1-based index
    url: str
    width: int
    height: int
    quality_score: float | None = None  # JoyQuality score (0-1)


@dataclass
class SceneState:
    """
    In-memory storage for the current scene components.

    Each component is stored as a proper Python/Pydantic object,
    providing type validation and structure guarantees.
    """
    elements: Elements = field(default_factory=lambda: Elements(elements=[]))
    objects: list[SceneObject] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    cinematography: Cinematography | None = None

    # Metadata from generation
    short_description: str = ""
    critic_score: float = 0.0
    critic_issues: list[str] = field(default_factory=list)

    # Reference images from Freepik search
    reference_images: list[ReferenceImage] = field(default_factory=list)
    selected_reference: ReferenceImage | None = None

    # Generated images from Fal AI
    generated_images: list[GeneratedImage] = field(default_factory=list)
    selected_generated: GeneratedImage | None = None

    def to_scene(self) -> Scene:
        """Assemble components into a complete Scene object."""
        return Scene(
            elements=self.elements,
            objects=self.objects,
            actions=self.actions,
            cinematography=self.cinematography or DEFAULT_CINEMATOGRAPHY,
        )

    def from_scene(self, scene: Scene) -> None:
        """Update state from a complete Scene object."""
        self.elements = scene.elements
        self.objects = scene.objects
        self.actions = scene.actions
        self.cinematography = scene.cinematography

    def clear(self) -> None:
        """Reset to empty state."""
        self.elements = Elements(elements=[])
        self.objects = []
        self.actions = []
        self.cinematography = None
        self.short_description = ""
        self.critic_score = 0.0
        self.critic_issues = []
        self.reference_images = []
        self.selected_reference = None
        self.generated_images = []
        self.selected_generated = None

    def to_dict(self) -> dict[str, Any]:
        """Convert state to JSON-serializable dict for frontend."""
        return {
            "elements": self.elements.model_dump(),
            "objects": [obj.model_dump() for obj in self.objects],
            "actions": [act.model_dump() for act in self.actions],
            "cinematography": self.cinematography.model_dump() if self.cinematography else None,
            "short_description": self.short_description,
            "critic_score": self.critic_score,
            "critic_issues": self.critic_issues,
            "reference_images": [
                {"id": img.id, "url": img.url, "thumbnail": img.thumbnail, "title": img.title}
                for img in self.reference_images
            ],
            "selected_reference": {
                "id": self.selected_reference.id,
                "url": self.selected_reference.url,
                "thumbnail": self.selected_reference.thumbnail,
                "title": self.selected_reference.title,
            } if self.selected_reference else None,
            "generated_images": [
                {"index": img.index, "url": img.url, "width": img.width, "height": img.height, "quality_score": img.quality_score}
                for img in self.generated_images
            ],
            "selected_generated": {
                "index": self.selected_generated.index,
                "url": self.selected_generated.url,
                "width": self.selected_generated.width,
                "height": self.selected_generated.height,
                "quality_score": self.selected_generated.quality_score,
            } if self.selected_generated else None,
        }


# Global singleton instance
_scene_state: SceneState | None = None


def get_scene_state() -> SceneState:
    """Get the global scene state instance (lazy initialization)."""
    global _scene_state
    if _scene_state is None:
        _scene_state = SceneState()
    return _scene_state


def reset_scene_state() -> SceneState:
    """Reset and return a fresh scene state."""
    global _scene_state
    _scene_state = SceneState()
    return _scene_state
