"""
DSPy Guardrails for Image-to-Prompt Verification.

Verifies that generated images align with the Scene prompt components:
- ELEMENTS: Are all described elements present in the image?
- OBJECTS: Do object descriptions match what's visible?
- ACTIONS: Do described actions align with the image?
- CINEMATOGRAPHY: Does camera, lighting, composition match?

Uses a multimodal LLM to perform verification.
"""

import os
import dspy
from pydantic import BaseModel, Field
from typing import Literal

from .dspy_signatures import (
    Scene,
    Elements,
    SceneObject,
    Action,
    Cinematography,
)


# ============================================================
# Configuration
# ============================================================

MODEL_MULTIMODAL = os.getenv("MODEL_MULTIMODAL", "gemini/gemini-2.0-flash")


# ============================================================
# Verification Result Models
# ============================================================

class ElementVerification(BaseModel):
    """Verification result for a single element."""
    element_id: str
    element_type: str
    description: str
    present: bool = Field(description="Whether this element is clearly visible in the image")
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence level of the assessment")
    notes: str = Field(default="", description="Additional notes about the element's presence or absence")


class ObjectVerification(BaseModel):
    """Verification result for an object description."""
    object_index: int
    category: str
    description: str
    matches: bool = Field(description="Whether the object description matches what's in the image")
    accuracy: Literal["exact", "partial", "mismatch"] = Field(description="How well the description matches")
    issues: list[str] = Field(default_factory=list, description="Specific issues or discrepancies found")


class ActionVerification(BaseModel):
    """Verification result for an action."""
    action_index: int
    action_class: str
    description: str
    visible: bool = Field(description="Whether this action is visible/implied in the image")
    alignment: Literal["strong", "weak", "none"] = Field(description="How well the action aligns with the image")
    notes: str = Field(default="", description="Notes about action visibility")


class CinematographyVerification(BaseModel):
    """Verification result for cinematography settings."""
    camera_match: bool = Field(description="Whether camera settings (shot size, angle, etc.) match")
    lighting_match: bool = Field(description="Whether lighting matches the description")
    composition_match: bool = Field(description="Whether composition matches")
    look_match: bool = Field(description="Whether artistic style/look matches")
    overall_score: float = Field(ge=0, le=1, description="Overall cinematography alignment score")
    issues: list[str] = Field(default_factory=list, description="Specific cinematography issues")


class ImageVerificationResult(BaseModel):
    """Complete verification result for an image against a scene prompt."""
    # Overall pass/fail
    passed: bool = Field(description="Whether the image passes verification (all components must pass)")

    # Component scores (0 = fail, 1 = pass)
    elements_score: Literal[0, 1] = Field(description="Elements alignment: 1 if all elements present, 0 otherwise")
    objects_score: Literal[0, 1] = Field(description="Objects alignment: 1 if all objects match, 0 otherwise")
    actions_score: Literal[0, 1] = Field(description="Actions alignment: 1 if all actions visible, 0 otherwise")
    cinematography_score: Literal[0, 1] = Field(description="Cinematography alignment: 1 if settings match, 0 otherwise")

    # Total score (0-4)
    total_score: int = Field(ge=0, le=4, description="Sum of component scores (0-4)")

    # Detailed verifications
    elements: list[ElementVerification] = Field(default_factory=list)
    objects: list[ObjectVerification] = Field(default_factory=list)
    actions: list[ActionVerification] = Field(default_factory=list)
    cinematography: CinematographyVerification | None = None

    # Summary
    missing_elements: list[str] = Field(default_factory=list, description="Element IDs not found in image")
    critical_issues: list[str] = Field(default_factory=list, description="Critical issues that cause failure")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")


# ============================================================
# DSPy Signatures for Verification
# ============================================================

class VerifyElements(dspy.Signature):
    """Verify that scene elements are present in the generated image.

    Examine the image carefully and check if each described element is visible.
    Consider that some elements may be partially visible or implied rather than explicit.
    """
    image: dspy.Image = dspy.InputField(desc="The generated image to verify")
    elements_json: str = dspy.InputField(desc="JSON array of elements with id, type, and description")

    verification_json: str = dspy.OutputField(desc="JSON array of ElementVerification objects with present, confidence, notes for each element")


class VerifyObjects(dspy.Signature):
    """Verify that object descriptions match what's visible in the image.

    For each object, check if its category and description accurately represent
    what's shown in the image. Note any discrepancies in appearance, position, or attributes.
    """
    image: dspy.Image = dspy.InputField(desc="The generated image to verify")
    objects_json: str = dspy.InputField(desc="JSON array of objects with category and description")

    verification_json: str = dspy.OutputField(desc="JSON array of ObjectVerification objects with matches, accuracy, issues for each object")


class VerifyActions(dspy.Signature):
    """Verify that described actions are visible or implied in the image.

    Check if each action can be seen happening or is clearly implied by the scene.
    Consider that static images capture moments, so ongoing actions may be shown mid-motion.
    """
    image: dspy.Image = dspy.InputField(desc="The generated image to verify")
    actions_json: str = dspy.InputField(desc="JSON array of actions with action_class and description")

    verification_json: str = dspy.OutputField(desc="JSON array of ActionVerification objects with visible, alignment, notes for each action")


class VerifyCinematography(dspy.Signature):
    """Verify that cinematography settings match the generated image.

    Evaluate camera work (shot size, angle, framing), lighting (direction, mood, shadows),
    composition (subject placement, balance), and artistic look (style, color scheme).
    """
    image: dspy.Image = dspy.InputField(desc="The generated image to verify")
    cinematography_json: str = dspy.InputField(desc="JSON object with camera, lighting, composition, and look settings")

    camera_match: bool = dspy.OutputField(desc="True if camera settings match (shot size, angle, lens, etc.)")
    lighting_match: bool = dspy.OutputField(desc="True if lighting matches (conditions, direction, mood)")
    composition_match: bool = dspy.OutputField(desc="True if composition matches (subject layout, framing)")
    look_match: bool = dspy.OutputField(desc="True if artistic look matches (style, color scheme, atmosphere)")
    issues: str = dspy.OutputField(desc="Comma-separated list of specific cinematography issues found")


# ============================================================
# Image Verification Pipeline
# ============================================================

class ImageVerificationPipeline:
    """
    Pipeline for verifying generated images against scene prompts.

    Uses multimodal LLM to check alignment of:
    - Elements (characters, objects, settings)
    - Object descriptions
    - Actions
    - Cinematography settings

    Each component returns a binary score (0 = fail, 1 = pass).
    Total score ranges from 0 to 4. Passed = all 4 components pass.
    """

    def __init__(self, model: str | None = None):
        """
        Initialize the verification pipeline.

        Args:
            model: Multimodal model to use (default: MODEL_MULTIMODAL env var)
        """
        self.model = model or MODEL_MULTIMODAL

        # Initialize predictors
        self.verify_elements = dspy.Predict(VerifyElements)
        self.verify_objects = dspy.Predict(VerifyObjects)
        self.verify_actions = dspy.Predict(VerifyActions)
        self.verify_cinematography = dspy.Predict(VerifyCinematography)

    def verify(
        self,
        image: dspy.Image | str,
        scene: Scene | None = None,
        elements: Elements | None = None,
        objects: list[SceneObject] | None = None,
        actions: list[Action] | None = None,
        cinematography: Cinematography | None = None,
    ) -> ImageVerificationResult:
        """
        Verify an image against scene prompt components.

        Args:
            image: The image to verify (dspy.Image or URL string)
            scene: Complete Scene object (alternative to individual components)
            elements: Elements to verify
            objects: Objects to verify
            actions: Actions to verify
            cinematography: Cinematography settings to verify

        Returns:
            ImageVerificationResult with detailed verification results
        """
        import json

        # Extract components from scene if provided
        if scene:
            elements = elements or scene.elements
            objects = objects or scene.objects
            actions = actions or scene.actions
            cinematography = cinematography or scene.cinematography

        # Convert string URL to dspy.Image if needed
        if isinstance(image, str):
            image = dspy.Image(url=image)

        # Use multimodal model
        lm = dspy.LM(self.model, cache=True)

        # Initialize result containers
        element_verifications: list[ElementVerification] = []
        object_verifications: list[ObjectVerification] = []
        action_verifications: list[ActionVerification] = []
        cinematography_verification: CinematographyVerification | None = None

        with dspy.settings.context(lm=lm):
            # Verify elements
            if elements and elements.elements:
                elements_data = [
                    {"id": e.element_id, "type": e.element_type, "description": e.description}
                    for e in elements.elements
                ]
                try:
                    result = self.verify_elements(
                        image=image,
                        elements_json=json.dumps(elements_data)
                    )
                    parsed = json.loads(result.verification_json)
                    for i, item in enumerate(parsed):
                        if i < len(elements.elements):
                            elem = elements.elements[i]
                            element_verifications.append(ElementVerification(
                                element_id=elem.element_id,
                                element_type=elem.element_type,
                                description=elem.description,
                                present=item.get("present", False),
                                confidence=item.get("confidence", "low"),
                                notes=item.get("notes", ""),
                            ))
                except Exception as e:
                    print(f"Elements verification failed: {e}")

            # Verify objects
            if objects:
                objects_data = [
                    {"index": i, "category": o.category, "description": o.description}
                    for i, o in enumerate(objects)
                ]
                try:
                    result = self.verify_objects(
                        image=image,
                        objects_json=json.dumps(objects_data)
                    )
                    parsed = json.loads(result.verification_json)
                    for i, item in enumerate(parsed):
                        if i < len(objects):
                            obj = objects[i]
                            object_verifications.append(ObjectVerification(
                                object_index=i,
                                category=obj.category,
                                description=obj.description,
                                matches=item.get("matches", False),
                                accuracy=item.get("accuracy", "mismatch"),
                                issues=item.get("issues", []),
                            ))
                except Exception as e:
                    print(f"Objects verification failed: {e}")

            # Verify actions
            if actions:
                actions_data = [
                    {"index": i, "action_class": a.action_class, "description": a.description}
                    for i, a in enumerate(actions)
                ]
                try:
                    result = self.verify_actions(
                        image=image,
                        actions_json=json.dumps(actions_data)
                    )
                    parsed = json.loads(result.verification_json)
                    for i, item in enumerate(parsed):
                        if i < len(actions):
                            act = actions[i]
                            action_verifications.append(ActionVerification(
                                action_index=i,
                                action_class=act.action_class,
                                description=act.description,
                                visible=item.get("visible", False),
                                alignment=item.get("alignment", "none"),
                                notes=item.get("notes", ""),
                            ))
                except Exception as e:
                    print(f"Actions verification failed: {e}")

            # Verify cinematography
            if cinematography:
                cinematography_data = {
                    "camera": cinematography.camera.model_dump() if cinematography.camera else {},
                    "lighting": cinematography.lighting.model_dump() if cinematography.lighting else {},
                    "composition": cinematography.composition.model_dump() if cinematography.composition else {},
                    "look": cinematography.look.model_dump() if cinematography.look else {},
                }
                try:
                    result = self.verify_cinematography(
                        image=image,
                        cinematography_json=json.dumps(cinematography_data)
                    )
                    issues_list = [i.strip() for i in result.issues.split(",") if i.strip()]
                    matches = [result.camera_match, result.lighting_match, result.composition_match, result.look_match]
                    overall = sum(1 for m in matches if m) / len(matches)

                    cinematography_verification = CinematographyVerification(
                        camera_match=result.camera_match,
                        lighting_match=result.lighting_match,
                        composition_match=result.composition_match,
                        look_match=result.look_match,
                        overall_score=overall,
                        issues=issues_list,
                    )
                except Exception as e:
                    print(f"Cinematography verification failed: {e}")

        # Calculate binary scores (0 or 1)
        elements_score = self._calculate_elements_score(element_verifications)
        objects_score = self._calculate_objects_score(object_verifications)
        actions_score = self._calculate_actions_score(action_verifications)
        cinematography_score = self._calculate_cinematography_score(cinematography_verification)

        # Total score (0-4)
        total_score = elements_score + objects_score + actions_score + cinematography_score

        # Passed only if all components pass
        passed = total_score == 4

        # Identify missing elements and critical issues
        missing_elements = [e.element_id for e in element_verifications if not e.present]
        critical_issues = []

        if elements_score == 0:
            critical_issues.append(f"ELEMENTS failed: {len(missing_elements)} element(s) missing")
        if objects_score == 0:
            mismatched = sum(1 for o in object_verifications if not o.matches)
            critical_issues.append(f"OBJECTS failed: {mismatched} object(s) don't match")
        if actions_score == 0:
            invisible = sum(1 for a in action_verifications if not a.visible)
            critical_issues.append(f"ACTIONS failed: {invisible} action(s) not visible")
        if cinematography_score == 0 and cinematography_verification:
            critical_issues.append(f"CINEMATOGRAPHY failed: {', '.join(cinematography_verification.issues[:3])}")

        # Generate suggestions
        suggestions = self._generate_suggestions(
            element_verifications,
            object_verifications,
            action_verifications,
            cinematography_verification,
        )

        return ImageVerificationResult(
            passed=passed,
            elements_score=elements_score,
            objects_score=objects_score,
            actions_score=actions_score,
            cinematography_score=cinematography_score,
            total_score=total_score,
            elements=element_verifications,
            objects=object_verifications,
            actions=action_verifications,
            cinematography=cinematography_verification,
            missing_elements=missing_elements,
            critical_issues=critical_issues,
            suggestions=suggestions,
        )

    def _calculate_elements_score(self, verifications: list[ElementVerification]) -> Literal[0, 1]:
        """Return 1 if ALL elements are present, 0 otherwise."""
        if not verifications:
            return 1  # No elements to verify = pass

        all_present = all(v.present for v in verifications)
        return 1 if all_present else 0

    def _calculate_objects_score(self, verifications: list[ObjectVerification]) -> Literal[0, 1]:
        """Return 1 if ALL objects match, 0 otherwise."""
        if not verifications:
            return 1  # No objects to verify = pass

        all_match = all(v.matches for v in verifications)
        return 1 if all_match else 0

    def _calculate_actions_score(self, verifications: list[ActionVerification]) -> Literal[0, 1]:
        """Return 1 if ALL actions are visible, 0 otherwise."""
        if not verifications:
            return 1  # No actions to verify = pass

        all_visible = all(v.visible for v in verifications)
        return 1 if all_visible else 0

    def _calculate_cinematography_score(self, verification: CinematographyVerification | None) -> Literal[0, 1]:
        """Return 1 if ALL cinematography settings match, 0 otherwise."""
        if not verification:
            return 1  # No cinematography to verify = pass

        all_match = (
            verification.camera_match and
            verification.lighting_match and
            verification.composition_match and
            verification.look_match
        )
        return 1 if all_match else 0

    def _generate_suggestions(
        self,
        elements: list[ElementVerification],
        objects: list[ObjectVerification],
        actions: list[ActionVerification],
        cinematography: CinematographyVerification | None,
    ) -> list[str]:
        """Generate improvement suggestions based on verification results."""
        suggestions = []

        # Missing elements
        missing = [e for e in elements if not e.present]
        if missing:
            elem_types = list(set(e.element_type for e in missing))
            suggestions.append(f"Ensure {', '.join(elem_types)} elements are clearly visible")

        # Mismatched objects
        mismatched = [o for o in objects if o.accuracy == "mismatch"]
        if mismatched:
            categories = list(set(o.category for o in mismatched))
            suggestions.append(f"Improve accuracy of {', '.join(categories)} objects")

        # Invisible actions
        invisible = [a for a in actions if not a.visible]
        if invisible:
            action_types = list(set(a.action_class for a in invisible))
            suggestions.append(f"Make {', '.join(action_types)} actions more apparent")

        # Cinematography issues
        if cinematography:
            if not cinematography.camera_match:
                suggestions.append("Adjust camera angle or shot size to match specification")
            if not cinematography.lighting_match:
                suggestions.append("Modify lighting to match the specified conditions")
            if not cinematography.composition_match:
                suggestions.append("Reframe composition to match the layout specification")
            if not cinematography.look_match:
                suggestions.append("Adjust artistic style or color grading")

        return suggestions


# ============================================================
# Convenience Functions
# ============================================================

def verify_image(
    image_url: str,
    scene: Scene,
) -> ImageVerificationResult:
    """
    Convenience function to verify an image against a scene prompt.

    Args:
        image_url: URL of the generated image
        scene: The Scene object used to generate the image

    Returns:
        ImageVerificationResult with binary scores (0 or 1) for each component
    """
    pipeline = ImageVerificationPipeline()
    return pipeline.verify(image=image_url, scene=scene)


def verify_image_components(
    image_url: str,
    elements: Elements | None = None,
    objects: list[SceneObject] | None = None,
    actions: list[Action] | None = None,
    cinematography: Cinematography | None = None,
) -> ImageVerificationResult:
    """
    Verify an image against individual scene components.

    Args:
        image_url: URL of the generated image
        elements: Elements to verify
        objects: Objects to verify
        actions: Actions to verify
        cinematography: Cinematography settings to verify

    Returns:
        ImageVerificationResult with binary scores (0 or 1) for each component
    """
    pipeline = ImageVerificationPipeline()
    return pipeline.verify(
        image=image_url,
        elements=elements,
        objects=objects,
        actions=actions,
        cinematography=cinematography,
    )


# ============================================================
# CLI for Testing
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Verify an image against a scene prompt")
    parser.add_argument("--image", required=True, help="URL of the image to verify")
    parser.add_argument("--scene", required=True, help="Path to scene JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")

    args = parser.parse_args()

    # Load scene from JSON
    with open(args.scene, "r") as f:
        scene_data = json.load(f)

    scene = Scene(**scene_data)

    # Run verification
    print(f"Verifying image: {args.image}")
    print(f"Against scene with {len(scene.elements.elements)} elements, {len(scene.objects)} objects, {len(scene.actions)} actions")
    print("-" * 50)

    result = verify_image(args.image, scene)

    # Print results
    print(f"\n{'✓ PASSED' if result.passed else '✗ FAILED'} (Score: {result.total_score}/4)")
    print(f"\nComponent Scores (0=fail, 1=pass):")
    print(f"  ELEMENTS:       {result.elements_score}")
    print(f"  OBJECTS:        {result.objects_score}")
    print(f"  ACTIONS:        {result.actions_score}")
    print(f"  CINEMATOGRAPHY: {result.cinematography_score}")

    if result.missing_elements:
        print(f"\nMissing Elements: {', '.join(result.missing_elements)}")

    if result.critical_issues:
        print(f"\nFailed Components:")
        for issue in result.critical_issues:
            print(f"  ✗ {issue}")

    if result.suggestions:
        print(f"\nSuggestions:")
        for suggestion in result.suggestions:
            print(f"  → {suggestion}")

    if args.verbose:
        print(f"\n{'='*50}")
        print("Detailed Results:")
        print(json.dumps(result.model_dump(), indent=2))
