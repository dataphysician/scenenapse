"""
DSPy Evaluators

Quality Checker: Validates FIBO visual attribute compliance
Alignment Checker: Validates user intent fulfillment
"""

import dspy
from typing import Dict, Any, List


# ============================================================================
# Quality Checker (FIBO Visual Compliance)
# ============================================================================

class QualityCheckerSignature(dspy.Signature):
    """Check if image matches FIBO-specified visual/technical attributes."""
    image: dspy.Image = dspy.InputField(desc="Generated image to evaluate")
    json_prompt: str = dspy.InputField(desc="FIBO structured prompt (JSON string) with target visual specs")
    
    # FIBO visual attribute compliance
    objects_match_spec: bool = dspy.OutputField(desc="Matches FIBO 'objects' list details")
    background_matches_spec: bool = dspy.OutputField(desc="Matches FIBO 'background_setting'")
    lighting_matches_spec: bool = dspy.OutputField(desc="Matches FIBO 'lighting'")
    aesthetics_match_spec: bool = dspy.OutputField(desc="Matches FIBO 'aesthetics'")
    photo_characteristics_match_spec: bool = dspy.OutputField(desc="Matches FIBO 'photographic_characteristics'")
    style_matches_spec: bool = dspy.OutputField(desc="Matches FIBO 'style_medium' and 'artistic_style'")
    
    passed: bool = dspy.OutputField(desc="All FIBO visual specs rendered correctly")
    feedback: str = dspy.OutputField(desc="Which FIBO visual specs didn't match")


class QualityChecker(dspy.Module):
    """Evaluates if image matches FIBO-specified visual attributes."""
    
    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(QualityCheckerSignature)
    
    def forward(self, image: dspy.Image, json_prompt: Dict[str, Any]) -> dspy.Prediction:
        """Check image against FIBO specs.
        
        Args:
            image: DSPy Image to evaluate
            json_prompt: FIBO structured JSON prompt
            
        Returns:
            Prediction with compliance results and feedback
        """
        import json
        prompt_str = json.dumps(json_prompt, indent=2) if isinstance(json_prompt, dict) else str(json_prompt)
        return self.check(image=image, json_prompt=prompt_str)


# ============================================================================
# Alignment Checker (User Intent)
# ============================================================================

class AlignmentCheckerSignature(dspy.Signature):
    """Check if image fulfills user's original intent and semantic content."""
    image: dspy.Image = dspy.InputField(desc="Generated image to evaluate")
    original_prompt: str = dspy.InputField(desc="User's original natural language prompt")
    
    # User intent fulfillment
    subject_rendered: bool = dspy.OutputField(desc="Main subject from prompt is present")
    requested_elements_present: bool = dspy.OutputField(desc="All requested elements visible")
    style_matches_intent: bool = dspy.OutputField(desc="Visual style matches user's intent")
    mood_matches_intent: bool = dspy.OutputField(desc="Overall mood/atmosphere as intended")
    
    passed: bool = dspy.OutputField(desc="Image fulfills user's intent")
    missing_from_intent: List[str] = dspy.OutputField(desc="Elements user wanted but missing")
    feedback: str = dspy.OutputField(desc="How image fails to match user intent")


class AlignmentChecker(dspy.Module):
    """Evaluates if image captures user's original intent."""
    
    def __init__(self):
        super().__init__()
        self.check = dspy.ChainOfThought(AlignmentCheckerSignature)
    
    def forward(self, image: dspy.Image, original_prompt: str) -> dspy.Prediction:
        """Check image against user intent.
        
        Args:
            image: DSPy Image to evaluate
            original_prompt: User's original prompt
            
        Returns:
            Prediction with alignment results and feedback
        """
        return self.check(image=image, original_prompt=original_prompt)


def test_evaluators():
    """Test the evaluators (requires Gemini API key and test image)."""
    import os
    from PIL import Image as PILImage
    
    print("Testing DSPy Evaluators...")
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY or GEMINI_API_KEY found")
        print("   Skipping live test...")
        return False
    
    # Configure DSPy with Gemini 2.5 Flash Lite (Low latency, Multimodal)
    try:
        lm = dspy.LM("gemini/gemini-2.5-flash-lite-preview-09-2025", api_key=api_key)
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"⚠️  Failed to configure DSPy: {e}")
        return False
    
    # Create a test image
    test_image = PILImage.new('RGB', (512, 512), (135, 206, 235))  # Sky blue
    dspy_image = dspy.Image(test_image)
    
    # Test Quality Checker
    print("\n--- Testing Quality Checker ---")
    quality_checker = QualityChecker()
    test_json = {
        "description": "A clear blue sky",
        "objects": [{"description": "clouds", "location": "scattered"}],
        "background_setting": "Open sky",
        "lighting": "Natural daylight, bright",
        "aesthetics": "Clean, minimal",
        "photographic_characteristics": "Wide angle",
        "style_medium": "Photography",
        "artistic_style": "Realism",
        "context": "Daytime"
    }
    
    try:
        result = quality_checker(dspy_image, test_json)
        print(f"  Passed: {result.passed}")
        print(f"  Feedback: {result.feedback}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test Alignment Checker
    print("\n--- Testing Alignment Checker ---")
    alignment_checker = AlignmentChecker()
    
    try:
        result = alignment_checker(dspy_image, "A clear blue sky")
        print(f"  Passed: {result.passed}")
        print(f"  Subject rendered: {result.subject_rendered}")
        print(f"  Feedback: {result.feedback}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✅ Evaluators test complete!")
    return True


if __name__ == "__main__":
    test_evaluators()
