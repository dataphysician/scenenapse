"""
Prompt Rewriter

GEPA-optimized prompt rewriter that uses visual feedback to improve prompts.
"""

import dspy
from typing import Dict, Any


class PromptRewriterSignature(dspy.Signature):
    """Rewrite prompt based on failed image and checker feedback."""
    image: dspy.Image = dspy.InputField(desc="The failed image for visual grounding")
    original_prompt: str = dspy.InputField(desc="Original user prompt")
    quality_feedback: str = dspy.InputField(desc="Feedback from Quality Checker about FIBO spec compliance")
    alignment_feedback: str = dspy.InputField(desc="Feedback from Alignment Checker about user intent")
    
    # No reasoning field - ChainOfThought adds it automatically
    improved_prompt: str = dspy.OutputField(desc="Rewritten prompt for FIBO that addresses the issues")


class PromptRewriter(dspy.Module):
    """Rewrites prompts based on visual feedback from checkers."""
    
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought(PromptRewriterSignature)
    
    def forward(
        self,
        image: dspy.Image,
        original_prompt: str,
        quality_feedback: str,
        alignment_feedback: str
    ) -> dspy.Prediction:
        """Rewrite prompt to address quality and alignment issues.
        
        Args:
            image: The failed image for visual grounding
            original_prompt: User's original prompt
            quality_feedback: Feedback about FIBO spec compliance
            alignment_feedback: Feedback about user intent alignment
            
        Returns:
            Prediction with improved_prompt
        """
        return self.rewrite(
            image=image,
            original_prompt=original_prompt,
            quality_feedback=quality_feedback,
            alignment_feedback=alignment_feedback
        )


def test_prompt_rewriter():
    """Test the prompt rewriter (requires Gemini API key)."""
    import os
    from PIL import Image as PILImage
    
    print("Testing Prompt Rewriter...")
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY or GEMINI_API_KEY found")
        print("   Skipping live test...")
        return False
    
    # Configure DSPy with Gemini 3
    try:
        lm = dspy.LM("gemini/gemini-2.0-flash-exp", api_key=api_key)
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"⚠️  Failed to configure DSPy: {e}")
        return False
    
    # Create dummy image
    img = PILImage.new('RGB', (100, 100), color='red')
    dspy_img = dspy.Image(img)
    
    # Test rewriting
    rewriter = PromptRewriter()
    
    try:
        result = rewriter(
            image=dspy_image,
            original_prompt="A dramatic sunset over mountains",
            quality_feedback="Lighting is flat and lacks drama. Color palette is muted gray instead of warm sunset tones.",
            alignment_feedback="No sunset visible. No mountains visible. Image is just a gray background."
        )
        print(f"  Original: A dramatic sunset over mountains")
        print(f"  Improved: {result.improved_prompt}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✅ Prompt Rewriter test complete!")
    return True


if __name__ == "__main__":
    test_prompt_rewriter()
