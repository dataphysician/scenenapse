"""
FIBO Structured Prompt Generator

Converts natural language prompts to structured JSON prompts using Gemini 3.
"""

import dspy
from typing import Dict, Any


class FIBOPromptSignature(dspy.Signature):
    """Convert user prompt to FIBO-style structured JSON."""
    user_prompt: str = dspy.InputField(desc="Natural language description of desired image")
    
    json_prompt: dict = dspy.OutputField(desc="""Structured JSON adhering to Bria AI's FIBO schema:
    - description: str (General visual description of the scene)
    - objects: list[dict] (Detailed list of elements. Each object dict keys: 
        - description: what is it?
        - location: where is it? (e.g. "center", "background")
        - relative_size: "small", "large", etc.
        - pose: if animate (e.g. "sitting", "running")
        - clothing: if applicable
        - expression: facial expression)
    - background_setting: str (The environment/setting, e.g. "A cyberpunk city street at night")
    - lighting: str (Specific lighting setup, e.g. "Neon lights, rim lighting, blue and pink hues")
    - aesthetics: str (Artistic vibe, e.g. "Cinematic, high contrast, moody")
    - photographic_characteristics: str (Camera details: "Wide angle", "f/1.8", "85mm lens", "Macro")
    - style_medium: str (The medium: "Photography", "Oil painting", "3D Render")
    - artistic_style: str (Specific style reference: "Cyberpunk", "Minimalist", "Baroque")
    - context: str (Additional context or narrative elements)
    """)


class FIBOGenerator(dspy.Module):
    """Generates FIBO-style structured JSON prompts from natural language."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FIBOPromptSignature)
    
    def forward(self, user_prompt: str) -> Dict[str, Any]:
        """Convert user prompt to structured JSON.
        
        Args:
            user_prompt: Natural language description
            
        Returns:
            Structured JSON prompt dict
        """
        result = self.generate(user_prompt=user_prompt)
        return result.json_prompt


def test_fibo_generator():
    """Test the FIBO generator (requires Gemini API key)."""
    import os
    
    print("Testing FIBO Generator...")
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY or GEMINI_API_KEY found in environment")
        print("   Set one to test FIBO generator with real API calls")
        print("   Skipping live test...")
        return False
    
    # Configure DSPy with Gemini
    try:
        lm = dspy.LM("gemini/gemini-2.0-flash-exp", api_key=api_key)
        dspy.configure(lm=lm)
    except Exception as e:
        print(f"⚠️  Failed to configure DSPy with Gemini: {e}")
        return False
    
    # Test generation
    generator = FIBOGenerator()
    
    test_prompts = [
        "A dramatic sunset over mountains",
        "A portrait of a woman in Renaissance style",
        "A futuristic cityscape at night"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        try:
            result = generator(prompt)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ FIBO Generator test complete!")
    return True


if __name__ == "__main__":
    test_fibo_generator()
