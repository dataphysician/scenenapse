"""
Local FIBO Generator using Diffusers ModularPipeline.

This script runs the 'briaai/FIBO-VLM-prompt-to-JSON' model locally.
It requires 'diffusers' and 'accelerate' libraries.

Usage:
    ./.venv/bin/python src/fibo_local_generator.py "A futuristic city"
"""

import sys
import json
try:
    from diffusers import ModularPipeline
except ImportError:
    try:
        from diffusers.modular_pipelines import ModularPipeline
    except ImportError:
        print("Error: 'diffusers' library not found or ModularPipeline missing.")
        sys.exit(1)

class FIBOLocalGenerator:
    """Runs the FIBO model locally using ModularPipeline."""
    
    MODEL_ID = "briaai/FIBO-VLM-prompt-to-JSON"
    
    def __init__(self):
        """Initialize the pipeline."""
        print(f"Loading local FIBO pipeline ({self.MODEL_ID})...")
        try:
            # Uses custom code from Hugging Face
            self.pipeline = ModularPipeline.from_pretrained(
                self.MODEL_ID, 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            raise

    def generate(self, user_prompt: str) -> dict:
        """Generate structured JSON from a text prompt.
        
        Args:
            user_prompt: Natural language description
            
        Returns:
            Dict containing the FIBO structured prompt
        """
        # Run inference
        # The pipeline returns the result directly (likely a list or dict)
        output = self.pipeline(prompt=user_prompt)
        
        # Output handling might depend on exact return type
        # Assuming it returns the JSON object or a list containing it
        return output

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A serene lake at sunset with mountains in the background."
    
    try:
        generator = FIBOLocalGenerator()
        print(f"\nInput Prompt: {prompt}")
        result = generator.generate(prompt)
        print("\nGenerated FIBO Result:")
        print(result) 
        # Attempt to pretty print if it's a dict/list
        # try:
        #     print(json.dumps(result, indent=2))
        # except:
        #     print(result)
            
    except Exception as e:
        print(f"Failed to run local generator: {e}")

if __name__ == "__main__":
    main()
