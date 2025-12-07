"""
Nano Banana Pro API Client

Client for Google's Nano Banana Pro API (Gemini Image Generation) for text-to-image generation.
Uses the official google-genai SDK.
"""

import os
import io
from typing import Optional, List, Tuple, Union, Dict, Any
from PIL import Image

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class NanoBananaProAPI:
    """Client for Nano Banana Pro API (Gemini Image Generation)."""
    
    # Model options
    MODEL_FLASH = "gemini-2.5-flash-preview-05-20"  # Nano Banana (text+image)
    MODEL_PRO = "gemini-3-pro-image-preview"        # Nano Banana Pro
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """Initialize the Nano Banana Pro API client.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY env var.
            model: Model to use (default: gemini-2.5-flash-image)
        """
        if not HAS_GENAI:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable.")
        
        # Configure the client
        self.client = genai.Client(api_key=self.api_key)
        self.model = model or self.MODEL_PRO  # Default to Nano Banana Pro
    
    async def generate_stream(self, prompt: Union[str, Dict[str, Any]], num_seeds: int = 10, size: Tuple[int, int] = (1024, 1024)):
        """Generate images and yield them as they complete.
        
        Args:
            prompt: Text or JSON prompt
            num_seeds: Number of variations
            
        Yields:
            (seed_idx, image) tuple
        """
        import asyncio
        import json
        
        # Prepare prompt
        if isinstance(prompt, dict):
            json_str = json.dumps(prompt, indent=2)
            base_prompt = f"Generate an image following this detailed specification:\n\n```json\n{json_str}\n```"
        else:
            base_prompt = str(prompt)
            
        print(f"  Generating {num_seeds} variations concurrently...")

        async def fetch_one(seed_idx):
            varied_prompt = f"{base_prompt}\n\n(Variation seed: {seed_idx})\n\n[Aspect Ratio: 1:1]"
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=[varied_prompt],
                )
                for part in response.parts:
                    if part.inline_data is not None:
                        genai_img = part.as_image()
                        return seed_idx, self._genai_to_pil(genai_img)
            except Exception as e:
                print(f"  Warning: Async generation failed for seed {seed_idx}: {e}")
            return seed_idx, None

        # Create tasks
        pending = [asyncio.create_task(fetch_one(i)) for i in range(num_seeds)]
        
        for task in asyncio.as_completed(pending):
            seed_idx, img = await task
            if img:
                yield seed_idx, img

    async def generate(self, prompt: Union[str, Dict[str, Any]], num_seeds: int = 10, size: Tuple[int, int] = (1024, 1024)) -> List[Image.Image]:
        """Generate multiple image variations concurrently (batch)."""
        images = []
        async for _, img in self.generate_stream(prompt, num_seeds, size):
            images.append(img)
        return images
    
    def generate_single(self, prompt: str) -> Optional[Image.Image]:
        """Generate a single image (synchronous wrapper)."""
        # Kept for backward compatibility or simple tests
        import asyncio
        try:
             # Run the async generation for 1 image in a new loop
             return asyncio.run(self.generate(prompt, num_seeds=1))[0]
        except Exception as e:
            print(f"Error in generate_single: {e}")
            return None
    
    def _genai_to_pil(self, genai_image) -> Image.Image:
        """Convert a genai.Image to PIL.Image.
        
        Args:
            genai_image: Image from genai response
            
        Returns:
            PIL Image
        """
        # genai.Image has _pil_image property or can be converted via bytes
        if hasattr(genai_image, '_pil_image') and genai_image._pil_image:
            return genai_image._pil_image
        elif hasattr(genai_image, 'data'):
            return Image.open(io.BytesIO(genai_image.data))
        else:
            # Try to access the underlying data
            import base64
            if hasattr(genai_image, '_image_bytes'):
                return Image.open(io.BytesIO(genai_image._image_bytes))
            # Last resort - save and reload
            temp_path = '/tmp/genai_temp.png'
            genai_image.save(temp_path)
            return Image.open(temp_path)
    
    def _json_to_prompt(self, json_prompt: Dict[str, Any]) -> str:
        """Convert FIBO JSON to natural language prompt.
        
        Args:
            json_prompt: FIBO structured prompt
            
        Returns:
            Natural language prompt string
        """
        parts = []
        
        if isinstance(json_prompt, dict):
            # Subject - most important
            if "subject" in json_prompt:
                subj = json_prompt["subject"]
                if isinstance(subj, dict):
                    if subj.get("description"):
                        parts.append(subj["description"])
                    else:
                        subj_parts = []
                        if subj.get("pose"):
                            subj_parts.append(subj["pose"])
                        if subj.get("expression"):
                            subj_parts.append(subj["expression"])
                        if subj_parts:
                            parts.append(" ".join(subj_parts))
                elif isinstance(subj, str):
                    parts.append(subj)
            
            # Composition
            if "composition" in json_prompt:
                comp = json_prompt["composition"]
                if isinstance(comp, dict):
                    if comp.get("framing"):
                        parts.append(f"{comp['framing']} framing")
                    if comp.get("focal_point"):
                        parts.append(f"focus on {comp['focal_point']}")
            
            # Lighting
            if "lighting" in json_prompt:
                light = json_prompt["lighting"]
                if isinstance(light, dict):
                    light_desc = []
                    if light.get("type"):
                        light_desc.append(light["type"])
                    if light.get("intensity"):
                        light_desc.append(light["intensity"])
                    if light_desc:
                        parts.append(f"{' '.join(light_desc)} lighting")
            
            # Camera
            if "camera" in json_prompt:
                cam = json_prompt["camera"]
                if isinstance(cam, dict):
                    if cam.get("angle"):
                        parts.append(f"{cam['angle']} angle")
                    if cam.get("focal_length"):
                        parts.append(f"{cam['focal_length']} lens")
            
            # Color
            if "color" in json_prompt:
                color = json_prompt["color"]
                if isinstance(color, dict):
                    if color.get("palette"):
                        parts.append(f"{color['palette']} color palette")
                    if color.get("mood"):
                        parts.append(f"{color['mood']} mood")
            
            # Background
            if "background" in json_prompt:
                bg = json_prompt["background"]
                if isinstance(bg, dict):
                    if bg.get("description"):
                        parts.append(f"with {bg['description']} background")
                    elif bg.get("type"):
                        parts.append(f"with {bg['type']} background")
        else:
            return str(json_prompt)
        
        return ", ".join(parts) if parts else str(json_prompt)
    
    def _create_placeholder_image(self, size: tuple, seed: int) -> Image.Image:
        """Create a placeholder image for testing.
        
        Args:
            size: Image dimensions
            seed: Seed for color variation
            
        Returns:
            PIL Image
        """
        import random
        random.seed(seed)
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        
        return Image.new('RGB', size, (r, g, b))


def test_nano_banana():
    """Test the Nano Banana Pro API client."""
    import os
    
    print("Testing Nano Banana Pro API...")
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No API key found. Set GOOGLE_API_KEY to test.")
        return False
    
    if not HAS_GENAI:
        print("⚠️  google-genai not installed. Run: pip install google-genai")
        return False
    
    try:
        client = NanoBananaProAPI(api_key=api_key)
        print(f"  Using model: {client.model}")
    except Exception as e:
        print(f"⚠️  Failed to initialize: {e}")
        return False
    
    # Test single generation
    print("\nGenerating single image...")
    img = client.generate_single("A beautiful sunset over mountains")
    if img:
        print(f"  ✅ Generated image: {img.size}, mode={img.mode}")
        img.save("/tmp/nano_banana_test.png")
        print(f"  Saved to /tmp/nano_banana_test.png")
    else:
        print("  ❌ Failed to generate image")
    
    # Test batch generation with FIBO prompt
    print("\nGenerating batch with FIBO prompt...")
    test_json = {
        "composition": {"framing": "rule_of_thirds", "focal_point": "center"},
        "lighting": {"type": "natural", "intensity": "high"},
        "camera": {"angle": "eye_level", "focal_length": "35mm"},
        "color": {"palette": "warm", "mood": "dramatic"},
        "subject": {"description": "mountain landscape at sunset"},
        "background": {"type": "sky with clouds"}
    }
    
    images = client.generate(test_json, num_seeds=2)
    print(f"  Generated {len(images)} images")
    for i, img in enumerate(images):
        print(f"    Image {i}: {img.size}, mode={img.mode}")
    
    print("\n✅ Nano Banana Pro API test complete!")
    return True


if __name__ == "__main__":
    test_nano_banana()
