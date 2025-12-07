"""
JoyQuality Image Selector

Uses JoyQuality SigLIP2 model to rank and select the best image from candidates.
"""

import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor, SiglipImageProcessor
from typing import List, Tuple


class JoyQualitySelector:
    """Selects the best image from candidates using JoyQuality SigLIP2 embeddings."""
    
    MODEL_ID = "fancyfeast/joyquality-siglip2-so400m-512-16-o8eg1n4c"
    # Base SigLIP2 processor to use if model doesn't have one
    BASE_PROCESSOR_ID = "google/siglip2-so400m-patch14-384"
    
    def __init__(self, device: str = None):
        """Initialize the JoyQuality model.
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading JoyQuality model on {self.device}...")
        
        # Load the classification model
        self.model = AutoModelForImageClassification.from_pretrained(
            self.MODEL_ID, 
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load processor from model, fallback to SigLIP base with correct size
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        except Exception:
            print(f"  Using base SigLIP processor with 512px size...")
            # Model expects 512x512 images (from model name: siglip2-so400m-512-16)
            self.processor = SiglipImageProcessor.from_pretrained(
                self.BASE_PROCESSOR_ID,
                size={"height": 512, "width": 512},
                crop_size={"height": 512, "width": 512}
            )
        
        print("JoyQuality model loaded successfully!")
    
    def get_quality_scores(self, images: List[Image.Image]) -> List[float]:
        """Get quality scores for a batch of images.
        
        Args:
            images: List of PIL Images to score
            
        Returns:
            List of quality scores (higher is better)
        """
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model output (regression - single value per image)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Model has 1 output = direct quality regression score
            # Squeeze to get float scores, apply sigmoid if needed for 0-1 range
            scores = logits.squeeze(-1)
            
            # Apply sigmoid to normalize to 0-1 range if scores are raw logits
            scores = torch.sigmoid(scores)
            
        return scores.cpu().tolist()
    
    def select_best(self, images: List[Image.Image]) -> Tuple[Image.Image, int, float]:
        """Select the best image from candidates.
        
        Args:
            images: List of PIL Images to choose from
            
        Returns:
            Tuple of (best_image, best_index, best_score)
        """
        if not images:
            raise ValueError("No images provided")
        
        scores = self.get_quality_scores(images)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        return images[best_idx], best_idx, scores[best_idx]
    
    def rank_images(self, images: List[Image.Image]) -> List[Tuple[int, float]]:
        """Rank all images by quality score.
        
        Args:
            images: List of PIL Images to rank
            
        Returns:
            List of (index, score) tuples sorted by score descending
        """
        scores = self.get_quality_scores(images)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked

    def score_image(self, image: Image.Image) -> float:
        """Score a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Quality score (0-1)
        """
        return self.get_quality_scores([image])[0]


def test_joyquality():
    """Test the JoyQuality selector with sample images."""
    import os
    import urllib.request
    
    print("Testing JoyQuality Selector...")
    
    # Create test images (colored squares)
    test_images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (512, 512), color)
        test_images.append(img)
        print(f"  Created test image {i}: {color}")
    
    # Initialize selector
    selector = JoyQualitySelector()
    
    # Test scoring
    print("\nGetting quality scores...")
    scores = selector.get_quality_scores(test_images)
    for i, score in enumerate(scores):
        print(f"  Image {i}: score = {score:.4f}")
    
    # Test selection
    print("\nSelecting best image...")
    best_img, best_idx, best_score = selector.select_best(test_images)
    print(f"  Best image: index {best_idx} with score {best_score:.4f}")
    
    # Test ranking
    print("\nRanking all images...")
    rankings = selector.rank_images(test_images)
    for rank, (idx, score) in enumerate(rankings):
        print(f"  Rank {rank + 1}: Image {idx} (score: {score:.4f})")
    
    print("\n✅ JoyQuality Selector test complete!")
    return True


if __name__ == "__main__":
    test_joyquality()
