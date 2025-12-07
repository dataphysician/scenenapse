"""
Data Collector Strategy

Helper to save failed prompt trajectories for GEPA training.
"""

import json
import os
import uuid
from typing import Dict, Any

class DataCollector:
    """Collects optimization data."""
    
    def __init__(self, data_dir: str = "data/failures"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def log_failure(
        self,
        original_prompt: str,
        quality_feedback: str,
        alignment_feedback: str,
        json_prompt: Dict[str, Any],
        image_path: str = None
    ):
        """Log a failed trajectory."""
        entry = {
            "id": str(uuid.uuid4()),
            "original_prompt": original_prompt,
            "quality_feedback": quality_feedback,
            "alignment_feedback": alignment_feedback,
            "json_prompt": json_prompt,
            "image_path": image_path,
            "improved_prompt": None  # To be filled by human/oracle later
        }
        
        filename = f"{self.data_dir}/{entry['id']}.json"
        with open(filename, "w") as f:
            json.dump(entry, f, indent=2)
            
        print(f"Logged failure case to {filename}")

def test_collector():
    collector = DataCollector()
    collector.log_failure(
        "A bad prompt", 
        "Bad quality", 
        "Wrong subject", 
        {"some": "json"}
    )

if __name__ == "__main__":
    test_collector()
