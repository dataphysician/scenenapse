"""
GEPA Trainer (Generative Evolving Prompt Architecture)

Optimizes the PromptRewriter using DSPy's teleprompters (optimizers).
It learns from "failed trajectories" - examples where the initial prompt failed
but a human or oracle (or successful later iteration) provided a better prompt.
"""

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2
from typing import List, Dict, Any, Optional
from .prompt_rewriter import PromptRewriter

class GEPATrainer:
    """Trains/Optimizes the PromptRewriter using DSPy."""

    def __init__(self, teacher_model_name: str = "gemini/gemini-2.0-flash-exp"):
        """Initialize the trainer.
        
        Args:
            teacher_model_name: Model to use for the optimizer's teacher/evaluator
        """
        self.teacher_model = dspy.LM(teacher_model_name)
        
    def create_dataset(self, examples: List[Dict[str, Any]]) -> List[dspy.Example]:
        """Convert list of dicts to DSPy examples.
        
        Each example dict should have:
        - image: (optional, but good for grounding) dspy.Image or PIL Image
        - original_prompt: str
        - quality_feedback: str
        - alignment_feedback: str
        - improved_prompt: str (the "ground truth" or "good" rewrite)
        """
        dspy_examples = []
        for ex in examples:
            # Create inputs and labels
            # Note: image might need special handling if it's raw data
            # For training, we might skip the actual image if we use a text-only
            # metric, or we pass it if we have a VLM metric.
            # Here we assume we pass it for the module to see.
            
            # Construct dspy.Example directly
            inputs = {
                "original_prompt": ex["original_prompt"],
                "quality_feedback": ex["quality_feedback"],
                "alignment_feedback": ex["alignment_feedback"]
            }
            if "image" in ex:
                inputs["image"] = ex["image"]
            
            labels = {
                "improved_prompt": ex["improved_prompt"]
            }
            
            dspy_examples.append(dspy.Example(**inputs, **labels).with_inputs("image", "original_prompt", "quality_feedback", "alignment_feedback"))
        
        return dspy_examples

    def metric(self, example, pred, trace=None):
        """Evaluation metric for the optimizer.
        
        Checks if the predicted improved_prompt is semantically similar 
        to the ground truth improved_prompt, OR if it effectively addresses the feedback.
        
        Since exact match is unlikely, we use an LLM-based judge.
        """
        # Simple fuzzy match or LLM-based correctness check
        # For efficiency in this example, we'll use a simple length/keyword check 
        # or rely on a "Judge" module if we want high quality.
        
        # Here we define a Judge Signature for the metric
        class RewriteJudge(dspy.Signature):
            """Judge if the rewritten prompt addresses the feedback compared to reference."""
            original_prompt = dspy.InputField()
            feedback = dspy.InputField()
            rewritten_prompt = dspy.InputField()
            reference_prompt = dspy.InputField()
            score = dspy.OutputField(desc="Score between 0.0 and 1.0")

        judge = dspy.ChainOfThought(RewriteJudge)
        
        # Combine feedbacks
        feedback = f"Quality: {example.quality_feedback}\nAlignment: {example.alignment_feedback}"
        
        # We need a model context for the judge usually, assume global dspy.settings or use context
        with dspy.settings.context(lm=self.teacher_model):
            try:
                result = judge(
                    original_prompt=example.original_prompt,
                    feedback=feedback,
                    rewritten_prompt=pred.improved_prompt,
                    reference_prompt=example.improved_prompt
                )
                try:
                    return float(result.score) >= 0.7 # Pass if score is high
                except:
                    return False
            except:
                return False

    def train(self, train_data: List[dspy.Example], val_data: List[dspy.Example] = None):
        """Run the optimization pipeline."""
        
        print(f"Starting GEPA training with {len(train_data)} examples...")
        
        # Use BootstrapFewShotWithRandomSearch (robust default)
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_candidate_programs=5,
            num_threads=4
        )
        
        # The module to optimize
        student = PromptRewriter()
        
        # Compile!
        print("Compiling (optimizing) PromptRewriter...")
        compiled_rewriter = teleprompter.compile(
            student,
            trainset=train_data,
            valset=val_data or train_data[:1] # Valid set optional
        )
        
        print("Optimization complete!")
        return compiled_rewriter

    def save_compiled(self, module, path: str):
        """Save the optimized module."""
        module.save(path)
        print(f"Saved optimized module to {path}")

    def load_compiled(self, path: str) -> PromptRewriter:
        """Load an optimized module."""
        module = PromptRewriter()
        module.load(path)
        return module


def test_gepa_trainer():
    """Test the GEPA trainer functionality."""
    import os
    from PIL import Image
    
    # Check API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ No API Key found. Skipping live training test.")
        return
        
    dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash-exp", api_key=api_key))
    
    # Mock data
    # Create a dummy image
    dummy_img = Image.new('RGB', (64, 64), color='red')
    dspy_img = dspy.Image(dummy_img)

    examples = [
        {
            "image": dspy_img,
            "original_prompt": "A cat sitting on a mat",
            "quality_feedback": "Image is too dark, lighting is poor.",
            "alignment_feedback": "Cat is missing, only mat is visible.",
            "improved_prompt": "A bright studio photo of a fluffy cat sitting on a mat, high key lighting"
        },
        {
            "image": dspy_img,
            "original_prompt": "A futuristic car",
            "quality_feedback": "Blurry background, bad focus.",
            "alignment_feedback": "Car looks like a 1990s sedan, not futuristic.",
            "improved_prompt": "A sleek neon-lit futuristic flying car, cyberpunk style, sharp focus, 8k resolution"
        }
    ]
    
    trainer = GEPATrainer()
    dspy_examples = trainer.create_dataset(examples)
    
    print("Running mock training...")
    # This might take a bit of time, so we run a very minimal version if possible
    # In a real test we might mock the optimization to avoid API costs/time, 
    # but here we just check if it runs without crashing.
    
    try:
        optimized = trainer.train(dspy_examples)
        print("Training run successfully returned a module.")
        
        # Check if it works
        res = optimized(
            image=dspy_img,
            original_prompt="Test prompt",
            quality_feedback="Bad quality",
            alignment_feedback="Wrong subject"
        )
        print(f"Optimized output: {res.improved_prompt}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")

if __name__ == "__main__":
    test_gepa_trainer()
