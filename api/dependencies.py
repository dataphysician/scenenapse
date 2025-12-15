"""
API Dependencies for SceneNapse Studio.

Provides singleton instances of the DSPy pipelines and handles
LM configuration at startup.
"""

import os
from functools import lru_cache

import dspy

# Import pipelines from backend
from backend.dspy_pipeline import (
    SceneGenerationPipeline,
    MODEL_DEFAULT,
)
from backend.dspy_refinement import SmartRefinementPipeline


def configure_dspy() -> None:
    """
    Configure DSPy with the default LM.

    Uses OPENAI_API_KEY from environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY environment variable."
        )

    lm = dspy.LM(MODEL_DEFAULT, api_key=api_key, cache=False)
    dspy.configure(lm=lm)


@lru_cache(maxsize=1)
def get_generation_pipeline() -> SceneGenerationPipeline:
    """
    Get singleton instance of the SceneGenerationPipeline.

    Uses lru_cache to ensure only one instance is created.
    """
    return SceneGenerationPipeline()


def get_refinement_pipeline() -> SmartRefinementPipeline:
    """
    Get a new instance of the SmartRefinementPipeline.

    Creates a fresh instance for each request (no caching).
    """
    return SmartRefinementPipeline()
