"""
API Request/Response schemas for SceneNapse Studio.

These schemas define the API contract between the Gradio frontend
and the FastAPI backend. They wrap the DSPy pipeline outputs.
"""

from pydantic import BaseModel, Field

# Import the core data models from backend
from backend.dspy_signatures import (
    Scene,
    Elements,
    SceneObject,
    Action,
    Cinematography,
)


class TimingInfo(BaseModel):
    """Timing information from pipeline execution."""
    stage1_elements_sec: float = 0.0
    stage2_parallel_sec: float = 0.0
    stage3_parallel_sec: float = 0.0
    total_sec: float = 0.0


class SceneResponse(BaseModel):
    """Response from the /api/generate endpoint."""
    scene: Scene = Field(description="Complete structured scene")
    critic_issues: list[str] = Field(
        default_factory=list,
        description="List of validation issues found by the critic"
    )
    critic_score: float = Field(
        default=1.0,
        description="Consistency score from 0.0 to 1.0"
    )
    short_description: str = Field(
        default="",
        description="Human-readable scene summary (400-500 tokens)"
    )
    timing: TimingInfo = Field(
        default_factory=TimingInfo,
        description="Execution timing breakdown"
    )
    summary_tokens: int = Field(
        default=0,
        description="Token count for the summary"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries due to low critic score"
    )


class RefineRequest(BaseModel):
    """Request body for the /api/refine endpoint."""
    scene: Scene = Field(description="Current scene to refine")
    instruction: str = Field(description="Natural language refinement instruction")
    regenerate_summary: bool = Field(
        default=True,
        description="Whether to regenerate the scene summary"
    )


class RefineResponse(BaseModel):
    """Response from the /api/refine endpoint."""
    scene: Scene = Field(description="Refined scene")
    short_description: str | None = Field(
        default=None,
        description="Updated scene summary (if regenerated)"
    )
    affected_heads: list[str] = Field(
        default_factory=list,
        description="Which heads were modified (ELEMENTS, OBJECTS, ACTIONS, CINEMATOGRAPHY)"
    )
    critic_score: float = Field(
        default=1.0,
        description="Consistency score from 0.0 to 1.0 after refinement"
    )
    critic_issues: list[str] = Field(
        default_factory=list,
        description="List of validation issues found by the critic after refinement"
    )
    retry_count: int = Field(
        default=0,
        description="Number of retries due to low critic score"
    )


class HealthResponse(BaseModel):
    """Response from the /api/health endpoint."""
    status: str = "ok"
    message: str = "SceneNapse API is running"
