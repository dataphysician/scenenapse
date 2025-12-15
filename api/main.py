"""
FastAPI Application for SceneNapse Studio.

Main entry point for the API server.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import configure_dspy
from .routes import router
from backend.joy_quality import initialize_joy_quality


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure DSPy and JoyQuality on startup."""
    configure_dspy()

    # Initialize JoyQuality model for image quality scoring
    print("\n" + "=" * 60)
    print("Initializing JoyQuality Image Quality Model...")
    print("=" * 60)
    try:
        initialize_joy_quality()
        print("JoyQuality model ready for image quality scoring!")
    except Exception as e:
        print(f"Warning: JoyQuality model failed to load: {e}")
        print("Image quality scoring will be disabled.")
    print("=" * 60 + "\n")

    yield


app = FastAPI(
    title="SceneNapse Studio API",
    description="API for cinematic scene generation and refinement using DSPy",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Vite dev server
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
