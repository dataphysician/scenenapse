#!/usr/bin/env python3
"""
SceneNapse Studio - Main Entry Point

Run the FastAPI backend server.
For the frontend, run `npm run dev` in the frontend/ directory.
"""

import os
import sys


def main():
    """Start the FastAPI backend server."""
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Add current directory to path for imports
    sys.path.insert(0, os.getcwd())

    import uvicorn
    from api.main import app

    print("=" * 60)
    print("SceneNapse Studio - Backend Server")
    print("=" * 60)
    print()
    print("Starting FastAPI server on http://localhost:8000")
    print()
    print("To start the frontend:")
    print("  cd frontend && npm install && npm run dev")
    print()
    print("Frontend will be available at http://localhost:3000")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
