from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import asyncio
import time
import sys

# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from src.prompt_optimizer import PromptOptimizer

app = FastAPI()

# Mount output directory for serving images
os.makedirs("output", exist_ok=True)
app.mount("/static", StaticFiles(directory="output"), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API Key and Configure DSPy BEFORE creating optimizer
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ WARNING: GOOGLE_API_KEY not found in environment - evaluators will fail!")
else:
    # Configure DSPy at module level so all evaluators have access
    try:
        lm = dspy.LM("gemini/gemini-2.5-flash-lite-preview-09-2025", api_key=api_key)
        dspy.configure(lm=lm)
        print("✅ DSPy configured successfully with Gemini 2.5 Flash Lite")
    except Exception as e:
        print(f"❌ DSPy configuration failed: {e}")
        # Try fallback model
        try:
            lm = dspy.LM("gemini/gemini-2.0-flash-exp", api_key=api_key)
            dspy.configure(lm=lm)
            print("✅ DSPy configured with fallback model (Gemini 2.0 Flash)")
        except Exception as e2:
            print(f"❌ DSPy fallback also failed: {e2}")

# Global optimizer instance (now DSPy is configured)
optimizer = PromptOptimizer(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: str
    mode: str = "optimized"  # "optimized" or "regular"

@app.post("/api/generate")
async def generate(request: PromptRequest):
    """Stream generation events."""
    
    async def event_generator():
        try:
            if request.mode == "optimized":
                # Full Optimization Loop
                print(f"Starting optimized generation for: {request.prompt}")
                async for event in optimizer.optimize_stream_generator(request.prompt):
                    yield f"data: {json.dumps(event)}\n\n"
                    # Add small delay to ensure frontend renders smoothly if needed
                    await asyncio.sleep(0.01)
            else:
                # Regular Mode (Nano Banana Only - No Evals/Rewrite)
                print(f"Starting regular generation for: {request.prompt}")
                yield f"data: {json.dumps({'type': 'status', 'message': 'Generating (Standard Mode)...', 'step': 'init'})}\n\n"
                
                # 1. FIBO (Still useful for structure, or fallback to simple dict)
                yield f"data: {json.dumps({'type': 'status', 'message': 'Structuring prompt...', 'step': 'fibo'})}\n\n"
                try:
                    # Use FIBO to give Nano Banana a fair chance with structure
                    json_prompt = optimizer.fibo(request.prompt)
                    yield f"data: {json.dumps({'type': 'fibo_json', 'json_prompt': json_prompt})}\n\n"
                except:
                    json_prompt = {"description": request.prompt}
                    yield f"data: {json.dumps({'type': 'fibo_json', 'json_prompt': json_prompt})}\n\n"
                
                # 2. Stream Generation
                yield f"data: {json.dumps({'type': 'status', 'message': 'Streaming images...', 'step': 'generation'})}\n\n"
                
                async for seed_idx, img in optimizer.generator.generate_stream(json_prompt, num_seeds=10):
                    # Resize/Save Temp logic duplicated from optimizer
                    # We reuse optimizer components but simplified flow
                    try:
                        eval_img = img.resize((512, 512))
                        timestamp = int(time.time())
                        temp_name = f"temp_{timestamp}_seed{seed_idx}.png"
                        temp_path = os.path.join("output/temp", temp_name)
                        os.makedirs("output/temp", exist_ok=True)
                        eval_img.save(temp_path)
                        
                        # Score? Yes, why not show score even in regular mode
                        score = optimizer.selector.score_image(eval_img)
                        
                        yield f"data: {json.dumps({'type': 'image_generated', 'seed': seed_idx, 'url': f'/static/temp/{temp_name}', 'score': score, 'is_best': False})}\n\n"
                    except Exception as e:
                        print(f"Error in regular gen: {e}")
                
                yield f"data: {json.dumps({'type': 'success', 'message': 'Generation Complete (Regular Mode)'})}\n\n"

        except Exception as e:
            print(f"Stream Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("Starting Scenenapse API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
