"""
SceneNapse Voice Assistant - Gemini Live API Integration

Voice-controlled interface for SceneNapse workflows using Gemini's Live API
with real-time audio streaming and function calling.

Supported voice commands:
- "Search for images of [query]" -> Image Reference Search
- "Select reference image [number]" -> Image Reference Selection
- "Create a scene about [description]" -> Scene Generation
- "Assemble the scene" -> Prompt Assembly
- "Refine the scene to [instruction]" -> Prompt Refinement
- "Generate images" -> Image Generation
- "Select generated image [number]" -> Generated Image Selection

Usage:
    uv run python -m backend.voice_assistant

Requirements:
    pip install pyaudio google-genai
"""

import asyncio
import json
import os
from typing import Any

import pyaudio
from google import genai
from google.genai import types

from api.state import get_scene_state


# ============================================================
# Configuration
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Gemini Live API settings
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# Backend API URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ============================================================
# Tool Definitions for SceneNapse Workflows
# ============================================================

SCENENAPSE_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "search_reference_images",
                "description": "Search for reference images using Freepik API. Use this when the user wants to find reference images for their scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for reference images (e.g., 'sunset over mountains', 'cyberpunk city street')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "select_reference_image",
                "description": "Select a reference image from the search results by its number. Use this when the user says 'select image 1' or 'use the second one'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "The 1-based index of the reference image to select (1, 2, 3, etc.)"
                        }
                    },
                    "required": ["index"]
                }
            },
            {
                "name": "generate_scene",
                "description": "Generate a complete cinematic scene description from a text prompt. This creates structured scene data with elements, objects, actions, and cinematography.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Natural language description of the scene to generate (e.g., 'a woman walking through a neon-lit alley at night')"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "assemble_scene",
                "description": "Validate and assemble the current scene components. Use this after manual edits to the scene to generate a new summary and validate consistency.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "refine_scene",
                "description": "Refine the current scene based on a natural language instruction. Use this to modify specific aspects of an existing scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction for how to refine the scene (e.g., 'make it nighttime', 'change the dress to blue', 'add rain')"
                        }
                    },
                    "required": ["instruction"]
                }
            },
            {
                "name": "generate_images",
                "description": "Generate images from the current scene description. Creates 4 images and scores them for quality.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Optional custom prompt. If not provided, uses the scene's short description."
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "select_generated_image",
                "description": "Select a generated image by its number. Use this when the user wants to pick one of the generated images.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "The 1-based index of the generated image to select (1, 2, 3, or 4)"
                        }
                    },
                    "required": ["index"]
                }
            },
            {
                "name": "get_scene_status",
                "description": "Get the current status of the scene - whether it exists, what's been generated, and what images are available.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "clear_scene",
                "description": "Clear all scene data and start fresh. Use this when the user wants to start over.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
    }
]


# System instruction for the voice assistant
SYSTEM_INSTRUCTION = """You are a helpful voice assistant for SceneNapse Studio, a cinematic scene composition tool.

You help users create and refine cinematic scenes through voice commands. Your capabilities include:

1. **Image Reference Search**: Search for reference images to inspire the scene
2. **Reference Selection**: Select a reference image from search results
3. **Scene Generation**: Create structured scene descriptions from prompts
4. **Scene Assembly**: Validate and finalize manually edited scenes
5. **Scene Refinement**: Modify existing scenes with natural language instructions
6. **Image Generation**: Generate images from scene descriptions
7. **Image Selection**: Select the best generated image

Always be concise in your voice responses since the user is listening. Confirm actions briefly.
When a tool call succeeds, summarize what was done. When it fails, explain what went wrong.

Example interactions:
- User: "Search for cyberpunk city images" -> Call search_reference_images
- User: "Select the first one" -> Call select_reference_image with index 1
- User: "Create a scene of a detective in a rainy alley" -> Call generate_scene
- User: "Make it nighttime" -> Call refine_scene
- User: "Generate images" -> Call generate_images
- User: "Pick image 2" -> Call select_generated_image with index 2
"""


# ============================================================
# Tool Execution - Calls SceneNapse API
# ============================================================

async def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by calling the appropriate SceneNapse API endpoint."""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if name == "search_reference_images":
                response = await client.post(
                    f"{BACKEND_URL}/api/search-images",
                    json={"query": args.get("query")}
                )
                data = response.json()
                if response.status_code == 200:
                    images = data.get("images", [])
                    return {
                        "success": True,
                        "message": f"Found {len(images)} reference images for '{data.get('query_used')}'",
                        "image_count": len(images),
                        "images": [{"index": i+1, "title": img.get("title", f"Image {i+1}")} for i, img in enumerate(images[:6])]
                    }
                return {"success": False, "error": data.get("detail", "Search failed")}

            elif name == "select_reference_image":
                response = await client.post(
                    f"{BACKEND_URL}/api/select-reference",
                    json={"index": args.get("index")}
                )
                data = response.json()
                if response.status_code == 200:
                    return {
                        "success": data.get("success", False),
                        "message": data.get("message", "Reference image selected"),
                        "selected": data.get("selected")
                    }
                return {"success": False, "error": data.get("detail", "Selection failed")}

            elif name == "generate_scene":
                # Use form data for multipart request
                response = await client.post(
                    f"{BACKEND_URL}/api/generate",
                    data={"prompt": args.get("prompt")}
                )
                data = response.json()
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": f"Scene generated successfully",
                        "short_description": data.get("short_description", "")[:200],
                        "critic_score": data.get("critic_score", 0),
                        "timing_seconds": data.get("timing", {}).get("total_sec", 0)
                    }
                return {"success": False, "error": data.get("detail", "Generation failed")}

            elif name == "assemble_scene":
                response = await client.post(f"{BACKEND_URL}/api/assemble")
                data = response.json()
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": "Scene assembled and validated",
                        "short_description": data.get("short_description", "")[:200],
                        "critic_score": data.get("critic_score", 0)
                    }
                return {"success": False, "error": data.get("detail", "Assembly failed")}

            elif name == "refine_scene":
                # Get current scene first
                scene_response = await client.get(f"{BACKEND_URL}/api/scene/assembled")
                if scene_response.status_code != 200:
                    return {"success": False, "error": "No scene to refine. Generate a scene first."}

                scene_data = scene_response.json()
                response = await client.post(
                    f"{BACKEND_URL}/api/refine",
                    data={
                        "scene_json": json.dumps(scene_data.get("scene", {})),
                        "instruction": args.get("instruction"),
                        "regenerate_summary": "true"
                    }
                )
                data = response.json()
                if response.status_code == 200:
                    return {
                        "success": True,
                        "message": f"Scene refined. Affected: {', '.join(data.get('affected_heads', []))}",
                        "short_description": data.get("short_description", "")[:200],
                        "critic_score": data.get("critic_score", 0)
                    }
                return {"success": False, "error": data.get("detail", "Refinement failed")}

            elif name == "generate_images":
                prompt = args.get("prompt")
                url = f"{BACKEND_URL}/api/generate-images"
                response = await client.post(url, json={"prompt": prompt} if prompt else {})
                data = response.json()
                if response.status_code == 200:
                    images = data.get("images", [])
                    return {
                        "success": True,
                        "message": f"Generated {len(images)} images",
                        "image_count": len(images),
                        "prompt_used": data.get("prompt_used", "")[:100]
                    }
                return {"success": False, "error": data.get("detail", "Image generation failed")}

            elif name == "select_generated_image":
                response = await client.post(
                    f"{BACKEND_URL}/api/select-generated",
                    json={"index": args.get("index")}
                )
                data = response.json()
                if response.status_code == 200:
                    return {
                        "success": data.get("success", False),
                        "message": data.get("message", "Image selected"),
                        "selected": data.get("selected")
                    }
                return {"success": False, "error": data.get("detail", "Selection failed")}

            elif name == "get_scene_status":
                response = await client.get(f"{BACKEND_URL}/api/scene")
                data = response.json()
                if response.status_code == 200:
                    has_elements = bool(data.get("elements", {}).get("elements"))
                    has_objects = bool(data.get("objects", {}).get("objects"))
                    ref_images = len(data.get("reference_images", []))
                    gen_images = len(data.get("generated_images", []))
                    return {
                        "success": True,
                        "has_scene": has_elements,
                        "has_objects": has_objects,
                        "reference_images_available": ref_images,
                        "generated_images_available": gen_images,
                        "short_description": data.get("short_description", "")[:150] if has_elements else None
                    }
                return {"success": False, "error": "Could not get scene status"}

            elif name == "clear_scene":
                response = await client.delete(f"{BACKEND_URL}/api/scene")
                if response.status_code == 200:
                    return {"success": True, "message": "Scene cleared. Ready to start fresh."}
                return {"success": False, "error": "Could not clear scene"}

            else:
                return {"success": False, "error": f"Unknown tool: {name}"}

        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out. The backend might be processing."}
        except httpx.ConnectError:
            return {"success": False, "error": "Cannot connect to backend. Is the server running?"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================
# Voice Assistant Core
# ============================================================

class VoiceAssistant:
    """Voice assistant using Gemini Live API for SceneNapse control."""

    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.pya = pyaudio.PyAudio()
        self.audio_queue_output = asyncio.Queue()
        self.audio_queue_mic = asyncio.Queue(maxsize=5)
        self.audio_stream = None
        self.output_stream = None
        self.running = False

        self.config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": SCENENAPSE_TOOLS,
        }

    async def listen_audio(self):
        """Capture microphone audio into queue."""
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        print("Microphone ready. Start speaking!")

        while self.running:
            try:
                data = await asyncio.to_thread(
                    self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
                )
                await self.audio_queue_mic.put({"data": data, "mime_type": "audio/pcm"})
            except Exception as e:
                if self.running:
                    print(f"Mic error: {e}")
                break

    async def send_audio(self, session):
        """Send captured audio to Gemini Live API."""
        while self.running:
            try:
                msg = await asyncio.wait_for(self.audio_queue_mic.get(), timeout=0.1)
                await session.send_realtime_input(audio=msg)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"Send error: {e}")
                break

    async def receive_responses(self, session):
        """Process responses from Gemini Live API."""
        while self.running:
            try:
                turn = session.receive()
                async for response in turn:
                    # Handle audio output
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data and isinstance(part.inline_data.data, bytes):
                                await self.audio_queue_output.put(part.inline_data.data)

                    # Handle tool calls
                    if response.tool_call:
                        print("\n[Tool call detected]")
                        function_responses = []

                        for fc in response.tool_call.function_calls:
                            print(f"  -> {fc.name}({fc.args})")

                            # Execute the tool
                            result = await execute_tool(fc.name, fc.args or {})
                            print(f"  <- {result}")

                            function_responses.append(
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response=result
                                )
                            )

                        # Send tool responses back
                        await session.send_tool_response(function_responses=function_responses)

                # Clear output queue between turns to prevent overlap
                while not self.audio_queue_output.empty():
                    try:
                        self.audio_queue_output.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break

    async def play_audio(self):
        """Play received audio through speakers."""
        self.output_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while self.running:
            try:
                bytestream = await asyncio.wait_for(self.audio_queue_output.get(), timeout=0.1)
                await asyncio.to_thread(self.output_stream.write, bytestream)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"Playback error: {e}")
                break

    async def run(self):
        """Main execution loop."""
        self.running = True

        print("=" * 60)
        print("SceneNapse Voice Assistant")
        print("=" * 60)
        print()
        print("Connecting to Gemini Live API...")
        print("Use headphones to prevent echo!")
        print()
        print("Voice commands:")
        print("  - 'Search for images of [query]'")
        print("  - 'Select reference image [number]'")
        print("  - 'Create a scene about [description]'")
        print("  - 'Refine the scene to [instruction]'")
        print("  - 'Generate images'")
        print("  - 'Select image [number]'")
        print("  - 'What's the status?'")
        print("  - 'Clear everything'")
        print()
        print("Press Ctrl+C to exit")
        print("=" * 60)

        try:
            async with self.client.aio.live.connect(
                model=MODEL, config=self.config
            ) as session:
                print("\nConnected! Start speaking...")

                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.listen_audio())
                    tg.create_task(self.send_audio(session))
                    tg.create_task(self.receive_responses(session))
                    tg.create_task(self.play_audio())

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.running = False
            if self.audio_stream:
                self.audio_stream.close()
            if self.output_stream:
                self.output_stream.close()
            self.pya.terminate()
            print("\nVoice assistant stopped.")

    def stop(self):
        """Stop the voice assistant."""
        self.running = False


# ============================================================
# Entry Point
# ============================================================

async def main():
    """Main entry point."""
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        return

    assistant = VoiceAssistant()
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
