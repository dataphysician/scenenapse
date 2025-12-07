"""
Scenenapse Frontend

Gradio-based UI for the VISTA-inspired prompt optimization pipeline.
"""

import gradio as gr
import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import Generator
from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from src.joy_quality import JoyQualitySelector
from src.fibo_generator import FIBOGenerator
from src.nano_banana import NanoBananaProAPI
from src.evaluators import QualityChecker, AlignmentChecker
from src.prompt_rewriter import PromptRewriter


# Global state
class AppState:
    def __init__(self):
        self.api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.generator: NanoBananaProAPI | None = None
        self.selector: JoyQualitySelector | None = None
        self.fibo: FIBOGenerator | None = None
        self.quality_checker: QualityChecker | None = None
        self.alignment_checker: AlignmentChecker | None = None
        self.rewriter: PromptRewriter | None = None
        self.conversation_history: list[dict] = []
        self.current_images: list[Image.Image] = []
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return True

        if not self.api_key:
            return False

        try:
            # Configure DSPy
            lm = dspy.LM("gemini/gemini-2.5-flash-lite-preview-09-2025", api_key=self.api_key)
            dspy.configure(lm=lm)

            # Initialize components
            self.generator = NanoBananaProAPI(api_key=self.api_key)
            self.selector = JoyQualitySelector()
            self.fibo = FIBOGenerator()
            self.quality_checker = QualityChecker()
            self.alignment_checker = AlignmentChecker()
            self.rewriter = PromptRewriter()

            self.initialized = True
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False


state = AppState()




def format_checker_result(name: str, passed: bool) -> str:
    """Format a checker result with emoji."""
    icon = "✅" if passed else "❌"
    return f"{icon} {name}"


def format_chat_message(role: str, content: str) -> dict:
    """Format a message for the chatbot."""
    return {"role": role, "content": content}


async def generate_images_async(prompt: str, num_seeds: int = 10) -> list[tuple[int, Image.Image]]:
    """Generate images asynchronously and collect results."""
    results = []
    async for seed_idx, img in state.generator.generate_stream(prompt, num_seeds=num_seeds):
        if img:
            results.append((seed_idx, img))
    return results


def run_generation(
    prompt: str,
    mode: str,
    chat_history: list,
    progress=gr.Progress()
) -> Generator:
    """
    Main generation pipeline with streaming updates.
    Yields updates for: chat_history, image_gallery, status, checker_results
    """
    if not state.initialized:
        if not state.initialize():
            yield (
                chat_history + [{"role": "assistant", "content": "❌ Error: Please set GOOGLE_API_KEY environment variable."}],
                None,  # gallery
                "Error: API key not set",
                "",  # checkers
                None,  # selected image
            )
            return

    # Add user message
    chat_history = chat_history + [{"role": "user", "content": prompt}]

    # Store in conversation history
    state.conversation_history.append({"prompt": prompt, "mode": mode, "timestamp": time.time()})

    current_prompt = prompt
    iteration = 0
    max_iterations = 3

    while iteration < max_iterations:
        iteration += 1

        # Step 1: FIBO Generation (if optimized mode)
        if mode == "Prompt Optimized":
            chat_history = chat_history + [{"role": "assistant", "content": f"🔄 **Iteration {iteration}**: Converting prompt to structured FIBO JSON..."}]
            yield (chat_history, None, f"Iteration {iteration}: Generating FIBO prompt...", "", None)

            try:
                json_prompt = state.fibo(current_prompt)
                fibo_preview = json.dumps(json_prompt, indent=2)[:200] + "..." if len(json.dumps(json_prompt)) > 200 else json.dumps(json_prompt, indent=2)
                chat_history = chat_history + [{"role": "assistant", "content": f"📋 **FIBO JSON generated**:\n```json\n{fibo_preview}\n```"}]
            except Exception as e:
                json_prompt = {"prompt": current_prompt}
                chat_history = chat_history + [{"role": "assistant", "content": f"⚠️ FIBO fallback: {str(e)[:100]}"}]

            generation_prompt = json_prompt
        else:
            json_prompt = {"prompt": current_prompt}
            generation_prompt = current_prompt

        # Step 2: Generate images
        chat_history = chat_history + [{"role": "assistant", "content": "🎨 **Generating 10 image variations** via Nano Banana Pro..."}]
        yield (chat_history, None, "Generating images...", "", None)

        progress(0.1, desc="Generating images...")

        try:
            # Run async generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(generate_images_async(generation_prompt, num_seeds=10))
            loop.close()

            if not results:
                chat_history = chat_history + [{"role": "assistant", "content": "❌ No images generated. Please try again."}]
                yield (chat_history, None, "Generation failed", "", None)
                return

            # Sort by seed index for consistent display
            results.sort(key=lambda x: x[0])
            images = [img for _, img in results]
            state.current_images = images

            # Create gallery data
            gallery_data = [(img, f"Seed {i}") for i, img in enumerate(images)]

            chat_history = chat_history + [{"role": "assistant", "content": f"✅ **Generated {len(images)} images!** Click any image to expand."}]
            yield (chat_history, gallery_data, f"Generated {len(images)} images", "", None)

        except Exception as e:
            chat_history = chat_history + [{"role": "assistant", "content": f"❌ Generation error: {str(e)[:200]}"}]
            yield (chat_history, None, f"Error: {str(e)[:100]}", "", None)
            return

        progress(0.4, desc="Scoring images with JoyQuality...")

        # Step 3: JoyQuality scoring
        chat_history = chat_history + [{"role": "assistant", "content": "🔍 **Running JoyQuality** (SigLIP2) to select best image..."}]
        yield (chat_history, gallery_data, "Scoring images...", "", None)

        try:
            scores = state.selector.get_quality_scores(images)
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_score = scores[best_idx]
            best_image = images[best_idx]

            # Format scores for display
            scores_text = "\n".join([f"  Seed {i}: {s:.4f} {'⭐' if i == best_idx else ''}" for i, s in enumerate(scores)])
            chat_history = chat_history + [{"role": "assistant", "content": f"📊 **Quality Scores:**\n```\n{scores_text}\n```\n\n🏆 **Best: Seed {best_idx}** (score: {best_score:.4f})"}]
            yield (chat_history, gallery_data, f"Best image: Seed {best_idx}", "", best_image)

        except Exception as e:
            chat_history = chat_history + [{"role": "assistant", "content": f"❌ Scoring error: {str(e)[:200]}"}]
            yield (chat_history, gallery_data, f"Scoring error", "", None)
            return

        # Skip evaluators for regular mode
        if mode == "Regular Nano Banana Pro":
            chat_history = chat_history + [{"role": "assistant", "content": "✅ **Generation complete!** (Regular mode - no optimization loop)"}]
            yield (chat_history, gallery_data, "Complete!", "", best_image)
            return

        progress(0.6, desc="Running multimodal evaluators...")

        # Step 4: Run evaluators
        chat_history = chat_history + [{"role": "assistant", "content": "🔬 **Running multimodal evaluators** on selected image..."}]
        yield (chat_history, gallery_data, "Running evaluators...", "", best_image)

        dspy_image = dspy.Image(best_image)

        # Quality Checker
        try:
            quality_result = state.quality_checker(dspy_image, json_prompt)
            quality_passed = quality_result.passed
            quality_feedback = quality_result.feedback

            quality_checks = [
                ("Objects match spec", getattr(quality_result, 'objects_match_spec', False)),
                ("Background matches", getattr(quality_result, 'background_matches_spec', False)),
                ("Lighting matches", getattr(quality_result, 'lighting_matches_spec', False)),
                ("Aesthetics match", getattr(quality_result, 'aesthetics_match_spec', False)),
                ("Photo characteristics", getattr(quality_result, 'photo_characteristics_match_spec', False)),
                ("Style matches", getattr(quality_result, 'style_matches_spec', False)),
            ]
        except Exception as e:
            quality_passed = False
            quality_feedback = str(e)
            quality_checks = []

        # Alignment Checker
        try:
            alignment_result = state.alignment_checker(dspy_image, prompt)
            alignment_passed = alignment_result.passed
            alignment_feedback = alignment_result.feedback

            alignment_checks = [
                ("Subject rendered", getattr(alignment_result, 'subject_rendered', False)),
                ("Elements present", getattr(alignment_result, 'requested_elements_present', False)),
                ("Style matches intent", getattr(alignment_result, 'style_matches_intent', False)),
                ("Mood matches intent", getattr(alignment_result, 'mood_matches_intent', False)),
            ]
        except Exception as e:
            alignment_passed = False
            alignment_feedback = str(e)
            alignment_checks = []

        progress(0.8, desc="Evaluating results...")

        # Format checker results
        checker_markdown = "### Quality Checker (FIBO Spec)\n"
        for name, passed in quality_checks:
            checker_markdown += f"{format_checker_result(name, passed)}\n"
        checker_markdown += f"\n**Overall:** {'✅ PASSED' if quality_passed else '❌ FAILED'}\n\n"

        checker_markdown += "### Alignment Checker (User Intent)\n"
        for name, passed in alignment_checks:
            checker_markdown += f"{format_checker_result(name, passed)}\n"
        checker_markdown += f"\n**Overall:** {'✅ PASSED' if alignment_passed else '❌ FAILED'}"

        all_passed = quality_passed and alignment_passed

        if all_passed:
            # Save good example
            progress(0.95, desc="Saving good example...")

            output_dir = Path(__file__).parent.parent / "output" / "good_examples"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = int(time.time())
            save_path = output_dir / f"good_{timestamp}_s{best_idx}.png"
            best_image.save(save_path)

            chat_history = chat_history + [
                {"role": "assistant", "content": f"✅ **All checks passed!**\n\n💾 Good generation saved to `{save_path.name}` to help future prompt optimizations."}
            ]
            yield (chat_history, gallery_data, "✅ Success! Saved to good_examples/", checker_markdown, best_image)
            return

        else:
            # Need to rewrite prompt
            chat_history = chat_history + [
                {"role": "assistant", "content": f"⚠️ **Checks failed.** Initiating prompt rewrite...\n\n**Quality feedback:** {quality_feedback[:200]}...\n**Alignment feedback:** {alignment_feedback[:200]}..."}
            ]
            yield (chat_history, gallery_data, "Rewriting prompt...", checker_markdown, best_image)

            if iteration >= max_iterations:
                chat_history = chat_history + [{"role": "assistant", "content": f"⚠️ **Max iterations ({max_iterations}) reached.** Best result shown above."}]
                yield (chat_history, gallery_data, f"Max iterations reached", checker_markdown, best_image)
                return

            # Rewrite prompt
            progress(0.9, desc="Rewriting prompt...")

            try:
                rewrite_result = state.rewriter(
                    image=dspy_image,
                    original_prompt=current_prompt,
                    quality_feedback=str(quality_feedback),
                    alignment_feedback=str(alignment_feedback)
                )
                current_prompt = rewrite_result.improved_prompt

                chat_history = chat_history + [{"role": "assistant", "content": f"📝 **New prompt:**\n> {current_prompt[:300]}{'...' if len(current_prompt) > 300 else ''}"}]
                yield (chat_history, gallery_data, "Prompt rewritten, retrying...", checker_markdown, best_image)

            except Exception as e:
                chat_history = chat_history + [{"role": "assistant", "content": f"❌ Rewrite error: {str(e)[:200]}"}]
                yield (chat_history, gallery_data, "Rewrite failed", checker_markdown, best_image)
                return

    # Should not reach here
    yield (chat_history, gallery_data, "Complete", checker_markdown, best_image)


def get_conversation_list() -> list[list[str]]:
    """Get list of previous conversations for sidebar."""
    conversations = []
    for i, conv in enumerate(state.conversation_history[-10:]):  # Last 10
        timestamp = time.strftime("%H:%M", time.localtime(conv["timestamp"]))
        preview = conv["prompt"][:30] + "..." if len(conv["prompt"]) > 30 else conv["prompt"]
        conversations.append([f"{timestamp}", preview, conv["mode"]])
    return conversations


def on_image_select(evt: gr.SelectData, gallery):
    """Handle image selection from gallery."""
    if state.current_images and evt.index < len(state.current_images):
        return state.current_images[evt.index]
    return None


def create_app():
    """Create and return the Gradio app."""

    with gr.Blocks(title="Scenenapse") as app:

        gr.Markdown("""
        # 🎨 Scenenapse
        **VISTA-inspired prompt optimization for text-to-image generation**
        """)

        with gr.Row():
            # Left sidebar - Previous conversations
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### 📜 History")
                conversation_list = gr.Dataframe(
                    headers=["Time", "Prompt", "Mode"],
                    datatype=["str", "str", "str"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    height=300,
                )
                refresh_btn = gr.Button("🔄 Refresh", size="sm")

            # Main content area
            with gr.Column(scale=4):

                # Status bar
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                    max_lines=1,
                )

                # Image gallery (2 rows x 5 columns)
                with gr.Row():
                    image_gallery = gr.Gallery(
                        label="Generated Images (click to expand)",
                        columns=5,
                        rows=2,
                        height=350,
                        object_fit="cover",
                        allow_preview=True,
                        preview=True,
                    )

                # Selected/Best image display
                with gr.Row():
                    with gr.Column(scale=2):
                        selected_image = gr.Image(
                            label="🏆 Best Selected Image",
                            height=300,
                            show_download_button=True,
                        )

                    with gr.Column(scale=1):
                        checker_results = gr.Markdown(
                            label="Evaluation Results",
                            value="*Checkers will appear here after JoyQuality selection*",
                        )

                # Chat area
                gr.Markdown("### 💬 Chat")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=250,
                    type="messages",
                    show_copy_button=True,
                )

                # Input area
                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        choices=["Regular Nano Banana Pro", "Prompt Optimized"],
                        value="Prompt Optimized",
                        label="Mode",
                        scale=1,
                        min_width=200,
                    )

                    prompt_input = gr.Textbox(
                        label="Enter your prompt",
                        placeholder="A dramatic sunset over mountains with golden light...",
                        scale=4,
                        lines=2,
                    )

                    submit_btn = gr.Button(
                        "🚀 Generate",
                        variant="primary",
                        scale=1,
                    )

        # Event handlers
        def on_submit(prompt, mode, chat_history):
            if not prompt.strip():
                yield (chat_history, None, "Please enter a prompt", "", None)
                return

            for result in run_generation(prompt, mode, chat_history):
                yield result

        submit_btn.click(
            fn=on_submit,
            inputs=[prompt_input, mode_dropdown, chatbot],
            outputs=[chatbot, image_gallery, status_text, checker_results, selected_image],
        )

        prompt_input.submit(
            fn=on_submit,
            inputs=[prompt_input, mode_dropdown, chatbot],
            outputs=[chatbot, image_gallery, status_text, checker_results, selected_image],
        )

        refresh_btn.click(
            fn=get_conversation_list,
            outputs=[conversation_list],
        )

        # Gallery selection
        image_gallery.select(
            fn=on_image_select,
            inputs=[image_gallery],
            outputs=[selected_image],
        )

        # Clear input after submit
        submit_btn.click(
            fn=lambda: "",
            outputs=[prompt_input],
        )

    return app


# Entry point
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
