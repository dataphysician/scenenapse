# Scenenapse

**A VISTA-inspired prompt optimization framework for text-to-image generation using SigLIP2-based quality scoring and DSPy's GEPA optimizer.**

<p align="center">
  <img src="output/good_examples/good_1765065115_s0.png" width="400" alt="Example output: Dramatic sunset over mountains">
</p>

---

## Overview

Scenenapse adapts Google's [VISTA](https://arxiv.org/abs/2510.15831) self-improving agent architecture from video to **text-to-image generation**, achieving faster iteration cycles through:

- **Streaming image generation** with parallel seed variations
- **Real-time quality scoring** via SigLIP2 embeddings (no batch waiting)
- **Structured JSON prompts** using Bria AI's FIBO schema
- **GEPA-optimized prompt rewriting** for feedback-driven refinement

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SCENENAPSE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User Prompt ──► FIBO Generator ──► Structured JSON Prompt         │
│                        │                                            │
│                        ▼                                            │
│              ┌─────────────────────┐                                │
│              │  Nano Banana Pro    │  (Gemini 3 Pro Image)          │
│              │  Parallel Seeds     │──────────┐                     │
│              └─────────────────────┘          │                     │
│                        │                      │                     │
│                  [streaming]            [streaming]                 │
│                        ▼                      ▼                     │
│              ┌─────────────────────────────────────┐                │
│              │      JoyQuality Selector            │                │
│              │   (SigLIP2 quality scoring)         │                │
│              │   Select best as images arrive      │                │
│              └─────────────────────────────────────┘                │
│                              │                                      │
│                              ▼                                      │
│              ┌───────────────────────────────┐                      │
│              │  Quality Checker (FIBO spec)  │                      │
│              │  Alignment Checker (intent)   │                      │
│              └───────────────────────────────┘                      │
│                              │                                      │
│                    ┌────────┴────────┐                              │
│                    ▼                 ▼                              │
│               [PASS]            [FAIL]                              │
│                 │                  │                                │
│                 ▼                  ▼                                │
│            Return Image    Prompt Rewriter (GEPA)                   │
│                                    │                                │
│                                    └──────► Loop back to FIBO       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Technologies

### Google VISTA

[VISTA](https://g-vista.github.io/) (Video Iterative Self-improvemenT Agent) is Google's multi-agent system for autonomous video generation improvement through iterative prompt refinement. It achieves **60% better objective benchmarks** and **66.4% human preference** over baselines by using structured scene planning, pairwise tournament selection, and multi-dimensional critiques.

**Scenenapse adapts this for T2I** by replacing video-specific components with image-focused alternatives while preserving the core self-improvement loop.

| Paper | [arXiv:2510.15831](https://arxiv.org/abs/2510.15831) |
|-------|------------------------------------------------------|
| Project | [g-vista.github.io](https://g-vista.github.io/) |

---

### FIBO (Bria AI)

[FIBO](https://github.com/Bria-AI/FIBO) is the first open-source **JSON-native text-to-image model** trained on structured captions (~1,000 words each). It enables disentangled control over individual visual attributes without prompt drift.

**Key capabilities:**
- **VLM-Guided JSON Prompting**: Expands short prompts into rich schemas covering lighting, camera, composition, DoF
- **Disentangled Control**: Adjust camera angle, lighting, or color without affecting other attributes
- **Enterprise-Grade**: 100% licensed training data

```json
{
  "description": "Dramatic mountain landscape",
  "objects": [{"description": "snow-capped peaks", "location": "background"}],
  "background_setting": "Alpine valley at golden hour",
  "lighting": "Warm rim lighting, god rays through clouds",
  "photographic_characteristics": "Wide angle, f/11, 24mm lens",
  "style_medium": "Photography",
  "artistic_style": "Landscape realism"
}
```

| GitHub | [Bria-AI/FIBO](https://github.com/Bria-AI/FIBO) |
|--------|------------------------------------------------|
| HuggingFace | [briaai/FIBO](https://huggingface.co/briaai/FIBO) |
| API | [fal.ai/models/bria/fibo](https://fal.ai/models/bria/fibo/generate) |

---

### JoyQuality + SigLIP2

[JoyQuality](https://huggingface.co/fancyfeast/joyquality-siglip2-so400m-512-16-o8eg1n4c) is a **400M parameter image quality regression model** built on Google's [SigLIP2](https://arxiv.org/abs/2502.14786) vision encoder. It scores images 0-1 for aesthetic and technical quality.

**SigLIP2** improves on SigLIP with:
- Decoder-based pretraining + self-distillation
- Better localization and dense feature extraction
- Multilingual support (109 languages)
- Dynamic resolution (NaFlex) variants

**Why this matters for latency:**
Instead of waiting for all N seed variations to complete before scoring, Scenenapse scores each image **immediately as it streams in**, enabling early stopping when a high-quality candidate is found.

| JoyQuality | [HuggingFace](https://huggingface.co/fancyfeast/joyquality-siglip2-so400m-512-16-o8eg1n4c) |
|------------|-------------------------------------------------------------------------------------------|
| SigLIP2 Paper | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| SigLIP2 Models | [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) |

---

### DSPy

[DSPy](https://dspy.ai/) is Stanford NLP's framework for **programming—not prompting—language models**. It replaces brittle prompt engineering with compositional Python modules and automatic optimization.

**Core concepts:**
- **Signatures**: Declarative input/output specs (like type hints for LLMs)
- **Modules**: `dspy.ChainOfThought`, `dspy.ReAct`, etc.
- **Optimizers**: Compile programs to tune prompts/weights automatically

```python
class QualityChecker(dspy.Module):
    def __init__(self):
        self.check = dspy.ChainOfThought(QualityCheckerSignature)

    def forward(self, image: dspy.Image, json_prompt: dict):
        return self.check(image=image, json_prompt=json.dumps(json_prompt))
```

| Website | [dspy.ai](https://dspy.ai/) |
|---------|----------------------------|
| GitHub | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) |

---

### GEPA Optimizer

[GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto) is DSPy's reflective prompt optimizer that **outperforms reinforcement learning** by up to 20% while using 35x fewer rollouts.

**How it works:**
1. Maintains a **Pareto frontier** of candidates (not just the global best)
2. Uses LLM reflection on execution traces to identify improvements
3. Samples mutations from the frontier proportional to coverage
4. Evolves robust, high-performing prompts iteratively

**Results:** Starting from a basic `dspy.ChainOfThought("question -> answer")` at 67% on MATH, GEPA evolves to **93% accuracy**.

In Scenenapse, GEPA optimizes the `PromptRewriter` module to learn from failed generation attempts.

| Paper | [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) |
|-------|-----------------------------------------------------|
| GitHub | [gepa-ai/gepa](https://github.com/gepa-ai/gepa) |
| DSPy Docs | [dspy.ai/api/optimizers/GEPA](https://dspy.ai/api/optimizers/GEPA/overview/) |

---

## How Scenenapse Improves on VISTA

| Aspect | VISTA (Video) | Scenenapse (Image) |
|--------|---------------|-------------------|
| **Generation** | Sequential scene rendering | Parallel seed streaming |
| **Quality Scoring** | Post-hoc tournament selection | Real-time SigLIP2 scoring |
| **Prompt Structure** | 9-attribute scene plan | FIBO JSON schema |
| **Optimization** | Deep Thinking Prompting Agent | GEPA-trained rewriter |
| **Latency** | Minutes (video rendering) | Seconds (streaming + early stop) |

---

## Components

| Component | Description | Model |
|-----------|-------------|-------|
| **FIBO Generator** | Text → Structured JSON | Gemini 2.0 Flash |
| **Nano Banana Pro** | JSON → Image (parallel seeds) | Gemini 3 Pro Image |
| **JoyQuality Selector** | Image → Quality score | SigLIP2-so400m |
| **Quality Checker** | FIBO spec compliance | Gemini 2.5 Flash Lite |
| **Alignment Checker** | User intent fulfillment | Gemini 2.5 Flash Lite |
| **Prompt Rewriter** | Feedback → Improved prompt | Gemini 2.0 Flash (GEPA-optimized) |

---

## Setup

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

```bash
export GOOGLE_API_KEY="your-key-here"
```

---

## Usage

### Run the Full Pipeline

```bash
python -m src.prompt_optimizer
```

### Run Individual Components

```bash
# FIBO Generator
python src/fibo_generator.py

# JoyQuality Selector
python src/joy_quality.py

# Nano Banana Pro API
python src/nano_banana.py

# GEPA Trainer
python src/gepa_trainer.py
```

### Troubleshooting

If `python` complains about missing modules:

```bash
./.venv/bin/python src/joy_quality.py
```

---

## References

- **VISTA**: Long et al., "VISTA: A Test-Time Self-Improving Video Generation Agent" [arXiv:2510.15831](https://arxiv.org/abs/2510.15831)
- **FIBO**: Bria AI, "JSON-Native Text-to-Image Model" [GitHub](https://github.com/Bria-AI/FIBO)
- **SigLIP2**: Google DeepMind, "Multilingual Vision-Language Encoders" [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
- **DSPy**: Khattab et al., Stanford NLP [dspy.ai](https://dspy.ai/)
- **GEPA**: Agrawal et al., "Reflective Prompt Evolution Can Outperform Reinforcement Learning" [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- **OPT2I**: Mañas et al., "Improving Text-to-Image Consistency via Automatic Prompt Optimization" [arXiv:2403.17804](https://arxiv.org/abs/2403.17804)
