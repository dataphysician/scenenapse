# SceneNapse Studio - Demo Script (TTS)

**Duration:** ~60 seconds

---

## Script

<break time="0.5s"/>

Welcome to SceneNapse Studio! <break time="0.3s"/>

We're solving a real problem with text-to-image generation: <break time="0.2s"/> prompts are messy, unstructured, and when the image doesn't match? <break time="0.2s"/> You have <emphasis level="moderate">no idea</emphasis> what went wrong.

<break time="0.5s"/>

SceneNapse fixes this with a <emphasis level="strong">two-phase pipeline</emphasis>.

<break time="0.3s"/>

<emphasis level="moderate">Phase one</emphasis> — Prompt Enhancement. <break time="0.2s"/>

You give us a simple prompt like <break time="0.1s"/> "a woman walking through a neon city at night." <break time="0.3s"/>

Our DSPy pipeline decomposes this into four structured heads: <break time="0.2s"/> Elements — who exists. <break time="0.2s"/> Objects — what they look like. <break time="0.2s"/> Actions — what they're doing. <break time="0.2s"/> And Cinematography — camera, lighting, mood.

<break time="0.5s"/>

<emphasis level="moderate">Phase two</emphasis> — Generation and Validation. <break time="0.2s"/>

We generate multiple images using Nano Banana Pro. <break time="0.3s"/> Each image gets scored by JoyQuality — a pairwise preference-trained encoder that predicts aesthetic and technical quality. <break time="0.3s"/> Then our VLM Guardrails verify the image <emphasis level="moderate">actually matches</emphasis> what you asked for.

<break time="0.5s"/>

Here's where it gets powerful. <break time="0.2s"/>

Want to change something? <break time="0.3s"/> Just say "make it golden hour lighting" — and <emphasis level="moderate">only</emphasis> the cinematography head regenerates. <break time="0.2s"/> Ninety percent of refinements touch just one component. <break time="0.2s"/> Fast, cheap, and predictable.

<break time="0.5s"/>

We also integrated Freepik's semantic search for reference images, <break time="0.2s"/> and — bonus — you can control the entire workflow with your voice using Gemini's Live API.

<break time="0.5s"/>

<emphasis level="strong">SceneNapse</emphasis> — composable prompts, validated generations, and real control over your creative vision.

<break time="0.3s"/>

Thank you!

---

## ElevenLabs Settings

- **Voice:** Choose a clear, energetic voice (e.g., "Josh" or "Rachel")
- **Stability:** 0.5 (natural variation)
- **Clarity:** 0.75 (clear enunciation)
- **Style:** 0.3 (slight enthusiasm)

---

## Plain Text Version (no tags)

Welcome to SceneNapse Studio!

We're solving a real problem with text-to-image generation: prompts are messy, unstructured, and when the image doesn't match? You have no idea what went wrong.

SceneNapse fixes this with a two-phase pipeline.

Phase one — Prompt Enhancement.

You give us a simple prompt like "a woman walking through a neon city at night."

Our DSPy pipeline decomposes this into four structured heads: Elements — who exists. Objects — what they look like. Actions — what they're doing. And Cinematography — camera, lighting, mood.

Phase two — Generation and Validation.

We generate multiple images using Nano Banana Pro. Each image gets scored by JoyQuality — a pairwise preference-trained encoder that predicts aesthetic and technical quality. Then our VLM Guardrails verify the image actually matches what you asked for.

Here's where it gets powerful.

Want to change something? Just say "make it golden hour lighting" — and only the cinematography head regenerates. Ninety percent of refinements touch just one component. Fast, cheap, and predictable.

We also integrated Freepik's semantic search for reference images, and — bonus — you can control the entire workflow with your voice using Gemini's Live API.

SceneNapse — composable prompts, validated generations, and real control over your creative vision.

Thank you!
