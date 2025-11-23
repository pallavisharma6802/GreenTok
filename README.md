# Carbon-Aware Conversations — CAPO (prototype)

CAPO is a small toolkit that trims unnecessary words from prompts before they go to an LLM. The goal is simple: send fewer tokens, use less compute, and reduce emissions — without changing how people work.

What this repo contains

- A tiny CLI prototype that runs two steps on a prompt: a rule-based cleaner (remove polite scaffolding and filler) and an extractive summarizer (pick the most important sentences).
- A configurable `fillers.json` so you can control which phrases to strip.
- A lightweight CO2 estimate that converts saved tokens → Wh → grams CO2e for quick, local reporting.

Why this matters

Large models charge compute per token. Shortening prompts by even a modest percentage saves energy across many requests. CAPO shows you a practical way to reduce waste without changing models or user workflows.

Quick results (example)

- Prototype runs have reduced input size by roughly 30–60% on many longer prompts.
- This is a prototype — real savings depend on traffic, prompt style, and infra.

How it works (high level)

1. Rule-based stripping: remove polite phrases and filler using configurable regexes.
2. Extractive summarization: pick the most representative 1–2 sentences from the cleaned prompt using a small sentence-transformer model.

Try the prototype

1. Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the CLI and paste or pipe a prompt:

```bash
python main.py
# or
printf 'Your prompt here\n' | python main.py
```

3. Edit `fillers.json` to tune the rule-based step.

Notes and next steps

- This repo is a starting point: improvements include better compression models, semantic-safety checks to keep important facts, and a batch evaluation harness.


