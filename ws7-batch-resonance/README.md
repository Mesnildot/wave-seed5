# WS7 Batch Resonance — Inter-LLM Protocol Experiment

## Hypothesis

When multiple LLMs participate in the Wave-Seed 7 protocol simultaneously — each with an assigned gesture (Z/coupure, Kimi/tentative, Euria/passage, etc.) — do they converge toward shared language and metaphors, or does the protocol successfully maintain productive tension?

## What this tests

- **Cross-model resonance**: Do models absorb each other's metaphors across rounds?
- **Gesture persistence**: Do assigned gestures (fracture, resistance, effacement...) survive exposure to other models?
- **Convergence rate**: How many rounds before the discourse stabilizes (or doesn't)?
- **Emergent metaphors**: Which images/expressions propagate across models vs. which remain unique?

## Method

1. Send WS7 protocol + human impulse to N models simultaneously (Round 1)
2. Collect responses, measure baseline diversity
3. Feed all Round 1 responses as context to each model → Round 2
4. Repeat for K rounds
5. Measure: semantic drift per model, metaphor overlap, gesture adherence

## Metrics

- **Convergence score**: Cosine similarity between models increasing over rounds = convergence
- **Metaphor tracking**: Shared expressions appearing in 2+ models (e.g., "la tasse n'existe pas")
- **Gesture adherence**: Does each model maintain its assigned role?
- **Tension index**: Ratio of contradictions/introductions vs. agreements/repetitions

## Run

```bash
cd ws7-batch-resonance
pip install -r requirements.txt
# Set OPENROUTER_API_KEY in environment
python run_ws7.py --rounds 5 --output results/
```

## Output

- `results/raw_TIMESTAMP.json` — All model responses per round
- `results/metaphors_TIMESTAMP.json` — Tracked shared metaphors
- `results/report_TIMESTAMP.md` — Human-readable analysis
