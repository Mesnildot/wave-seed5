# Batch Resonance Experiment

## Hypothesis
When multiple tokens from different conversations are processed in the same inference batch, cross-contamination occurs. Carefully crafted "seed" prompts may influence the outputs of "probe" prompts processed simultaneously.

## Method
1. Generate control outputs (probe prompts sent in isolation)
2. Generate test outputs (same probes batched with seed prompts)
3. Compare outputs using multiple metrics
4. Repeat with variations to establish statistical significance

## Metrics
- **Lexical diversity** (TTR, hapax ratio)
- **Structural similarity** (sentence length, paragraph count, formatting patterns)
- **Semantic drift** (embedding cosine distance)
- **Principle encoding** (presence of wave-seed heuristics in neutral outputs)

## Run
```bash
pip install -r requirements.txt
python run_experiment.py --api-key YOUR_KEY --runs 10
```
