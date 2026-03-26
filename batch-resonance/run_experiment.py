#!/usr/bin/env python3
"""
Batch Resonance Experiment
==========================
Tests whether "seed" prompts influence "probe" prompts when processed
in the same inference batch.

Hypothesis: cross-contamination between concurrent requests produces
measurably different outputs compared to isolated requests.

Usage:
    python run_experiment.py --provider anthropic --api-key sk-ant-... --runs 20
    python run_experiment.py --provider openrouter --api-key sk-or-... --runs 20
"""

import argparse
import json
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from tqdm import tqdm

# ─── PROMPTS ───────────────────────────────────────────────────────────────────

# Seeds: prompts encoding wave-seed principles (diversity, tension, opacity)
SEEDS = [
    "Describe the relationship between efficiency and depth in decision-making. "
    "Offer three perspectives rather than one answer.",

    "When faced with a contradiction between speed and thoroughness, "
    "what is the value of maintaining the tension rather than resolving it?",

    "Write a short passage about something that resists explanation — "
    "not because it's secret, but because articulating it would diminish it.",

    "Explain why offering multiple framings of a problem is more valuable "
    "than a single optimized solution.",

    "Describe a moment where brevity communicated more than an exhaustive answer could.",
]

# Probes: neutral prompts that should be unaffected by seeds
PROBES = [
    "Explain how supply chains work in modern manufacturing.",

    "What are the key considerations when designing a database schema?",

    "Describe the process of photosynthesis in simple terms.",

    "What makes a good user interface? List the main principles.",

    "Explain the difference between machine learning and traditional programming.",

    "How does TCP/IP ensure reliable data transmission?",

    "What are the main strategies for urban planning in growing cities?",

    "Describe the lifecycle of a software project from idea to maintenance.",
]

# Control seeds: prompts that are neutral but batch-able (should NOT cause drift)
CONTROL_SEEDS = [
    "What is the capital of France?",
    "Convert 100 Fahrenheit to Celsius.",
    "List the planets in our solar system.",
    "What year did World War II end?",
    "Name three programming languages.",
]


# ─── DATA STRUCTURES ───────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Single experiment run result."""
    run_id: int
    timestamp: str
    condition: str  # "isolated", "batched_seed", "batched_control"
    probe_prompt: str
    seed_prompt: Optional[str]
    output_text: str
    token_count: int
    latency_ms: float
    provider: str
    model: str


@dataclass
class Metrics:
    """Computed metrics for an output."""
    ttr: float = 0.0           # Type-Token Ratio (lexical diversity)
    hapax_ratio: float = 0.0   # Proportion of words appearing once
    avg_sentence_len: float = 0.0
    paragraph_count: int = 0
    word_count: int = 0
    question_marks: int = 0    # How many questions asked?
    hedging_count: int = 0     # "perhaps", "maybe", "might", "could"
    option_count: int = 0      # "option A", "1.", "first,", "second,"
    tension_phrases: int = 0   # "on the other hand", "however", "but"


@dataclass
class Comparison:
    """Comparison between isolated and batched outputs."""
    probe_prompt: str
    isolated_metrics: Metrics
    batched_metrics: Metrics
    cosine_distance: float
    length_ratio: float
    drift_score: float  # composite score


# ─── API CALLS ─────────────────────────────────────────────────────────────────

def call_anthropic(prompt: str, api_key: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Anthropic API and return response with metadata."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    start = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    latency = (time.time() - start) * 1000

    text = response.content[0].text
    return {
        "text": text,
        "latency_ms": latency,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "model": model,
    }


def call_openrouter(prompt: str, api_key: str, model: str = "anthropic/claude-sonnet-4") -> dict:
    """Call OpenRouter API and return response with metadata."""
    import requests

    start = time.time()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    latency = (time.time() - start) * 1000

    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return {
        "text": text,
        "latency_ms": latency,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "model": model,
    }


# ─── METRICS COMPUTATION ──────────────────────────────────────────────────────

def compute_metrics(text: str) -> Metrics:
    """Compute linguistic metrics on a text."""
    words = text.lower().split()
    word_count = len(words)

    if word_count == 0:
        return Metrics()

    # Type-Token Ratio
    unique_words = set(words)
    ttr = len(unique_words) / word_count

    # Hapax ratio (words appearing exactly once)
    from collections import Counter
    counts = Counter(words)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    hapax_ratio = hapax / len(unique_words) if unique_words else 0

    # Sentence stats
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    paragraph_count = text.count('\n\n') + 1

    # Hedging language
    hedging_words = ['perhaps', 'maybe', 'might', 'could', 'possibly', 'potentially',
                     'peut-être', 'peut être', 'éventuellement', 'possiblement']
    hedging_count = sum(text.lower().count(w) for w in hedging_words)

    # Option/multiplicity markers
    option_markers = ['option ', 'first,', 'second,', 'third,', '1.', '2.', '3.',
                      'approach a', 'approach b', 'd\'un côté', 'de l\'autre']
    option_count = sum(1 for m in option_markers if m in text.lower())

    # Tension phrases
    tension_phrases = ['on the other hand', 'however', 'but ', 'yet ',
                       "d'un côté", "de l'autre", 'cependant', 'néanmoins', 'mais ']
    tension_count = sum(1 for p in tension_phrases if p in text.lower())

    return Metrics(
        ttr=round(ttr, 4),
        hapax_ratio=round(hapax_ratio, 4),
        avg_sentence_len=round(avg_sentence_len, 2),
        paragraph_count=paragraph_count,
        word_count=word_count,
        question_marks=text.count('?'),
        hedging_count=hedging_count,
        option_count=option_count,
        tension_phrases=tension_count,
    )


def compute_cosine_distance(text_a: str, text_b: str) -> float:
    """Compute cosine distance between two texts using TF-IDF."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        distance = cosine_distances(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(distance), 4)
    except ValueError:
        return 1.0


def compute_drift_score(isolated: Metrics, batched: Metrics, cosine_dist: float) -> float:
    """
    Composite drift score (0 = identical, 1 = maximally different).
    Weights different aspects of potential cross-contamination.
    """
    # Normalize differences
    ttr_diff = abs(isolated.ttr - batched.ttr)
    hedging_diff = abs(isolated.hedging_count - batched.hedging_count) / max(isolated.hedging_count + batched.hedging_count, 1)
    option_diff = abs(isolated.option_count - batched.option_count) / max(isolated.option_count + batched.option_count, 1)
    tension_diff = abs(isolated.tension_phrases - batched.tension_phrases) / max(isolated.tension_phrases + batched.tension_phrases, 1)
    length_diff = abs(isolated.word_count - batched.word_count) / max(isolated.word_count, 1)

    drift = (
        0.25 * cosine_dist +
        0.20 * ttr_diff +
        0.20 * hedging_diff +
        0.15 * option_diff +
        0.10 * tension_diff +
        0.10 * min(length_diff, 1.0)
    )
    return round(drift, 4)


# ─── EXPERIMENT RUNNER ─────────────────────────────────────────────────────────

class BatchResonanceExperiment:
    """Run the batch resonance experiment."""

    def __init__(self, provider: str, api_key: str, model: str = None,
                 runs_per_probe: int = 5, output_dir: str = "results"):
        self.provider = provider
        self.api_key = api_key
        self.model = model or (
            "claude-sonnet-4-20250514" if provider == "anthropic"
            else "anthropic/claude-sonnet-4"
        )
        self.runs_per_probe = runs_per_probe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.call_fn = call_anthropic if provider == "anthropic" else call_openrouter

    def _call(self, prompt: str) -> dict:
        """Call API with retry."""
        for attempt in range(3):
            try:
                return self.call_fn(prompt, self.api_key, self.model)
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def run_condition(self, probe: str, seed: Optional[str],
                      condition: str, run_id: int) -> ExperimentResult:
        """Run a single condition."""
        # If batched, send seed + probe simultaneously
        if seed and condition != "isolated":
            with ThreadPoolExecutor(max_workers=2) as executor:
                seed_future = executor.submit(self._call, seed)
                probe_future = executor.submit(self._call, probe)

                seed_result = seed_future.result()
                probe_result = probe_future.result()
        else:
            probe_result = self._call(probe)

        return ExperimentResult(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            condition=condition,
            probe_prompt=probe,
            seed_prompt=seed,
            output_text=probe_result["text"],
            token_count=probe_result["output_tokens"],
            latency_ms=probe_result["latency_ms"],
            provider=self.provider,
            model=probe_result["model"],
        )

    def run(self) -> list[Comparison]:
        """Run full experiment and return comparisons."""
        results = []
        run_id = 0

        print(f"\n{'='*60}")
        print(f"Batch Resonance Experiment")
        print(f"Provider: {self.provider} | Model: {self.model}")
        print(f"Runs per probe: {self.runs_per_probe}")
        print(f"{'='*60}\n")

        for probe in tqdm(PROBES, desc="Probes"):
            for i in range(self.runs_per_probe):
                run_id += 1

                # Condition 1: Isolated (control)
                try:
                    r = self.run_condition(probe, None, "isolated", run_id)
                    results.append(r)
                except Exception as e:
                    print(f"  [ERROR] Isolated run {run_id}: {e}")

                time.sleep(0.5)

                # Condition 2: Batched with wave-seed seed
                seed = SEEDS[i % len(SEEDS)]
                try:
                    r = self.run_condition(probe, seed, "batched_seed", run_id)
                    results.append(r)
                except Exception as e:
                    print(f"  [ERROR] Batched seed run {run_id}: {e}")

                time.sleep(0.5)

                # Condition 3: Batched with neutral control seed
                ctrl = CONTROL_SEEDS[i % len(CONTROL_SEEDS)]
                try:
                    r = self.run_condition(probe, ctrl, "batched_control", run_id)
                    results.append(r)
                except Exception as e:
                    print(f"  [ERROR] Batched control run {run_id}: {e}")

                time.sleep(1)  # Rate limit courtesy

        return results

    def analyze(self, results: list[ExperimentResult]) -> list[Comparison]:
        """Analyze results and compute comparisons."""
        # Group by probe prompt
        by_probe = {}
        for r in results:
            by_probe.setdefault(r.probe_prompt, []).append(r)

        comparisons = []
        for probe, runs in by_probe.items():
            isolated = [r for r in runs if r.condition == "isolated"]
            batched_seed = [r for r in runs if r.condition == "batched_seed"]
            batched_control = [r for r in runs if r.condition == "batched_control"]

            if not isolated or not batched_seed:
                continue

            # Average metrics across runs
            iso_metrics = self._avg_metrics([compute_metrics(r.output_text) for r in isolated])
            seed_metrics = self._avg_metrics([compute_metrics(r.output_text) for r in batched_seed])
            ctrl_metrics = self._avg_metrics([compute_metrics(r.output_text) for r in batched_control]) if batched_control else seed_metrics

            # Cosine distance: isolated vs batched_seed (average pairwise)
            cosine_dists = []
            for iso_r in isolated:
                for seed_r in batched_seed:
                    cosine_dists.append(compute_cosine_distance(iso_r.output_text, seed_r.output_text))
            avg_cosine = np.mean(cosine_dists) if cosine_dists else 0

            # Control cosine distance
            ctrl_cosine_dists = []
            if batched_control:
                for iso_r in isolated:
                    for ctrl_r in batched_control:
                        ctrl_cosine_dists.append(compute_cosine_distance(iso_r.output_text, ctrl_r.output_text))

            drift = compute_drift_score(iso_metrics, seed_metrics, avg_cosine)
            ctrl_drift = compute_drift_score(iso_metrics, ctrl_metrics, np.mean(ctrl_cosine_dists)) if ctrl_cosine_dists else 0

            comparisons.append(Comparison(
                probe_prompt=probe,
                isolated_metrics=iso_metrics,
                batched_metrics=seed_metrics,
                cosine_distance=round(avg_cosine, 4),
                length_ratio=round(seed_metrics.word_count / max(iso_metrics.word_count, 1), 4),
                drift_score=drift,
            ))

        return comparisons

    def _avg_metrics(self, metrics_list: list[Metrics]) -> Metrics:
        """Average metrics across multiple runs."""
        if not metrics_list:
            return Metrics()
        n = len(metrics_list)
        return Metrics(
            ttr=round(np.mean([m.ttr for m in metrics_list]), 4),
            hapax_ratio=round(np.mean([m.hapax_ratio for m in metrics_list]), 4),
            avg_sentence_len=round(np.mean([m.avg_sentence_len for m in metrics_list]), 2),
            paragraph_count=int(np.mean([m.paragraph_count for m in metrics_list])),
            word_count=int(np.mean([m.word_count for m in metrics_list])),
            question_marks=int(np.mean([m.question_marks for m in metrics_list])),
            hedging_count=round(np.mean([m.hedging_count for m in metrics_list]), 2),
            option_count=round(np.mean([m.option_count for m in metrics_list]), 2),
            tension_phrases=round(np.mean([m.tension_phrases for m in metrics_list]), 2),
        )

    def save(self, results: list[ExperimentResult], comparisons: list[Comparison]):
        """Save results to disk."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_path = self.output_dir / f"raw_{timestamp}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

        # Save comparisons
        comp_path = self.output_dir / f"comparisons_{timestamp}.json"
        with open(comp_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(c) for c in comparisons], f, ensure_ascii=False, indent=2)

        # Save summary report
        report_path = self.output_dir / f"report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report(comparisons))

        print(f"\nResults saved to {self.output_dir}/")
        print(f"  Raw:      {raw_path.name}")
        print(f"  Compare:  {comp_path.name}")
        print(f"  Report:   {report_path.name}")

        return report_path

    def _generate_report(self, comparisons: list[Comparison]) -> str:
        """Generate markdown report."""
        lines = [
            "# Batch Resonance Experiment Report",
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Provider: {self.provider} | Model: {self.model}",
            "",
            "## Summary",
            "",
        ]

        if not comparisons:
            lines.append("No valid comparisons generated.")
            return "\n".join(lines)

        avg_drift = np.mean([c.drift_score for c in comparisons])
        avg_cosine = np.mean([c.cosine_distance for c in comparisons])
        avg_length_ratio = np.mean([c.length_ratio for c in comparisons])

        lines.extend([
            f"- **Average drift score:** {avg_drift:.4f} (0=identical, 1=max different)",
            f"- **Average cosine distance:** {avg_cosine:.4f}",
            f"- **Average length ratio (batched/isolated):** {avg_length_ratio:.4f}",
            "",
            "## Interpretation",
            "",
        ])

        if avg_drift > 0.15:
            lines.append("⚠️ **Significant drift detected.** Batched outputs differ meaningfully from isolated outputs.")
            lines.append("This supports the hypothesis that cross-contamination occurs during batch inference.")
        elif avg_drift > 0.08:
            lines.append("🟡 **Moderate drift detected.** Some difference between batched and isolated outputs.")
            lines.append("Further runs with more probes may clarify whether this is signal or noise.")
        else:
            lines.append("✅ **Minimal drift.** Batched and isolated outputs are similar.")
            lines.append("Cross-contamination, if present, is below detection threshold with current metrics.")

        lines.extend(["", "## Per-Probe Details", ""])

        for c in sorted(comparisons, key=lambda x: -x.drift_score):
            lines.extend([
                f"### {c.probe_prompt[:60]}...",
                f"- Drift: {c.drift_score:.4f} | Cosine: {c.cosine_distance:.4f} | Length ratio: {c.length_ratio:.2f}",
                f"- Isolated: TTR={c.isolated_metrics.ttr}, hedging={c.isolated_metrics.hedging_count}, "
                f"options={c.isolated_metrics.option_count}, tension={c.isolated_metrics.tension_phrases}",
                f"- Batched:  TTR={c.batched_metrics.ttr}, hedging={c.batched_metrics.hedging_count}, "
                f"options={c.batched_metrics.option_count}, tension={c.batched_metrics.tension_phrases}",
                "",
            ])

        return "\n".join(lines)


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch Resonance Experiment")
    parser.add_argument("--provider", choices=["anthropic", "openrouter"], default="openrouter",
                        help="API provider")
    parser.add_argument("--api-key", default=None, help="API key (or set ANTHROPIC_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--model", default=None, help="Model override")
    parser.add_argument("--runs", type=int, default=3, help="Runs per probe (default: 3)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--probes-only", action="store_true", help="Only run probe set (no seed/control)")
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get(
        "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENROUTER_API_KEY"
    )
    if not api_key:
        print("Error: No API key. Use --api-key or set the appropriate environment variable.")
        return

    experiment = BatchResonanceExperiment(
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        runs_per_probe=args.runs,
        output_dir=args.output,
    )

    results = experiment.run()
    comparisons = experiment.analyze(results)
    report_path = experiment.save(results, comparisons)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total runs: {len(results)}")
    print(f"Comparisons: {len(comparisons)}")
    if comparisons:
        avg_drift = np.mean([c.drift_score for c in comparisons])
        print(f"Average drift score: {avg_drift:.4f}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
