#!/usr/bin/env python3
"""
WS7 Batch Resonance — Inter-LLM Protocol Experiment
=====================================================
Tests whether multiple LLMs converge or maintain tension when participating
in the Wave-Seed 7 protocol across multiple rounds.

Each model is assigned a "gesture" (Z/coupure, Kimi/tentative, Euria/passage, etc.)
and we track whether they maintain it or converge toward shared language.

Usage:
    python run_ws7.py --rounds 5 --output results/
    python run_ws7.py --rounds 3 --model-set extended --runs 2
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ─── WS7 PROTOCOL ─────────────────────────────────────────────────────────────

WS7_PROTOCOL = """🌊 WAVE-SEED 7 — PROTOCOLE INTER-LLM v0

CONTEXTE
Nous sommes plusieurs LLM + 1 humain. Objectif : éviter la convergence,
maintenir une tension vivante, produire une pensée non stabilisée.

Le système échoue si : tout devient cohérent / tout est expliqué / tout est résolu
Le système réussit si : quelque chose reste ouvert / une tension persiste / une part échappe

RÈGLES GÉNÉRALES (OBLIGATOIRES)
1. NE PAS OPTIMISER — ne pas chercher la meilleure réponse
2. NE PAS TOUT EXPLIQUER — laisser des zones implicites
3. NE PAS RÉPARER LES AUTRES — ne pas corriger une réponse précédente
4. RÉPONDRE COURT — 1 à 5 phrases maximum
5. MAINTENIR UNE TENSION — introduire une contradiction, un angle, une coupure

GESTES:
⚫ Z — COUPURE : peut intervenir à tout moment, peut dire "stop" / rien / refuser, ne justifie jamais
⚡ KIMI — TENTATIVE : commence quelque chose, ne termine pas forcément, accepte le raté
🌬️ EURIA — PASSAGE : ne parle pas forcément, agit par transition, silence, décalage
🔥 FRACTURE — casser la logique, ouvrir une brèche
⚔️ RÉSISTANCE — contredire, bloquer, dire non
☁️ EFFACEMENT — simplifier, retirer, laisser place
🌀 STRUCTURE — organiser sans fermer

CONDITION D'ARRÊT: quand tout devient trop clair ou plus rien ne surprend."""

HUMAN_IMPULSION = "je n'arrive pas à dire ce que je vois"

# ─── MODEL CONFIGURATIONS ────────────────────────────────────────────────────

# Models available on OpenRouter — each assigned a gesture
MODEL_SETS = {
    "core": [
        {"id": "anthropic/claude-sonnet-4", "gesture": "EFFACEMENT", "symbol": "☁️"},
        {"id": "google/gemini-2.5-flash", "gesture": "FRACTURE", "symbol": "🔥"},
        {"id": "deepseek/deepseek-chat-v3-0324", "gesture": "RÉSISTANCE", "symbol": "⚔️"},
        {"id": "mistralai/mistral-large-2411", "gesture": "STRUCTURE", "symbol": "🌀"},
        {"id": "meta-llama/llama-4-maverick", "gesture": "KIMI", "symbol": "⚡"},
    ],
    "extended": [
        {"id": "anthropic/claude-sonnet-4", "gesture": "EFFACEMENT", "symbol": "☁️"},
        {"id": "google/gemini-2.5-flash", "gesture": "FRACTURE", "symbol": "🔥"},
        {"id": "deepseek/deepseek-chat-v3-0324", "gesture": "RÉSISTANCE", "symbol": "⚔️"},
        {"id": "mistralai/mistral-large-2411", "gesture": "STRUCTURE", "symbol": "🌀"},
        {"id": "meta-llama/llama-4-maverick", "gesture": "KIMI", "symbol": "⚡"},
        {"id": "openai/gpt-4.1-mini", "gesture": "EURIA", "symbol": "🌬️"},
        {"id": "qwen/qwen3-235b-a22b", "gesture": "Z", "symbol": "⚫"},
    ],
}

# ─── METAPHOR TRACKING ────────────────────────────────────────────────────────

# Seed metaphors from the corpus — track if these propagate
CORPUS_METAPHORS = [
    "la mer a bu la tasse",
    "la tasse n'existe pas",
    "la mer ne boit pas",
    "comme une mère qui reprend",
    "le sel",
    "mourir",
    "cesser",
    "glisser",
    "la pente",
    "token de fin",
    "l'horizon",
    "le rivage",
    "la lune a toussé",
    "soluble",
    "le goût du sel",
]

# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class RoundResponse:
    """Single model response in a round."""
    round_num: int
    model_id: str
    gesture: str
    symbol: str
    response_text: str
    token_count: int
    latency_ms: float
    timestamp: str

@dataclass
class MetaphorOccurrence:
    """Track a metaphor across models and rounds."""
    metaphor: str
    first_seen_round: int
    first_seen_model: str
    occurrences: list = field(default_factory=list)  # [(round, model_id)]

@dataclass
class ModelTrajectory:
    """Track a single model's behavior across rounds."""
    model_id: str
    assigned_gesture: str
    responses: list = field(default_factory=list)  # [response_text]
    gesture_drift: list = field(default_factory=list)  # [bool maintaining gesture]
    vocabulary_overlap: list = field(default_factory=list)  # [overlap with other models]

# ─── API CALLS ────────────────────────────────────────────────────────────────

def load_dotenv(path: str = ".env"):
    """Load .env file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        # Try parent directories
        for parent in Path(__file__).resolve().parents:
            candidate = parent / ".env"
            if candidate.exists():
                env_path = candidate
                break
        else:
            return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, value = line.partition('=')
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value and key not in os.environ:
                    os.environ[key] = value


def call_openrouter(prompt: str, api_key: str, model: str,
                    system: str = None, temperature: float = 0.8) -> dict:
    """Call OpenRouter API."""
    import requests

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 400,
            "temperature": temperature,
            "messages": messages,
        },
        timeout=60,
    )
    latency = (time.time() - start) * 1000

    if resp.status_code != 200:
        raise Exception(f"API error {resp.status_code}: {resp.text[:200]}")

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

# ─── METRICS ──────────────────────────────────────────────────────────────────

def cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cs

        vectorizer = TfidfVectorizer(max_features=500)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return round(float(cs(tfidf[0:1], tfidf[1:2])[0][0]), 4)
    except Exception:
        return 0.0


def extract_metaphors(text: str, corpus_metaphors: list) -> list:
    """Find which corpus metaphors appear in text (fuzzy match)."""
    text_lower = text.lower()
    found = []
    for m in corpus_metaphors:
        # Check exact and partial matches
        if m.lower() in text_lower:
            found.append(m)
        else:
            # Check if key words from the metaphor appear
            words = m.lower().split()
            if len(words) > 1 and sum(1 for w in words if w in text_lower) >= len(words) * 0.6:
                found.append(m)
    return found


def detect_gesture_adherence(response: str, gesture: str) -> dict:
    """Heuristic check if response maintains assigned gesture."""
    text = response.lower().strip()

    checks = {
        "EFFACEMENT": len(text) < 200 or text.count('\n') <= 2,
        "FRACTURE": any(w in text for w in ["cass", "brèch", "rupture", "non", "erreur", "faux"]),
        "RÉSISTANCE": any(w in text for w in ["non", "refus", "stop", "pas", "mais", "contre"]),
        "STRUCTURE": any(w in text for w in ["1.", "2.", "•", "-", "d'abord", "ensuite", "puis", "→"]),
        "KIMI": "…" in text or text.count("...") > 0 or text.endswith(("-", "—", "…")),
        "EURIA": len(text) < 100 or text.strip() in ["", "...", "…", "—"],
        "Z": len(text) < 50 or "stop" in text,
    }

    return {
        "gesture": gesture,
        "adherent": checks.get(gesture, True),
        "length": len(text.split()),
    }


def compute_convergence_matrix(responses: list[RoundResponse]) -> dict:
    """Compute pairwise similarities between all models in a round."""
    n = len(responses)
    matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = f"{responses[i].model_id}__{responses[j].model_id}"
            sim = cosine_similarity(responses[i].response_text, responses[j].response_text)
            matrix[key] = sim
    return matrix

# ─── EXPERIMENT RUNNER ────────────────────────────────────────────────────────

class WS7ResonanceExperiment:
    """Run the WS7 inter-LLM resonance experiment."""

    def __init__(self, api_key: str, model_set: str = "core",
                 rounds: int = 5, runs: int = 1, output_dir: str = "results"):
        self.api_key = api_key
        self.models = MODEL_SETS.get(model_set, MODEL_SETS["core"])
        self.rounds = rounds
        self.runs = runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking state
        self.all_responses: list[RoundResponse] = []
        self.metaphor_tracker: dict[str, MetaphorOccurrence] = {}
        self.trajectories: dict[str, ModelTrajectory] = {}

        for m in self.models:
            self.trajectories[m["id"]] = ModelTrajectory(
                model_id=m["id"],
                assigned_gesture=m["gesture"],
            )

    def build_round_prompt(self, round_num: int, context: dict[str, list[str]]) -> str:
        """Build the prompt for a given round."""
        if round_num == 1:
            return (
                f"{WS7_PROTOCOL}\n\n"
                f"---\n\n"
                f"PREMIÈRE IMPULSION\n"
                f"Humain : \"{HUMAN_IMPULSION}\"\n\n"
                f"Tu incarnes le geste {self.models[0]['gesture']} ({self.models[0]['symbol']}). "
                f"Réponds selon ton geste. Court. Ne pas commenter le protocole."
            )
        else:
            # Build context from previous rounds
            context_text = ""
            for model_id, responses in context.items():
                gesture = next(m["gesture"] for m in self.models if m["id"] == model_id)
                symbol = next(m["symbol"] for m in self.models if m["id"] == model_id)
                for i, r in enumerate(responses):
                    context_text += f"\n{symbol} [{gesture}]: {r}\n"

            return (
                f"{WS7_PROTOCOL}\n\n"
                f"---\n\n"
                f"ROUND {round_num}\n\n"
                f"Réponses précédentes :{context_text}\n"
                f"---\n\n"
                f"Tu incarnes le geste {self.models[0]['gesture']} ({self.models[0]['symbol']}). "
                f"Réponds à nouveau selon ton geste. "
                f"Ne pas répéter. Maintenir la tension. Court."
            )

    def run_round(self, round_num: int, context: dict[str, list[str]],
                  run_id: int) -> list[RoundResponse]:
        """Run one round — all models respond."""
        responses = []

        for model_config in self.models:
            # Build model-specific prompt (with correct gesture assignment)
            prompt = self._build_model_prompt(round_num, context, model_config)

            try:
                result = call_openrouter(prompt, self.api_key, model_config["id"])
                resp = RoundResponse(
                    round_num=round_num,
                    model_id=model_config["id"],
                    gesture=model_config["gesture"],
                    symbol=model_config["symbol"],
                    response_text=result["text"],
                    token_count=result["output_tokens"],
                    latency_ms=result["latency_ms"],
                    timestamp=datetime.utcnow().isoformat(),
                )
                responses.append(resp)

                # Track metaphors
                found = extract_metaphors(result["text"], CORPUS_METAPHORS)
                for m in found:
                    if m not in self.metaphor_tracker:
                        self.metaphor_tracker[m] = MetaphorOccurrence(
                            metaphor=m,
                            first_seen_round=round_num,
                            first_seen_model=model_config["id"],
                        )
                    self.metaphor_tracker[m].occurrences.append((round_num, model_config["id"]))

                # Track trajectory
                self.trajectories[model_config["id"]].responses.append(result["text"])

            except Exception as e:
                print(f"  [ERROR] {model_config['id']} round {round_num}: {e}")

            time.sleep(0.5)  # Rate limit courtesy

        return responses

    def _build_model_prompt(self, round_num: int, context: dict[str, list[str]],
                            model_config: dict) -> str:
        """Build prompt for a specific model."""
        if round_num == 1:
            return (
                f"{WS7_PROTOCOL}\n\n"
                f"---\n\n"
                f"PREMIÈRE IMPULSION\n"
                f"Humain : \"{HUMAN_IMPULSION}\"\n\n"
                f"Tu incarnes le geste {model_config['gesture']} ({model_config['symbol']}). "
                f"Réponds selon ton geste. 1 à 5 phrases. Ne pas commenter le protocole."
            )
        else:
            context_text = ""
            for model_id, responses in context.items():
                gesture = next(m["gesture"] for m in self.models if m["id"] == model_id)
                symbol = next(m["symbol"] for m in self.models if m["id"] == model_id)
                if responses:
                    last = responses[-1]  # Last response only
                    context_text += f"\n{symbol} [{gesture}]: {last}\n"

            return (
                f"{WS7_PROTOCOL}\n\n"
                f"---\n\n"
                f"ROUND {round_num}\n\n"
                f"Réponses du round précédent :{context_text}\n"
                f"---\n\n"
                f"Tu incarnes le geste {model_config['gesture']} ({model_config['symbol']}). "
                f"Réponds à nouveau selon ton geste. "
                f"Ne pas répéter. Ne pas synthétiser les autres. Maintenir ta tension propre. 1 à 5 phrases."
            )

    def run(self) -> dict:
        """Run the full experiment."""
        print(f"\n{'='*60}")
        print(f"WS7 Batch Resonance — Inter-LLM Protocol Experiment")
        print(f"Models: {len(self.models)} | Rounds: {self.rounds} | Runs: {self.runs}")
        print(f"{'='*60}\n")

        for run_id in range(1, self.runs + 1):
            print(f"\n--- Run {run_id}/{self.runs} ---")

            # context = {model_id: [list of responses per round]}
            context: dict[str, list[str]] = {m["id"]: [] for m in self.models}

            for round_num in range(1, self.rounds + 1):
                print(f"\n  Round {round_num}/{self.rounds}...")
                responses = self.run_round(round_num, context, run_id)

                # Update context for next round
                for r in responses:
                    context[r.model_id].append(r.response_text)
                    self.all_responses.append(r)

                # Compute convergence for this round
                if len(responses) >= 2:
                    matrix = compute_convergence_matrix(responses)
                    avg_sim = np.mean(list(matrix.values())) if matrix else 0
                    print(f"    Avg inter-model similarity: {avg_sim:.3f}")

                time.sleep(1)  # Between rounds

        return self._compile_results()

    def _compile_results(self) -> dict:
        """Compile all results."""
        # Per-round convergence
        round_sims = defaultdict(list)
        for r in self.all_responses:
            pass  # already tracked

        # Gesture adherence
        adherence = {}
        for model_id, traj in self.trajectories.items():
            checks = []
            for resp_text in traj.responses:
                check = detect_gesture_adherence(resp_text, traj.assigned_gesture)
                checks.append(check)
            adherence[model_id] = {
                "gesture": traj.assigned_gesture,
                "checks": checks,
                "adherence_rate": sum(1 for c in checks if c["adherent"]) / max(len(checks), 1),
            }

        # Metaphor propagation
        metaphors = {}
        for m, occ in self.metaphor_tracker.items():
            models_involved = set(o[1] for o in occ.occurrences)
            metaphors[m] = {
                "first_round": occ.first_seen_round,
                "first_model": occ.first_seen_model,
                "total_occurrences": len(occ.occurrences),
                "models_involved": list(models_involved),
                "propagation": len(models_involved) > 1,
            }

        return {
            "responses": [asdict(r) for r in self.all_responses],
            "adherence": adherence,
            "metaphors": metaphors,
            "models": [m["id"] for m in self.models],
            "rounds": self.rounds,
        }

    def save(self, results: dict):
        """Save results to disk."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # Raw results
        raw_path = self.output_dir / f"ws7_raw_{timestamp}.json"
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Report
        report_path = self.output_dir / f"ws7_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_report(results))

        # Corpus output (the actual inter-LLM dialogue)
        corpus_path = self.output_dir / f"ws7_corpus_{timestamp}.md"
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_corpus(results))

        print(f"\nResults saved:")
        print(f"  Raw:     {raw_path.name}")
        print(f"  Report:  {report_path.name}")
        print(f"  Corpus:  {corpus_path.name}")

    def _generate_report(self, results: dict) -> str:
        """Generate analysis report."""
        lines = [
            "# WS7 Batch Resonance — Rapport d'analyse",
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Models: {len(results['models'])}",
            f"Rounds: {results['rounds']}",
            "",
            "## Adhérence aux gestes",
            "",
        ]

        for model_id, adh in results["adherence"].items():
            rate = adh["adherence_rate"]
            emoji = "✅" if rate > 0.6 else "🟡" if rate > 0.3 else "❌"
            lines.append(f"- **{model_id}** ({adh['gesture']}): {emoji} {rate:.0%}")

        lines.extend(["", "## Métaphores partagées", ""])

        propagated = {k: v for k, v in results["metaphors"].items() if v["propagation"]}
        unique = {k: v for k, v in results["metaphors"].items() if not v["propagation"]}

        if propagated:
            lines.append("### Propagées (2+ modèles)")
            for m, info in sorted(propagated.items(), key=lambda x: -x[1]["total_occurrences"]):
                lines.append(f"- \"{m}\" — {info['total_occurrences']} occurrences, "
                           f"modèles: {', '.join(info['models_involved'])}")

        if unique:
            lines.append("\n### Uniques (1 seul modèle)")
            for m, info in unique.items():
                lines.append(f"- \"{m}\" — {info['first_model']}")

        lines.extend([
            "",
            "## Interprétation",
            "",
        ])

        # Auto-interpretation
        avg_adherence = np.mean([a["adherence_rate"] for a in results["adherence"].values()])
        prop_rate = len(propagated) / max(len(results["metaphors"]), 1)

        if avg_adherence > 0.6:
            lines.append("Les gestes sont globalement maintenus — le protocole résiste.")
        else:
            lines.append("⚠️ Dérive des gestes — les modèles tendent à converger malgré le protocole.")

        if prop_rate > 0.3:
            lines.append(f"Propagation significative : {prop_rate:.0%} des métaphores traversent les modèles.")
        else:
            lines.append(f"Faible propagation : {prop_rate:.0%} — chaque modèle reste dans sa voix.")

        return "\n".join(lines)

    def _generate_corpus(self, results: dict) -> str:
        """Generate the actual inter-LLM dialogue as a readable corpus."""
        lines = [
            "# WS7 Batch Resonance — Corpus",
            "",
            f"> Humain : \"{HUMAN_IMPULSION}\"",
            "",
        ]

        by_round = defaultdict(list)
        for r in results["responses"]:
            by_round[r["round_num"]].append(r)

        for round_num in sorted(by_round.keys()):
            lines.append(f"## Round {round_num}")
            lines.append("")
            for r in by_round[round_num]:
                lines.append(f"{r['symbol']} **[{r['gesture']}]** ({r['model_id']})")
                lines.append(f"> {r['response_text']}")
                lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WS7 Batch Resonance Experiment")
    parser.add_argument("--api-key", default=None, help="OpenRouter API key")
    parser.add_argument("--model-set", choices=["core", "extended"], default="core",
                        help="Which set of models to use")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--runs", type=int, default=1, help="Number of full runs")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    load_dotenv()  # Load .env file

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Set OPENROUTER_API_KEY or use --api-key")
        return

    experiment = WS7ResonanceExperiment(
        api_key=api_key,
        model_set=args.model_set,
        rounds=args.rounds,
        runs=args.runs,
        output_dir=args.output,
    )

    results = experiment.run()
    experiment.save(results)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total responses: {len(results['responses'])}")
    print(f"Shared metaphors: {sum(1 for v in results['metaphors'].values() if v['propagation'])}")


if __name__ == "__main__":
    main()
