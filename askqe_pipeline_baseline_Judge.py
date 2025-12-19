#!/usr/bin/env python3
"""
AskQE pipeline utility.

Two entrypoints are provided:

1) Generation mode (`--run-generation`): run QG/QA/backtranslation with
   Qwen2.5-14B-Instruct-AWQ via vLLM on batches of JSONL inputs.
   - Uses the *original* AskQE prompts from the repository for atomic facts,
     question generation, and question answering.
   - Optional backtranslation prompt (LLM-based) if no `backtranslation`
     field is present in the input.
   - Outputs a JSONL file with the augmented fields.

2) Scoring mode (default): reproduce the official AskQE evaluation flow by
   consuming pre-generated QA artifacts under `QA/` and computing the
   string-comparison metrics plus SBERT similarity, saving a CSV aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Ensure we reuse the original metric implementations and prompts.
REPO_ROOT = Path(__file__).resolve().parent
UTILS_PATH = REPO_ROOT / "evaluation" / "string-comparison"
sys.path.insert(0, str(UTILS_PATH))

from utils import bleu_score, chrf_score, exact_match_score, f1_score  # noqa: E402
from QG.code.prompt import nli as qg_nli_prompt  # noqa: E402
from QA.code.prompt import qa_prompt  # noqa: E402
from biomqm.askqe.prompt import atomic_fact_prompt  # noqa: E402


LANGUAGES = ["es", "fr", "hi", "tl", "zh"]
PIPELINES = ["vanilla", "semantic", "atomic"]
PERTURBATIONS = [
    "alteration",
    "expansion_impact",
    "expansion_noimpact",
    "intensifier",
    "omission",
    "spelling",
    "synonym",
    "word_order",
]


def build_vllm_components(model_id: str, gpu_mem_util: float, max_tokens: int):
    """Create a shared vLLM model + tokenizer and sampling params."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    llm = LLM(
        model=model_id,
        dtype="auto",
        max_model_len=max_tokens,
        gpu_memory_utilization=gpu_mem_util,
        seed=0,
        tensor_parallel_size=1,
    )
    sampling = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=["```", "\n\n"],
    )
    return llm, tokenizer, sampling


def apply_chat(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    sampling: SamplingParams,
) -> List[str]:
    if not prompts:
        return []
    formatted = [apply_chat(tokenizer, p) for p in prompts]
    outputs = llm.generate(formatted, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


def parse_list_output(text: str) -> List[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, str)]
    except Exception:
        pass
    return []


def run_generation(args: argparse.Namespace) -> None:
    """Run QG/QA/backtranslation with Qwen2.5-14B-Instruct-AWQ on JSONL input."""
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llm, tokenizer, sampling = build_vllm_components(
        args.model_id, args.gpu_mem_util, args.max_tokens
    )

    logging.info("Loading input from %s", input_path)
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = entry.get("en") or entry.get("src")
            target = entry.get("tgt")
            backtranslation = entry.get("backtranslation") or entry.get("bt_tgt")
            if not source:
                continue

            # Phase 1: atomic facts
            facts_prompt = atomic_fact_prompt.replace("{{sentence}}", source)
            facts_text = generate_batch(llm, tokenizer, [facts_prompt], sampling)[0]
            atomic_facts = parse_list_output(facts_text)

            # Phase 2: question generation (NLI template)
            qg_prompt = (
                qg_nli_prompt.replace("{{sentence}}", source).replace(
                    "{{atomic_facts}}", str(atomic_facts)
                )
            )
            qg_text = generate_batch(llm, tokenizer, [qg_prompt], sampling)[0]
            questions = parse_list_output(qg_text)

            # Optional backtranslation if missing
            if not backtranslation and target:
                bt_prompt = (
                    "Translate the following text to English.\n\n"
                    f"Input: {target}\nOutput:"
                )
                backtranslation = generate_batch(
                    llm, tokenizer, [bt_prompt], sampling
                )[0]

            # Phase 3/4: QA on source and backtranslation
            qa_prompts = []
            qa_prompts.append(
                qa_prompt.replace("{{sentence}}", source).replace(
                    "{{questions}}", str(questions)
                )
            )
            qa_prompts.append(
                qa_prompt.replace(
                    "{{sentence}}", backtranslation or "No context provided."
                ).replace("{{questions}}", str(questions))
            )
            qa_text = generate_batch(llm, tokenizer, qa_prompts, sampling)
            answers_src = parse_list_output(qa_text[0]) if qa_text else []
            answers_bt = parse_list_output(qa_text[1]) if len(qa_text) > 1 else []

            entry["atomic_facts"] = atomic_facts
            entry["questions"] = questions
            entry["backtranslation"] = backtranslation
            entry["answers_src"] = answers_src
            entry["answers_bt"] = answers_bt
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logging.info("Generation complete. Output -> %s", output_path)


def parse_answers(entry: dict) -> List[str]:
    """Return the normalized list of answers from a QA JSONL entry."""
    answers = entry.get("answers", [])
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            return []
    if not isinstance(answers, list):
        return []
    return [a for a in answers if isinstance(a, str) and a.strip()]


def load_qa_pairs(
    predicted_path: Path, reference_path: Path
) -> List[Tuple[List[str], List[str]]]:
    """Load QA pairs, aligning predicted and reference answers."""
    pairs: List[Tuple[List[str], List[str]]] = []
    with predicted_path.open("r", encoding="utf-8") as pred_file, reference_path.open(
        "r", encoding="utf-8"
    ) as ref_file:
        for pred_line, ref_line in zip(pred_file, ref_file):
            try:
                pred_entry = json.loads(pred_line)
                ref_entry = json.loads(ref_line)
            except json.JSONDecodeError:
                continue

            pred_answers = parse_answers(pred_entry)
            ref_answers = parse_answers(ref_entry)

            if not pred_answers or not ref_answers:
                continue
            if len(pred_answers) != len(ref_answers):
                continue

            pairs.append((pred_answers, ref_answers))
    return pairs


def average_string_metrics(pairs: Iterable[Tuple[List[str], List[str]]]) -> dict:
    """Compute mean F1/EM/chrF/BLEU across all answer pairs."""
    f1_scores: List[float] = []
    em_scores: List[float] = []
    chrf_scores: List[float] = []
    bleu_scores: List[float] = []

    for pred_answers, ref_answers in pairs:
        for pred, ref in zip(pred_answers, ref_answers):
            f1_scores.append(f1_score(pred, ref, normalize=True))
            em_scores.append(float(exact_match_score(pred, ref, normalize=True)))
            chrf_scores.append(chrf_score(pred, ref, normalize=True))
            bleu_scores.append(bleu_score(pred, ref, normalize=True))

    if not f1_scores:
        return {
            "f1": 0.0,
            "em": 0.0,
            "chrf": 0.0,
            "bleu": 0.0,
            "count": 0,
        }

    return {
        "f1": float(np.mean(f1_scores)),
        "em": float(np.mean(em_scores)),
        "chrf": float(np.mean(chrf_scores)),
        "bleu": float(np.mean(bleu_scores)),
        "count": len(f1_scores),
    }


def average_sbert_similarity(
    pairs: Iterable[Tuple[List[str], List[str]]], model: SentenceTransformer
) -> Tuple[float, int]:
    """Compute mean SBERT cosine similarity over all answer pairs."""
    src_texts: List[str] = []
    tgt_texts: List[str] = []

    for pred_answers, ref_answers in pairs:
        for pred, ref in zip(pred_answers, ref_answers):
            src_texts.append(pred)
            tgt_texts.append(ref)

    if not src_texts:
        return 0.0, 0

    src_emb = model.encode(src_texts, batch_size=64, convert_to_tensor=True)
    tgt_emb = model.encode(tgt_texts, batch_size=64, convert_to_tensor=True)
    similarities = util.cos_sim(src_emb, tgt_emb).diagonal()
    return float(similarities.mean().item()), len(similarities)


def evaluate_combination(
    predicted_path: Path, reference_path: Path, sbert_model: SentenceTransformer
) -> dict | None:
    """Compute metrics for a single language/pipeline/perturbation combo."""
    if not predicted_path.exists() or not reference_path.exists():
        logging.warning("Missing files: %s or %s", predicted_path, reference_path)
        return None

    pairs = load_qa_pairs(predicted_path, reference_path)
    if not pairs:
        logging.warning("No aligned QA pairs found for %s", predicted_path)
        return None

    string_metrics = average_string_metrics(pairs)
    sbert_mean, sbert_count = average_sbert_similarity(pairs, sbert_model)

    return {
        **string_metrics,
        "sbert": sbert_mean,
        "sbert_count": sbert_count,
    }


def build_output_row(
    language: str, perturbation: str, pipeline: str, metrics: dict
) -> List[str | float | int]:
    return [
        language,
        perturbation,
        pipeline,
        metrics["count"],
        metrics["f1"],
        metrics["em"],
        metrics["chrf"],
        metrics["bleu"],
        metrics["sbert"],
        metrics["sbert_count"],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AskQE pipeline utility. "
            "Use --run-generation for QG/QA/backtranslation with Qwen2.5-14B AWQ, "
            "or omit to run the scoring/aggregation step."
        )
    )
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Run QG/QA/backtranslation with Qwen2.5-14B-Instruct-AWQ.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="JSONL input file with `en` (or `src`) and optional `tgt` / `backtranslation`.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=REPO_ROOT / "askqe_generation_output.jsonl",
        help="Path to write augmented JSONL when --run-generation is set.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-14B-Instruct-AWQ",
        help="Model ID to load with vLLM for generation.",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.85,
        help="vLLM gpu_memory_utilization (tune for L4 22.5GB).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for generation (vLLM SamplingParams).",
    )
    parser.add_argument(
        "--model",
        default="llama-70b",
        help="Model subdirectory under QA/ containing generated answers.",
    )
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=REPO_ROOT / "QA",
        help="Root directory containing QA outputs (default: ./QA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "askqe_pipeline_baseline_results.csv",
        help="CSV file to write aggregated metrics.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s - %(message)s", level=logging.INFO, force=True
    )

    if args.run_generation:
        if not args.input_file:
            parser.error("--input-file is required when using --run-generation")
        run_generation(args)
        return

    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    qa_root = args.qa_dir / args.model
    output_rows: List[List[str | float | int]] = []

    for language in LANGUAGES:
        for pipeline in PIPELINES:
            for perturbation in PERTURBATIONS:
                predicted_path = qa_root / f"{language}-{pipeline}-{perturbation}.jsonl"
                reference_path = qa_root / f"en-{pipeline}.jsonl"

                metrics = evaluate_combination(
                    predicted_path, reference_path, sbert_model
                )
                if metrics is None:
                    continue

                row = build_output_row(language, perturbation, pipeline, metrics)
                output_rows.append(row)
                logging.info(
                    "%s | %s | %s -> F1: %.3f, EM: %.3f, chrF: %.3f, BLEU: %.3f, SBERT: %.3f (%d pairs)",
                    language,
                    pipeline,
                    perturbation,
                    metrics["f1"],
                    metrics["em"],
                    metrics["chrf"],
                    metrics["bleu"],
                    metrics["sbert"],
                    metrics["count"],
                )

    header = [
        "language",
        "perturbation",
        "pipeline",
        "num_pairs",
        "f1",
        "em",
        "chrf",
        "bleu",
        "sbert",
        "sbert_pairs",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(output_rows)

    logging.info("Saved results to %s", args.output)


if __name__ == "__main__":
    main()
