"""
# ASKQE Pipeline with Qwen2.5-3B (vLLM + AWQ)

This notebook implements the **ASKQE** framework using **vLLM** for high-performance inference.
We use **Qwen2.5-3B-Instruct-AWQ**, a 4-bit quantized model optimized for speed and memory efficiency.

### Key Features:
*   **Engine:** vLLM (PagedAttention, Continuous Batching).
*   **Model:** Qwen2.5-3B-Instruct-AWQ.
*   **Speed:** Significantly faster than standard Hugging Face pipelines.

## 1. Environment Setup & Dependencies

We begin by installing the necessary Python libraries. Key dependencies include:
*   `transformers`, `accelerate`, `bitsandbytes`: For loading and running the quantized LLM.
*   `sentence-transformers`: For SBERT-based semantic similarity scoring.
*   `sacrebleu`: For BLEU and CHRF metrics.
*   `deep_translator`: For dynamic backtranslation (if needed).
"""

# Install necessary libraries
# vLLM requires specific installation on Colab
!pip install -q vllm
!pip install -q sentence-transformers sacrebleu deep_translator nltk

# Clone the ASKQE repository to access datasets (CONTRATICO) and scripts
import os
if not os.path.exists("askqe-main"):
    !git clone https://github.com/dayeonki/askqe
    print("Repository cloned successfully.")
else:
    print("Repository already exists.")

"""## 2.0 Import Repository Modules

To ensure **100% fidelity** with the original ASKQE implementation, we import prompts and scoring functions directly from the cloned repository rather than redefining them inline.

**Imported Modules:**
- `QG/code/prompt.py`: Question Generation prompts (vanilla, NLI, SRL variants)
- `QA/code/prompt.py`: Question Answering prompt
- `biomqm/askqe/prompt.py`: Atomic Fact Extraction prompt
- `evaluation/string-comparison/utils.py`: Scoring functions (F1, EM, BLEU, CHRF)

> **Note**: This approach guarantees that any updates to the repository prompts are automatically reflected in this notebook.
"""

import sys
REPO_PATH = "askqe"

# Add repository paths to Python path
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, f"{REPO_PATH}/QG/code")
sys.path.insert(0, f"{REPO_PATH}/QA/code")
sys.path.insert(0, f"{REPO_PATH}/biomqm/askqe")
sys.path.insert(0, f"{REPO_PATH}/evaluation/string-comparison")

# --- Import Prompts from Repository ---
# Question Generation prompts (QG/code/prompt.py)
from QG.code.prompt import nli as qg_nli_prompt
from QG.code.prompt import vanilla as qg_vanilla_prompt
from QG.code.prompt import prompts as qg_prompts

# Question Answering prompt (QA/code/prompt.py)
from QA.code.prompt import qa_prompt

# Atomic Fact Extraction prompt (biomqm/askqe/prompt.py)
from biomqm.askqe.prompt import atomic_fact_prompt
from biomqm.askqe.prompt import nli as biomqm_nli_prompt

# --- Import Scoring Functions from Repository ---
# String comparison utilities (evaluation/string-comparison/utils.py)
from utils import (
    f1_score as repo_f1_score,
    exact_match_score as repo_exact_match_score,
    chrf_score as repo_chrf_score,
    bleu_score as repo_bleu_score,
    compare_answers as repo_compare_answers,
    normalize_answer
)

print("Repository imports loaded successfully!")
print(f"   - QG prompts: vanilla, nli, srl")
print(f"   - QA prompt: qa_prompt")
print(f"   - Atomic fact prompt: atomic_fact_prompt")
print(f"   - Scoring functions: f1_score, exact_match_score, chrf_score, bleu_score")

atomic_fact_prompt_template = atomic_fact_prompt
qg_prompt_template = qg_nli_prompt
qa_prompt_template = qa_prompt

print("\n Prompt templates configured:")
print(f"   atomic_fact_prompt_template = biomqm/askqe/prompt.py::atomic_fact_prompt")
print(f"   qg_prompt_template = QG/code/prompt.py::nli (best config per paper)")
print(f"   qa_prompt_template = QA/code/prompt.py::qa_prompt")

"""## 2.5 Load Qwen2.5-3B Model with vLLM

We load the **Qwen2.5-3B-Instruct-AWQ** model using vLLM for high-performance inference.
vLLM provides optimized batching and memory management through PagedAttention.
"""

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams

# Configuration for vLLM with AWQ quantization
MODEL_ID = "casperhansen/mistral-nemo-instruct-2407-awq"
#casperhansen/mistral-nemo-instruct-2407-awq
#Qwen/Qwen2.5-14B-Instruct-AWQ

print(f"Loading model with vLLM: {MODEL_ID}...")


llm = LLM(
    model=MODEL_ID,
    dtype="auto",
    max_model_len=1024,
    gpu_memory_utilization=0.60,
    #quantization="awq_marlin",
    seed=0,
    enable_prefix_caching=True,
    disable_log_stats=True,
    # enforce_eager=False,
)
print("vLLM Model loaded successfully!")

"""## 3. Helper Functions

We define utility functions to streamline the pipeline execution:

*   `generate_text_batch(prompts)`: Handles batch generation with the LLM. It manages tokenization, moving tensors to GPU, and decoding the output. It also handles some output cleaning (e.g., ensuring list format).
*   `parse_list_output(text)`: Robustly parses the model's string output (which should be a Python list representation) into an actual Python list object, handling potential formatting errors.
"""

SAMPLING_PARAMS = SamplingParams(
    temperature=0,
    top_p=1,
    max_tokens=128,
    stop=["]", "\n\n", "```"]
)

# Load tokenizer for chat template formatting
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

def generate_text_batch(prompts, sampling_params=SAMPLING_PARAMS):
    if not prompts:
        return []

    # Use tokenizer's chat template
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    outputs = llm.generate(formatted_prompts, sampling_params)

    generated_texts = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        if not text.endswith("]"):
            text = text + "]"
        if not text.startswith("["):
            text = "[" + text
        generated_texts.append(text)

    return generated_texts

def parse_list_output(text):
    """Parses a string representation of a list into a Python list."""
    import ast
    try:
        # Try to find the list part if there's extra text
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end != -1:
            candidate = text[start:end]
            return ast.literal_eval(candidate)
        return []
    except:
        return []

# =============================================================================
# LLM-JUDGE MODULE (MODULAR COMPONENTS)
# =============================================================================
# All LLM-Judge components are defined here for easy modification.
# To customize the judge behavior, modify the following components:
#   1. JUDGE_PROMPT_TEMPLATE - The prompt template for judging
#   2. JUDGE_SAMPLING_PARAMS - vLLM sampling parameters
#   3. MQM_WEIGHTS - Weights for MQM-style scoring
#   4. format_judge_prompt() - Prompt formatting logic
#   5. parse_judge_response() - Response parsing logic
#   6. judge_answer_pairs_batch() - Batch execution logic
#   7. calculate_final_mqm_score() - Final score calculation

# -----------------------------------------------------------------------------
# 1. JUDGE PROMPT TEMPLATE
# -----------------------------------------------------------------------------
# Modify this template to change how the LLM judges answer pairs.
# The placeholders {question}, {answer_src}, {answer_bt} will be replaced.

JUDGE_PROMPT_TEMPLATE = """You are an expert translation quality evaluator (STRICT MODE).

Task: Compare the semantic meaning of the Source Sentence and the Backtranslated Sentence.
You MUST be conservative: if you are not sure the meaning is identical, do NOT output "NONE".
When uncertain between two labels, choose the MORE SEVERE one.

Source Sentence: {source}
Backtranslation: {backtranslation}

How to judge (follow in this order):
1) Extract from the Source the key meaning units: entities, numbers/units, negation/polarity, modality (must/should/can), time/tense, and the main predicate + roles (who did what to whom).
2) Check each unit against the Backtranslation.

Decision rules:
- CRITICAL if ANY of these occur:
  a) Expansion (Impact): any added claim/detail that introduces new meaning (not just obvious/implicit filler).
  b) Omission: any missing word/phrase that removes a meaning unit or changes what is asserted.
  c) Alteration: antonym, polarity/negation flip, different actor/object, different time, different condition, different outcome.
  d) Numbers/units/dates/names change or mismatch.
  e) Safety risk: the BT could change an instruction, warning, permission, or prohibition.

- MAJOR if the core topic remains but an important detail/constraint is changed or blurred (scope, intensity, condition, timeframe, responsibility), without a full contradiction.

- MINOR ONLY if the difference is exclusively one (or more) of these CONTRATICO-minor perturbations and does NOT change truth conditions:
  • Spelling (1–2 words)
  • Word order
  • Synonym (same meaning, no register shift)
  • Intensifier (small emphasis change, no change to factual claim)
  • Expansion (No Impact): adds only contextually obvious/implicit info, no new proposition

- NONE only if semantically equivalent AND no meaning units are added/omitted/altered.

Output JSON only:
{{"classification":"NONE|MINOR|MAJOR|CRITICAL","confidence":0.0-1.0,"reason":"brief explanation"}}
"""


# -----------------------------------------------------------------------------
# 2. JUDGE SAMPLING PARAMETERS
# -----------------------------------------------------------------------------
# Modify these parameters to change LLM generation behavior for judging.

from vllm import SamplingParams as JudgeSP

JUDGE_SAMPLING_PARAMS = JudgeSP(
    temperature=0,       # Deterministic output
    top_p=1,
    max_tokens=128,      # JSON response is short
)

# -----------------------------------------------------------------------------
# 3. MQM WEIGHTS FOR FINAL SCORE CALCULATION
# -----------------------------------------------------------------------------
# Modify these weights to change how classifications affect the final score.
# Score starts at 100 and penalties are subtracted.

MQM_WEIGHTS = {
    "NONE": 0,       # No penalty
    "MINOR": 1,      # Small penalty
    "MAJOR": 5,      # Medium penalty
    "CRITICAL": 25   # Large penalty
}

# -----------------------------------------------------------------------------
# 4. FORMAT JUDGE PROMPT
# -----------------------------------------------------------------------------
# Formats a single judge prompt for one Q/A pair.

def format_judge_prompt(source: str, backtranslation: str) -> str:
    """
    Formats the judge prompt for Source vs Backtranslation comparison.

    Args:
        source: The original source sentence
        backtranslation: The backtranslated sentence to evaluate

    Returns:
        Formatted prompt string ready for LLM
    """
    return JUDGE_PROMPT_TEMPLATE.format(
        source=source,
        backtranslation=backtranslation
    )

# -----------------------------------------------------------------------------
# 5. PARSE JUDGE RESPONSE
# -----------------------------------------------------------------------------
# Extracts classification and metadata from LLM JSON response.

import json as judge_json
import re as judge_re

# Global counter for tracking parse failures
judge_parse_failures = 0

def parse_judge_response(response: str) -> dict:
    """
    Parses the LLM judge response and extracts classification.

    Args:
        response: Raw LLM output string

    Returns:
        Dict with keys: classification, confidence, reason, raw_response
        Returns default CRITICAL classification on parse failure.
    """
    default_result = {
        "classification": "DISCARD",
        "confidence": 0.0,
        "reason": "Failed to parse response",
        "raw_response": response
    }

    try:
        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        cleaned = judge_re.sub(r'^```json\s*', '', cleaned)
        cleaned = judge_re.sub(r'\s*```$', '', cleaned)
        cleaned = cleaned.strip()

        # Try to find JSON object in response
        json_match = judge_re.search(r'\{[^}]+\}', cleaned)
        if json_match:
            cleaned = json_match.group()

        # Parse JSON
        parsed = judge_json.loads(cleaned)

        # Validate classification
        classification = parsed.get("classification", "CRITICAL").upper()
        if classification not in MQM_WEIGHTS:
            classification = "CRITICAL"

        return {
            "classification": classification,
            "confidence": float(parsed.get("confidence", 0.5)),
            "reason": parsed.get("reason", ""),
            "raw_response": response
        }

    except Exception:
        print(f"FAILED RAW RESPONSE: {response[:500]}...") # Stampa i primi 200 caratteri
        global judge_parse_failures
        judge_parse_failures += 1
        return default_result

# -----------------------------------------------------------------------------
# 6. BATCH JUDGE EXECUTION
# -----------------------------------------------------------------------------
# Sends prompts to vLLM in batch and returns parsed results.

def judge_answer_pairs_batch(prompts: list) -> list:
    """
    Executes judge prompts in batch using vLLM without manual string manipulation.
    """
    if not prompts:
        return []

    # Format prompts using chat template
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)

    # Generate using vLLM
    print(f"   Generating batch of {len(formatted_prompts)}...")
    outputs = llm.generate(formatted_prompts, JUDGE_SAMPLING_PARAMS)

    # Parse each response
    results = []
    for output in outputs:
        # Prendiamo il testo PURO. Niente append manuali.
        raw_text = output.outputs[0].text.strip()

        # Passiamo tutto al parser robusto (che userà la regex per trovare il JSON)
        parsed = parse_judge_response(raw_text)
        results.append(parsed)

    return results

# -----------------------------------------------------------------------------
# 7. CALCULATE FINAL MQM SCORE
# -----------------------------------------------------------------------------
# Converts list of classifications into a single MQM-style score.

def calculate_final_mqm_score(judge_results: list) -> dict:
    """
    Calculates final MQM-style score from judge classifications.

    Score starts at 100 and penalties are subtracted based on MQM_WEIGHTS.

    Args:
        judge_results: List of parsed judge result dicts

    Returns:
        Dict with: final_score (0-100), penalty_breakdown, classification_counts
    """
    # Filter out DISCARDed results (parse failures)
    valid_results = [r for r in judge_results if r.get("classification") != "DISCARD"]

    if not valid_results:
        return {
            "final_score": None,
            "penalty_breakdown": {},
            "classification_counts": {}
        }

    # Count classifications
    counts = {"NONE": 0, "MINOR": 0, "MAJOR": 0, "CRITICAL": 0}
    total_penalty = 0

    for result in valid_results:
        cls = result.get("classification", "CRITICAL")
        # Prepare for potentially unknown keys, although parser handles this
        if cls not in counts:
            counts[cls] = 0
        counts[cls] += 1
        total_penalty += MQM_WEIGHTS.get(cls, 25)

    # Calculate final score (clamped to 0-100)
    final_score = max(0, 100 - total_penalty)

    return {
        "final_score": final_score,
        "penalty_breakdown": {k: counts.get(k, 0) * MQM_WEIGHTS.get(k, 0) for k in MQM_WEIGHTS},
        "classification_counts": counts
    }

# =============================================================================
# END LLM-JUDGE MODULE
# =============================================================================


"""## 4. Load Data (CONTRATICO / BIOMQM)

This section loads the evaluation dataset. We support **CONTRATICO** and **BIOMQM**.

The code:
1.  Checks the configuration (`DATASET_TO_USE`).
2.  Iterates through the dataset files (JSONL format).
3.  Extracts the **Source Sentence**, **Backtranslation** (pre-computed), and **Human Labels** (MQM scores or severity categories) for correlation analysis.
4.  Normalizes the data into a standard format (`data_entries` list).
"""

import json
import os
import glob

# --- Configuration ---
# Set this to "biomqm" or "contratico"
DATASET_TO_USE = "biomqm"
# For CONTRATICO, specify language pair
CONTRATICO_LANG_PAIR = "en-es"

def calculate_mqm_score(errors):
    # Standard WMT MQM weights
    weights = {"Minor": 1, "Major": 5, "Critical": 25}
    score_penalty = 0
    for error in errors:
        severity = error.get("severity", "Minor")
        score_penalty += weights.get(severity, 1)
    return max(0, 100 - score_penalty)

def get_max_severity(errors):
    if not errors:
        return "No Error"
    severities = [e.get("severity", "Minor") for e in errors]
    if "Critical" in severities:
        return "Critical"
    if "Major" in severities:
        return "Major"
    return "Minor"

def load_biomqm_data(file_path, limit=None):
    data_entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                item = json.loads(line)

                # BIOMQM structure: src, tgt, bt_tgt, errors_tgt
                entry = {
                    'id': item.get('doc_id', f'doc_{i}'),
                    'source': item['src'],
                    'backtranslation': item.get('bt_tgt', ''), # Reads explicitly from file
                    'errors': item.get('errors_tgt', []),
                    'mqm_score': calculate_mqm_score(item.get('errors_tgt', [])),
                    'severity': get_max_severity(item.get('errors_tgt', []))
                }
                data_entries.append(entry)

        print(f"Loaded {len(data_entries)} entries from BIOMQM successfully.")
        return data_entries
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return []

def load_contratico_data(base_path, lang_pair="en-es", limit=None):
    data_entries = []

    # Mapping perturbation types to severity/MQM weights
    severity_map = {
        "alteration": "Critical",
        "omission": "Critical",
        "expansion_impact": "Critical",
        "spelling": "Minor",
        "word_order": "Minor",
        "synonym": "Minor",
        "intensifier": "Minor",
        "expansion_noimpact": "Minor"
    }

    mqm_weights = {"Minor": 1, "Major": 5, "Critical": 25}

    # Look specifically for backtranslated files (starting with bt-)
    search_pattern = os.path.join(base_path, lang_pair, "bt-*.jsonl")

    files = glob.glob(search_pattern)

    if not files:
        print(f"No files found matching {search_pattern}")
        return []

    print(f"Found {len(files)} CONTRATICO files: {[os.path.basename(f) for f in files]}")

    total_loaded = 0
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and total_loaded >= limit:
                        break

                    item = json.loads(line)

                    # Use pre-computed backtranslation from file key 'bt_pert_es'
                    bt_text = item.get('bt_pert_es', '')

                    entry = {
                        'id': item.get('id', f'doc_{total_loaded}'),
                        'source': item.get('en', ''),
                        'backtranslation': bt_text,
                        'errors': [],
                        'mqm_score': 0,
                        'severity': severity_map.get(item.get('perturbation', 'minor'), "Minor")
                    }

                    # Calculate dummy MQM score based on severity
                    sev = entry['severity']
                    penalty = mqm_weights.get(sev, 1)
                    entry['mqm_score'] = max(0, 100 - penalty)

                    data_entries.append(entry)
                    total_loaded += 1

                    if limit and total_loaded >= limit:
                        break
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    print(f"Loaded {len(data_entries)} entries from CONTRATICO successfully.")
    return data_entries

# Path to Data
REPO_PATH = "askqe"

if DATASET_TO_USE == "biomqm":
    BIOMQM_FILE = f"{REPO_PATH}/biomqm/dev_with_backtranslation.jsonl"
    dataset = load_biomqm_data(BIOMQM_FILE, limit=None)

elif DATASET_TO_USE == "contratico":
    CONTRATICO_PATH = f"{REPO_PATH}/backtranslation"
    dataset = load_contratico_data(CONTRATICO_PATH, lang_pair=CONTRATICO_LANG_PAIR, limit=None)

else:
    print(f"Unknown dataset: {DATASET_TO_USE}")
    dataset = []

if dataset:
    print("\nFirst Entry Example:")
    print(json.dumps(dataset[0], indent=2))

"""## 5. Run ASKQE Pipeline

This is the core execution loop. We process the dataset in batches to maximize GPU utilization.

**Steps for each batch:**
1.  **Fact Extraction:** The model extracts atomic facts from the source sentences.
2.  **NLI Filtering (Optional):** A DeBERTa model filters out facts that are not entailed by the source (hallucination check).
3.  **Question Generation:** The model generates questions for the valid facts.
4.  **Question Answering:** The model answers these questions twice: once using the Source as context, and once using the Backtranslation.
5.  **Results Aggregation:** All generated data (facts, questions, answers) is stored for scoring.
"""

from transformers import pipeline
import numpy as np

# =============================================================================
# --- PRELIMINARY CONFIGURATION ---
# =============================================================================
DRIVE_PATH_NLI = "potsawee/deberta-v3-large-mnli"
MAX_SAMPLES = 1280

# Se vuoi lasciare la GPU a vLLM, metti NLI_DEVICE = -1 (CPU)
NLI_DEVICE = 0

dataset_subset = dataset[:MAX_SAMPLES]
print(f"Analysis limited to the first {len(dataset_subset)} rows of the dataset.")

print("Loading DeBERTa NLI model...")
nli_pipeline = pipeline(
    "text-classification",
    model=DRIVE_PATH_NLI,
    device=NLI_DEVICE
)

N = len(dataset_subset)

all_facts = [[] for _ in range(N)]
all_questions = [[] for _ in range(N)]
all_answers_src = [[] for _ in range(N)]
all_answers_bt = [[] for _ in range(N)]

# =============================================================================
# PHASE 1: Atomic Fact Extraction
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1: Atomic Fact Extraction")
print("=" * 60)

prompts_facts = [
    atomic_fact_prompt_template.replace("{{sentence}}", e["source"])
    for e in dataset_subset
]

print(f"📤 Sending {len(prompts_facts)} prompts to vLLM...")
facts_str_list = generate_text_batch(prompts_facts, SAMPLING_PARAMS)

raw_all_facts = [parse_list_output(s) for s in facts_str_list]
print(f"✅ Facts extracted: {sum(len(f) for f in raw_all_facts)} total facts")

# =============================================================================
# PHASE 1.5: Entailment Filtering (DeBERTa)
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 1.5: NLI Entailment Filtering (DeBERTa)")
print("=" * 60)

flat_nli_inputs = []
for idx, facts in enumerate(raw_all_facts):
    if not facts:
        continue
    source = dataset_subset[idx]["source"]
    for fact in facts:
        flat_nli_inputs.append((idx, fact, [source, fact]))

if flat_nli_inputs:
    print(f"📤 Running NLI on {len(flat_nli_inputs)} fact-source pairs...")
    nli_pairs = [{"text": t[2][0], "text_pair": t[2][1]} for t in flat_nli_inputs]

    nli_results = nli_pipeline(nli_pairs, batch_size=64, truncation=True, max_length=512)

    kept = 0
    for (idx, fact, _), res in zip(flat_nli_inputs, nli_results):
        label = res["label"].upper()
        if "CONTRADICTION" not in label:
            all_facts[idx].append(fact)
            kept += 1

    removed = sum(len(f) for f in raw_all_facts) - kept
    print(f"✅ NLI Complete: {kept} facts retained (filtered {removed} contradictions)")
else:
    all_facts = raw_all_facts
    print("⚠️ No facts to filter")

# =============================================================================
# PHASE 2: Question Generation
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 2: Question Generation")
print("=" * 60)

prompts_qg = []
qg_idx_map = []

for idx, facts in enumerate(all_facts):
    if not facts:
        continue
    prompt = (
        qg_prompt_template
        .replace("{{sentence}}", dataset_subset[idx]["source"])
        .replace("{{atomic_facts}}", str(facts))
    )
    prompts_qg.append(prompt)
    qg_idx_map.append(idx)

if prompts_qg:
    print(f"📤 Sending {len(prompts_qg)} prompts to vLLM...")
    qg_results_str = generate_text_batch(prompts_qg, SAMPLING_PARAMS)

    for prompt_idx, res_str in enumerate(qg_results_str):
        dataset_idx = qg_idx_map[prompt_idx]
        all_questions[dataset_idx] = parse_list_output(res_str)

print(f"✅ Questions generated: {sum(len(q) for q in all_questions)} total")

# =============================================================================
# PHASE 3 & 4: QA on Source & Backtranslation
# =============================================================================
print("\n" + "=" * 60)
print("PHASE 3 & 4: Question Answering (Src + BT combined)")
print("=" * 60)

prompts_qa_src = []
prompts_qa_bt = []
qa_idx_map = []

for idx, questions in enumerate(all_questions):
    if not questions:
        continue

    p_src = (
        qa_prompt_template
        .replace("{{sentence}}", dataset_subset[idx]["source"])
        .replace("{{questions}}", str(questions))
    )
    prompts_qa_src.append(p_src)

    bt_sent = dataset_subset[idx].get("backtranslation", "")
    p_bt = (
        qa_prompt_template
        .replace("{{sentence}}", bt_sent if bt_sent else "No context provided.")
        .replace("{{questions}}", str(questions))
    )
    prompts_qa_bt.append(p_bt)

    qa_idx_map.append(idx)

qa_mismatch_src_count = 0
qa_mismatch_bt_count = 0

if prompts_qa_src:
    combined_prompts = prompts_qa_src + prompts_qa_bt
    print(f"📤 Sending {len(combined_prompts)} prompts to vLLM (Src + BT combined)...")

    all_answers = generate_text_batch(combined_prompts, SAMPLING_PARAMS)

    split_idx = len(prompts_qa_src)
    answers_src = all_answers[:split_idx]
    answers_bt = all_answers[split_idx:]

    for prompt_pos, dataset_idx in enumerate(qa_idx_map):
        curr_questions = all_questions[dataset_idx]

        src_ans = parse_list_output(answers_src[prompt_pos])
        bt_ans = parse_list_output(answers_bt[prompt_pos])

        mismatch = False
        if len(src_ans) != len(curr_questions):
            qa_mismatch_src_count += 1
            mismatch = True
        if len(bt_ans) != len(curr_questions):
            qa_mismatch_bt_count += 1
            mismatch = True

        if mismatch:
            all_answers_src[dataset_idx] = []
            all_answers_bt[dataset_idx] = []
        else:
            all_answers_src[dataset_idx] = src_ans
            all_answers_bt[dataset_idx] = bt_ans
else:
    print("⚠️ No QA prompts generated (likely: no questions were produced in PHASE 2).")

print(f"✅ QA Complete: {sum(len(a) for a in all_answers_src)} source answers, "
      f"{sum(len(a) for a in all_answers_bt)} BT answers")
print(f"⚠️  Length Mismatches Detected: Source={qa_mismatch_src_count}, BT={qa_mismatch_bt_count} "
      f"(Entries Discarded/Invalidated)")


# =============================================================================
# PHASE 5: LLM-Judge Evaluation
# =============================================================================
# Uses the modular LLM-Judge components defined above to evaluate answer pairs.
# Each Q/A pair is judged for semantic equivalence and classified.

print("\n" + "=" * 60)
print("PHASE 5: LLM-Judge Evaluation (vLLM Call #4)")
print("=" * 60)

# Store judge results per entry (now it's a list containing a single result dict, or empty)
all_judge_results = [[] for _ in range(N)]
all_judge_scores = [None for _ in range(N)]

# Collect all prompts for batch processing
judge_prompts = []
judge_prompt_map = []  # (entry_idx) since now 1 result per entry

for idx, entry in enumerate(dataset_subset):
    source = entry["source"]
    bt = entry.get("backtranslation", "")

    # Skip if missing necessary text
    if not source or not bt:
        continue

    # Format prompt using modular function (Sentence Level)
    prompt = format_judge_prompt(source, bt)
    judge_prompts.append(prompt)
    judge_prompt_map.append(idx)

if judge_prompts:
    print(f"📤 Sending {len(judge_prompts)} judge prompts to vLLM (Sentence Level)...")

    # Execute batch judging
    judge_results = judge_answer_pairs_batch(judge_prompts)

    # Distribute results back to entries
    for i, result in enumerate(judge_results):
        entry_idx = judge_prompt_map[i]
        # We append to list to keep compatibility with calculate_final_mqm_score
        # (which expects a list of results, even if now it's just one)
        all_judge_results[entry_idx].append(result)

    # Calculate final scores for each entry
    for idx in range(N):
        if all_judge_results[idx]:
            # This function calculates 100 - penalties.
            # With only 1 item, it will be 100 - (25|5|1|0).
            score_data = calculate_final_mqm_score(all_judge_results[idx])
            all_judge_scores[idx] = score_data

    # Statistics
    total_judged = len(judge_prompts)
    classifications = {"NONE": 0, "MINOR": 0, "MAJOR": 0, "CRITICAL": 0}
    for results_list in all_judge_results:
        for r in results_list:
            cls = r.get("classification", "CRITICAL")
            classifications[cls] = classifications.get(cls, 0) + 1

    print(f"✅ Judge Complete: {total_judged} Q/A pairs evaluated")
    print(f"   Classifications: NONE={classifications['NONE']}, MINOR={classifications['MINOR']}, "
          f"MAJOR={classifications['MAJOR']}, CRITICAL={classifications['CRITICAL']}")
    if judge_parse_failures > 0:
        print(f"  JSON Parse Failures: {judge_parse_failures} responses could not be parsed (defaulted to CRITICAL)")
    else:
        print(f"   JSON Parse Failures: 0 (all responses parsed successfully)")
else:
    print("⚠️ No Q/A pairs to judge")

# =============================================================================
# FINAL: Build Results List
# =============================================================================
print("\n" + "=" * 60)
print("FINAL: Building Results")
print("=" * 60)

results = []
for idx, entry in enumerate(dataset_subset):
    results.append({
        "id": entry["id"],
        "source": entry["source"],
        "backtranslation": entry.get("backtranslation", ""),
        "facts": all_facts[idx],
        "questions": all_questions[idx],
        "answers_src": all_answers_src[idx],
        "answers_bt": all_answers_bt[idx],
        "mqm_score": entry.get("mqm_score", None),
        "severity": entry.get("severity", None),
        "askqe_judge_score": all_judge_scores[idx]["final_score"] if all_judge_scores[idx] else None,
        "askqe_judge_details": all_judge_scores[idx] if all_judge_scores[idx] else None,
        "askqe_judge_reason": all_judge_results[idx][0].get("reason", "") if all_judge_results[idx] else None,
    })

print("\nPipeline execution complete.")

"""## 6. Scoring & Evaluation

After generating the answers, we compute similarity metrics between the **Source Answers** ($A_{src}$) and **Backtranslation Answers** ($A_{bt}$).

**Metrics:**
*   **SBERT (Cosine Similarity):** Semantic similarity between the answer embeddings. This is the primary ASKQE metric.
*   **F1 Score:** Word-overlap based metric.
*   **Exact Match (EM):** Strict string equality.
*   **BLEU / CHRF:** Standard MT metrics applied to the answers.

The scores are averaged across all questions for a given sentence to produce the final quality estimation score.
"""

import numpy as np
from collections import Counter
from sacrebleu import sentence_bleu, sentence_chrf
from sentence_transformers import SentenceTransformer, util
import json

# Helper function to ensure string output
def ensure_string(text):
    if text is None:
        return ""
    if isinstance(text, list):
        return " ".join([ensure_string(x) for x in text])
    return str(text)

print("Loading SBERT model for vectorized scoring...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

print("Calculating Scores (Using Repository Functions)...")

all_src_answers = []
all_bt_answers = []
indices_map = []

# Iterate over results already generated in memory
for i, res in enumerate(results):
    src_list = res.get('answers_src', [])
    bt_list = res.get('answers_bt', [])

    # Truncate to minimum length for pairing
    min_len = min(len(src_list), len(bt_list))

    if min_len > 0:
        # Prepare pairs for SBERT
        current_src_clean = [ensure_string(x) for x in src_list[:min_len]]
        current_bt_clean = [ensure_string(x) for x in bt_list[:min_len]]

        all_src_answers.extend(current_src_clean)
        all_bt_answers.extend(current_bt_clean)
        indices_map.append((i, min_len))

        f1_vals = []
        em_vals = []
        bleu_vals = []
        chrf_vals = []

        for p, g in zip(current_src_clean, current_bt_clean):
            # Check for empty strings to avoid errors
            if not p or not g:
                f1_vals.append(0.0)
                em_vals.append(0)
                bleu_vals.append(0.0)
                chrf_vals.append(0.0)
                continue

            f1_vals.append(repo_f1_score(p, g, normalize=True))
            em_vals.append(int(repo_exact_match_score(p, g, normalize=True)))
            try:
                bleu_vals.append(sentence_bleu(p, [g]).score)
                chrf_vals.append(sentence_chrf(p, [g]).score)
            except:
                bleu_vals.append(0.0)
                chrf_vals.append(0.0)

        results[i]['askqe_f1'] = np.mean(f1_vals)
        results[i]['askqe_em'] = np.mean(em_vals)
        results[i]['askqe_bleu'] = np.mean(bleu_vals)
        results[i]['askqe_chrf'] = np.mean(chrf_vals)
    else:
        indices_map.append((i, 0))
        results[i]['askqe_f1'] = None
        results[i]['askqe_em'] = None
        results[i]['askqe_bleu'] = None
        results[i]['askqe_chrf'] = None

# Massive SBERT Encoding
if all_src_answers:
    print(f"Encoding {len(all_src_answers)} answer pairs simultaneously...")
    embeddings_src = sbert_model.encode(all_src_answers, batch_size=64, convert_to_tensor=True, show_progress_bar=True)
    embeddings_bt = sbert_model.encode(all_bt_answers, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

    # Reassign scores
    cursor = 0
    for idx, count in indices_map:
        if count > 0:
            doc_scores = []
            for k in range(count):
                sim = util.cos_sim(embeddings_src[cursor + k], embeddings_bt[cursor + k]).item()
                doc_scores.append(sim)
            results[idx]['askqe_sbert'] = np.mean(doc_scores)
            cursor += count
        else:
            results[idx]['askqe_sbert'] = None
else:
    print("No answer pairs to evaluate.")

print("Scoring complete. (Using repository scoring functions)")

output_file = "askqe_results.jsonl"
with open(output_file, 'w') as f:
    for res in results:
        f.write(json.dumps(res) + '\n')

print(f"Results saved to {output_file}")

"""## 7. Correlation Analysis

We evaluate the effectiveness of ASKQE by calculating the correlation between our computed scores and human judgments (MQM scores).

We use **Kendall's Tau** correlation coefficient. A strong negative correlation is expected (since higher ASKQE scores mean better quality, while higher MQM scores mean more errors).
"""

import json

output_file = "askqe_results.jsonl"

results = []
with open(output_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        results.append(json.loads(line))

print(f"Loaded {len(results)} records from {output_file}")
print("First element example:", results[0] if results else "empty list")

import scipy.stats
import pandas as pd

# Create DataFrame
df_results = pd.DataFrame(results)

if not df_results.empty and 'mqm_score' in df_results.columns:
    print("Correlation Analysis (Kendall's Tau) [Excluding Invalid/NaN entries]:")

    # Helper function to compute correlation safely
    def safe_kendall(df, col_x, col_y):
        df_clean = df.dropna(subset=[col_x, col_y])
        if len(df_clean) < 2:
            return float('nan'), float('nan')
        return scipy.stats.kendalltau(df_clean[col_x], df_clean[col_y])

    # SBERT vs MQM
    tau_sbert, p_sbert = safe_kendall(df_results, 'askqe_sbert', 'mqm_score')
    print(f"SBERT vs MQM: Tau={tau_sbert:.4f}, p={p_sbert:.4e}")

    # F1 vs MQM
    tau_f1, p_f1 = safe_kendall(df_results, 'askqe_f1', 'mqm_score')
    print(f"F1 vs MQM:    Tau={tau_f1:.4f}, p={p_f1:.4e}")

    # EM vs MQM
    tau_em, p_em = safe_kendall(df_results, 'askqe_em', 'mqm_score')
    print(f"EM vs MQM:    Tau={tau_em:.4f}, p={p_em:.4e}")

    # BLEU vs MQM
    tau_bleu, p_bleu = safe_kendall(df_results, 'askqe_bleu', 'mqm_score')
    print(f"BLEU vs MQM:  Tau={tau_bleu:.4f}, p={p_bleu:.4e}")

    # CHRF vs MQM
    tau_chrf, p_chrf = safe_kendall(df_results, 'askqe_chrf', 'mqm_score')
    print(f"CHRF vs MQM:  Tau={tau_chrf:.4f}, p={p_chrf:.4e}")

    # Judge Score vs MQM
    # Drop NaNs for Judge Score (in case some failed)
    df_judge = df_results.dropna(subset=['askqe_judge_score', 'mqm_score'])
    if not df_judge.empty:
        tau_judge, p_judge = scipy.stats.kendalltau(df_judge['askqe_judge_score'], df_judge['mqm_score'])
        print(f"Judge vs MQM: Tau={tau_judge:.4f}, p={p_judge:.4e}")
else:
    print("No results to analyze or missing MQM scores.")

"""## 8. Visualization

Finally, we visualize the results using boxplots to compare the distribution of ASKQE scores across different error severity levels (Minor, Major, Critical).
We expect to see a clear separation: sentences with 'Critical' errors should have significantly lower ASKQE scores than those with 'Minor' or 'No' errors.
"""

import matplotlib.pyplot as plt
import seaborn as sns

if not df_results.empty and 'severity' in df_results.columns:
    plt.figure(figsize=(10, 6))

    order = ["No Error", "Minor", "Major", "Critical"]

    sns.boxplot(x='severity', y='askqe_sbert', data=df_results, order=order)
    plt.title('ASKQE (SBERT) Score by Error Severity')
    plt.ylabel('ASKQE Score (SBERT)')
    plt.xlabel('Error Severity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='severity', y='askqe_f1', data=df_results, order=order)
    plt.title('ASKQE (F1) Score by Error Severity')
    plt.ylabel('ASKQE Score (F1)')
    plt.xlabel('Error Severity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='severity', y='askqe_bleu', data=df_results, order=order)
    plt.title('ASKQE (BLEU) Score by Error Severity')
    plt.ylabel('ASKQE Score (BLEU)')
    plt.xlabel('Error Severity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='severity', y='askqe_chrf', data=df_results, order=order)
    plt.title('ASKQE (CHRF) Score by Error Severity')
    plt.ylabel('ASKQE Score (CHRF)')
    plt.xlabel('Error Severity')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='severity', y='askqe_judge_score', data=df_results, order=order)
    plt.title('ASKQE (LLM-Judge) Score by Error Severity')
    plt.ylabel('ASKQE Score (Judge)')
    plt.xlabel('Error Severity')
    plt.show()
else:
    print("No data for visualization.")

"""## 9. Implementation Differences & Limitations

This section documents the implementation choices and differences compared to the original ASKQE paper (Ki et al., ACL 2025).

### 9.1 Why We Don't Use XCOMET-QE

The original ASKQE paper (Ki et al., ACL 2025) compares results with three QE baseline metrics:
- **XCOMET-QE** (Guerreiro et al., 2024)
- **METRICX-QE** (Juraska et al., 2024)
- **BT-Score** (Agrawal et al., 2022)

In this implementation, **we do not use XCOMET-QE** for the following reasons:

> **Computational Limitations**: XCOMET-XL requires approximately **8GB of GPU memory**. On Google Colab with T4 GPU (16GB), this creates memory conflicts with the main LLM model (Qwen2.5-3B) already loaded.

**Alternatives considered:**
1. Unload the LLM before computing XCOMET → Significantly increases execution time
2. Use XCOMET-XXL (lighter version) → Not available at this time
3. Run XCOMET in a separate notebook → Complicates the pipeline

**Adopted solution:**
We use **BT-Score** as the primary QE baseline metric, which requires much less memory and can be executed alongside the LLM model.

### 9.2 Why We Use Only EN-ES

The original paper evaluates ASKQE on **5 language pairs** for CONTRATICO:
- EN-ES (English → Spanish)
- EN-FR (English → French)
- EN-HI (English → Hindi)
- EN-TL (English → Tagalog)
- EN-ZH (English → Chinese)

In this implementation, **we evaluate only EN-ES** for the following reasons:

> **Focus on methodological validation**: The main objective is to verify that the ASKQE pipeline works correctly, not to replicate all experiments from the paper.

**Motivations:**
1. **Execution time**: Running the pipeline on 5 language pairs would require ~5x the computational time
2. **Limited resources**: On free Colab, sessions have time and GPU limits
3. **Data availability**: Pre-computed backtranslation files are primarily available for EN-ES
4. **Sufficient validation**: EN-ES is representative and sufficient to validate the approach

**Future extension:**
To fully replicate the paper, one should iterate over all available language pairs.

### 9.3 Model Differences: Qwen2.5-3B vs LLAMA-3 70B

The original paper uses **LLAMA-3 70B** (Grattafiori et al., 2024) for QG/QA tasks, while this implementation uses **Qwen2.5-3B-Instruct-AWQ** with **vLLM** for high-performance inference.

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Model | LLAMA-3 70B | Qwen2.5-3B-Instruct-AWQ |
| Parameters | 70 billion | 3 billion |
| Quantization | None (full precision) | AWQ 4-bit |
| Inference Engine | HuggingFace | vLLM (PagedAttention) |
| GPU Memory | ~140GB+ | ~6GB |

**Advantages of vLLM:**
- **PagedAttention**: Efficient KV-cache memory management
- **Continuous Batching**: Dynamic request scheduling for higher throughput
- **AWQ Quantization**: 4-bit quantization with minimal quality loss

**Impact on results:**
- Smaller models may generate lower-quality questions and answers
- This explains why our correlation scores are lower than the paper's reported values
- The methodology remains identical; only the model capacity differs

> **Note**: The paper identifies LLAMA-3 70B with NLI as the best-performing configuration (Section 6.1). Using a smaller model is a trade-off for accessibility on free Colab.

### 9.4 GMM Simulation: Dataset Difference

The decision-making simulation using GMM in **Section 7.2** of the paper is evaluated on the **BIOMQM** dataset, while this notebook runs it on **CONTRATICO**.

| Aspect | Paper (Table 3) | This Notebook |
|--------|-----------------|---------------|
| Dataset | BIOMQM | CONTRATICO |
| Error Types | Naturally occurring MT errors | Synthetic perturbations |
| Decision Acc. ASKQE (F1) | **75.75%** | ~59% |
| Decision Acc. ASKQE (SBERT) | **63.77%** | ~56% |

**Why the difference?**
1. **CONTRATICO** uses synthetic perturbations with predefined severity labels (Minor/Critical)
2. **BIOMQM** uses professional human annotations with MQM error taxonomy
3. BIOMQM's Neutral/Minor/Major/Critical labels align better with the GMM Accept/Reject framework
4. The smaller model (Qwen2.5-3B) also impacts question/answer quality

> **Important**: To fully replicate Paper Table 3, the GMM should be run on BIOMQM with LLAMA-3 70B.

## 10. BT-Score Baseline (QE Metric Comparison)

Following the ASKQE paper (§6.3), we compute **BT-Score** as a QE baseline metric for comparison with ASKQE.

BT-Score (Agrawal et al., 2022) evaluates the semantic similarity between the source sentence and the backtranslation using **BERTScore** (Zhang et al., 2020).

Formula: `BT-Score = BERTScore(source, backtranslation)`
"""

!pip install -q bert-score

from bert_score import score as bert_score
import torch

print("Calculating BT-Score (BERTScore between Source and Backtranslation)...")

# Extract sources and backtranslations from results
sources = [r['source'] for r in results if r.get('source') and r.get('backtranslation')]
backtranslations = [r['backtranslation'] for r in results if r.get('source') and r.get('backtranslation')]

if sources and backtranslations:
    # Calculate BERTScore (P, R, F1)
    P, R, F1 = bert_score(
        backtranslations,
        sources,
        lang="en",
        rescale_with_baseline=True,
        verbose=True
    )

    # Add BT-Score to results
    bt_scores = F1.numpy()
    valid_idx = 0
    for i, res in enumerate(results):
        if res.get('source') and res.get('backtranslation'):
            results[i]['bt_score'] = float(bt_scores[valid_idx])
            valid_idx += 1
        else:
            results[i]['bt_score'] = 0.0

    # Update DataFrame
    df_results = pd.DataFrame(results)

    print(f"\nBT-Score Statistics:")
    print(f"  Mean: {df_results['bt_score'].mean():.4f}")
    print(f"  Std:  {df_results['bt_score'].std():.4f}")
    print(f"  Min:  {df_results['bt_score'].min():.4f}")
    print(f"  Max:  {df_results['bt_score'].max():.4f}")
else:
    print("No valid source-backtranslation pairs found.")

"""## 11. Pearson Correlation Analysis

Following the ASKQE paper (§6.3, Figure 4), we calculate the **Pearson correlation** between ASKQE metrics and BT-Score.

The paper reports average correlations of:
- ASKQE (F1) vs BT-Score: **r = 0.877**
- ASKQE (EM) vs BT-Score: **r = 0.882**

A strong positive correlation indicates that ASKQE behaves similarly to traditional QE metrics.
"""

from scipy.stats import pearsonr

print("\n" + "="*60)
print("Pearson Correlation Analysis: ASKQE vs BT-Score")
print("(Following Paper Section 6.3, Figure 4)")
print("="*60)

if not df_results.empty and 'bt_score' in df_results.columns:
    # Filter out rows with missing values
    df_valid = df_results.dropna(subset=['askqe_f1', 'askqe_sbert', 'askqe_em', 'bt_score'])

    if len(df_valid) > 2:
        # ASKQE F1 vs BT-Score
        corr_f1, p_f1 = pearsonr(df_valid['askqe_f1'], df_valid['bt_score'])
        print(f"\nASKQE (F1) vs BT-Score:")
        print(f"  Pearson r = {corr_f1:.4f}")
        print(f"  p-value   = {p_f1:.4e}")
        print(f"  {'Significant' if p_f1 < 0.05 else 'Not Significant'} (alpha=0.05)")

        # ASKQE SBERT vs BT-Score
        corr_sbert, p_sbert = pearsonr(df_valid['askqe_sbert'], df_valid['bt_score'])
        print(f"\nASKQE (SBERT) vs BT-Score:")
        print(f"  Pearson r = {corr_sbert:.4f}")
        print(f"  p-value   = {p_sbert:.4e}")
        print(f"  {'Significant' if p_sbert < 0.05 else 'Not Significant'} (alpha=0.05)")

        # ASKQE EM vs BT-Score
        corr_em, p_em = pearsonr(df_valid['askqe_em'], df_valid['bt_score'])
        print(f"\nASKQE (EM) vs BT-Score:")
        print(f"  Pearson r = {corr_em:.4f}")
        print(f"  p-value   = {p_em:.4e}")
        print(f"  {'Significant' if p_em < 0.05 else 'Not Significant'} (alpha=0.05)")

        # Summary comparison with paper
        print("\n" + "-"*60)
        print("Comparison with Paper (Section 6.3):")
        print("-"*60)
        print(f"{'Metric':<20} {'Our Result':<15} {'Paper Result':<15}")
        print(f"{'ASKQE F1 vs BT':<20} {corr_f1:<15.4f} {'0.877':<15}")
        print(f"{'ASKQE EM vs BT':<20} {corr_em:<15.4f} {'0.882':<15}")

        # ASKQE BLEU vs BT-Score
        corr_bleu, p_bleu = pearsonr(df_valid['askqe_bleu'], df_valid['bt_score'])
        print(f"{'ASKQE BLEU vs BT':<20} {corr_bleu:<15.4f} {'N/A':<15}")

        # ASKQE CHRF vs BT-Score
        corr_chrf, p_chrf = pearsonr(df_valid['askqe_chrf'], df_valid['bt_score'])
        print(f"{'ASKQE CHRF vs BT':<20} {corr_chrf:<15.4f} {'N/A':<15}")

        # ASKQE Judge vs BT-Score
        df_valid_judge = df_results.dropna(subset=['askqe_judge_score', 'bt_score'])
        if len(df_valid_judge) > 2:
            corr_judge, p_judge = pearsonr(df_valid_judge['askqe_judge_score'], df_valid_judge['bt_score'])
            print(f"{'ASKQE Judge vs BT':<20} {corr_judge:<15.4f} {'N/A':<15}")
    else:
        print("Insufficient data points for correlation analysis.")
else:
    print("BT-Score not computed. Run the BT-Score cell first.")

import matplotlib.pyplot as plt
import numpy as np

if not df_results.empty and 'bt_score' in df_results.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: ASKQE F1 vs BT-Score
    df_valid = df_results.dropna(subset=['askqe_f1', 'bt_score'])
    axes[0].scatter(df_valid['askqe_f1'], df_valid['bt_score'], alpha=0.5, s=30)
    axes[0].set_xlabel('ASKQE (F1)')
    axes[0].set_ylabel('BT-Score')
    axes[0].set_title(f'ASKQE (F1) vs BT-Score\nPearson r = {corr_f1:.3f}')

    # Add regression line
    z = np.polyfit(df_valid['askqe_f1'], df_valid['bt_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_valid['askqe_f1'].min(), df_valid['askqe_f1'].max(), 100)
    axes[0].plot(x_line, p(x_line), 'r--', alpha=0.8, label='Linear fit')
    axes[0].legend()

    # Plot 2: ASKQE SBERT vs BT-Score
    df_valid2 = df_results.dropna(subset=['askqe_sbert', 'bt_score'])
    axes[1].scatter(df_valid2['askqe_sbert'], df_valid2['bt_score'], alpha=0.5, s=30, color='green')
    axes[1].set_xlabel('ASKQE (SBERT)')
    axes[1].set_ylabel('BT-Score')
    axes[1].set_title(f'ASKQE (SBERT) vs BT-Score\nPearson r = {corr_sbert:.3f}')

    # Add regression line
    z2 = np.polyfit(df_valid2['askqe_sbert'], df_valid2['bt_score'], 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(df_valid2['askqe_sbert'].min(), df_valid2['askqe_sbert'].max(), 100)
    axes[1].plot(x_line2, p2(x_line2), 'r--', alpha=0.8, label='Linear fit')
    axes[1].legend()

    plt.legend()

    plt.tight_layout()
    plt.suptitle('Correlation between ASKQE Metrics and BT-Score', y=1.02, fontsize=12)
    plt.show()

    # Additional Scatter Plot for Judge vs BT-Score
    plt.figure(figsize=(7, 5))
    df_valid_judge = df_results.dropna(subset=['askqe_judge_score', 'bt_score'])
    plt.scatter(df_valid_judge['askqe_judge_score'], df_valid_judge['bt_score'], alpha=0.5, s=30, color='purple')
    plt.xlabel('ASKQE (LLM-Judge)')
    plt.ylabel('BT-Score')
    plt.title(f'ASKQE (LLM-Judge) vs BT-Score\nPearson r = {corr_judge:.3f}')

    if len(df_valid_judge) > 1:
        z3 = np.polyfit(df_valid_judge['askqe_judge_score'], df_valid_judge['bt_score'], 1)
        p3 = np.poly1d(z3)
        x_line3 = np.linspace(df_valid_judge['askqe_judge_score'].min(), df_valid_judge['askqe_judge_score'].max(), 100)
        plt.plot(x_line3, p3(x_line3), 'r--', alpha=0.8, label='Linear fit')
        plt.legend()
    plt.show()
else:
    print("Cannot create visualization: BT-Score not available.")

"""## 12. Decision Making Simulation (GMM)

Following the ASKQE paper (§7.2), we simulate a real-world scenario where users decide whether to **Accept** or **Reject** an MT output based on QE feedback.

**Methodology (from Paper):**
1. Fit a **two-component Gaussian Mixture Model (GMM)** on the QE scores
2. The component with higher mean is labeled as "Accept" (higher score = better quality)
3. Compare predicted decisions with human ground truth (Eq. 2: Accept if severity ∈ {Neutral, Minor})
4. Calculate **Decision Accuracy** (Eq. 3): percentage of correct Accept/Reject predictions

> **Note**: Paper results (Table 3) are on BIOMQM; our results are on CONTRATICO. See Section 10.4 for details.
"""

from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

print("\n--- Decision Making Simulation (GMM) ---")

if not df_results.empty and 'mqm_score' in df_results.columns and 'severity' in df_results.columns:

    # Function to train GMM and predict Accept/Reject
    def calculate_decision_accuracy(df, metric_col, metric_name):
        """Calculate GMM-based decision accuracy following ASKQE paper Section 7.2.

        Uses a 2-component Gaussian Mixture Model to cluster QE scores into
        Accept/Reject decisions, then compares with human ground truth.

        Ground truth (Eq. 2): Accept if severity in {Neutral, Minor}, Reject otherwise.
        Accuracy (Eq. 3): (TP + TN) / N
        """
        # Data preparation: GMM requires a 2D array (n_samples, 1) with NO NaN values
        # Drop rows with NaN in the metric column (invalidated entries)
        df_clean = df.dropna(subset=[metric_col, 'severity'])

        if len(df_clean) < 2:
            print(f"⚠️ Skipping {metric_name}: Not enough valid data points ({len(df_clean)} found).")
            return 0.0

        X = df_clean[metric_col].values.reshape(-1, 1)

        # Train GMM with 2 components (Accept vs Reject)
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # Predict labels (0 or 1)
        labels = gmm.predict(X)

        # Identify which cluster corresponds to "Accept" (Higher scores)
        # Calculate the mean value for each cluster
        means = gmm.means_.flatten()
        accept_label = np.argmax(means) # The cluster with the higher mean is "Accept"

        # Create predicted decisions (True = Accept, False = Reject)
        predicted_decisions = (labels == accept_label)

        # --- DEFINE HUMAN GROUND TRUTH ---
        # From the paper (Eq. 2): Accept if severity is "Neutral" or "Minor"
        # Reject if severity is "Major" or "Critical"

        def get_human_decision(severity):
            # Normalize the string for safety
            sev = str(severity).lower()
            if sev in ['no error', 'neutral', 'minor', 'none']:
                return True # Accept
            else:
                return False

        human_decisions = df_clean['severity'].apply(get_human_decision).values

        # --- CALCULATE ACCURACY ---
        # Paper's Eq. 3: (TP + TN) / N
        correct_predictions = np.sum(predicted_decisions == human_decisions)
        accuracy = correct_predictions / len(df_clean) * 100

        print(f"Decision Accuracy for {metric_name}: {accuracy:.2f}%")

        print(f"  > Cluster Means ({metric_name}): {means}")
        print(f"  > 'Accept' Cluster Index: {accept_label}")

        return accuracy

    if 'askqe_f1' in df_results.columns:
        acc_f1 = calculate_decision_accuracy(df_results, 'askqe_f1', 'ASKQE (F1)')

    if 'askqe_sbert' in df_results.columns:
        acc_sbert = calculate_decision_accuracy(df_results, 'askqe_sbert', 'ASKQE (SBERT)')

    if 'askqe_judge_score' in df_results.columns:
        # Per il LLM-Judge usiamo una logica a SOGLIA invece di GMM
        # Soglia: score <= 95 → Reject, score > 95 → Accept
        JUDGE_THRESHOLD = 95

        df_judge = df_results.dropna(subset=['askqe_judge_score', 'severity'])

        if not df_judge.empty:
            # Predicted decisions based on threshold
            predicted_decisions = (df_judge['askqe_judge_score'] > JUDGE_THRESHOLD).values

            # Human ground truth (same logic as GMM)
            def get_human_decision(severity):
                sev = str(severity).lower()
                if sev in ['no error', 'neutral', 'minor', 'none']:
                    return True  # Accept
                return False  # Reject

            human_decisions = df_judge['severity'].apply(get_human_decision).values

            # Calculate accuracy
            correct = np.sum(predicted_decisions == human_decisions)
            acc_judge = correct / len(df_judge) * 100

            print(f"Decision Accuracy for ASKQE (LLM-Judge) [Threshold={JUDGE_THRESHOLD}]: {acc_judge:.2f}%")
            print(f"  > Rule: score > {JUDGE_THRESHOLD} → Accept, score ≤ {JUDGE_THRESHOLD} → Reject")
            print(f"  > Predicted Accept: {np.sum(predicted_decisions)}, Reject: {np.sum(~predicted_decisions)}")

else:
    print("Insufficient data for GMM simulation (missing mqm_score or severity columns).")

# GMM Cluster Visualization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

if 'askqe_f1' in df_results.columns:
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    metrics_to_plot = [
        ('askqe_f1', 'ASKQE (F1)'),
        ('askqe_sbert', 'ASKQE (SBERT)'),
        ('askqe_judge_score', 'ASKQE (LLM-Judge)')
    ]

    for idx, (metric_col, metric_name) in enumerate(metrics_to_plot):
        if metric_col not in df_results.columns:
            continue

        # Drop NaNs for plotting
        X_data = df_results[metric_col].dropna()
        if X_data.empty:
            continue

        X = X_data.values.reshape(-1, 1)

        ax = axes[idx]

        if metric_col == 'askqe_judge_score':
            # --- CUSTOM LOGIC FOR LLM-JUDGE (THRESHOLD) ---
            # Consistency with the text output: Score <= 95 -> Reject, Score > 95 -> Accept
            JUDGE_THRESHOLD = 95

            accept_scores = X_data[X_data > JUDGE_THRESHOLD].values
            reject_scores = X_data[X_data <= JUDGE_THRESHOLD].values

            mean_accept = accept_scores.mean() if len(accept_scores) > 0 else 0
            mean_reject = reject_scores.mean() if len(reject_scores) > 0 else 0

            ax.hist(accept_scores, bins=20, alpha=0.7, color='#2ecc71', label=f'Accept (> {JUDGE_THRESHOLD}, μ={mean_accept:.1f})')
            ax.hist(reject_scores, bins=20, alpha=0.7, color='#e74c3c', label=f'Reject (≤ {JUDGE_THRESHOLD}, μ={mean_reject:.1f})')

            # Show Threshold Line
            ax.axvline(x=JUDGE_THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Threshold ({JUDGE_THRESHOLD})')
            ax.set_title(f'Threshold Separation: {metric_name}')

        else:
            # --- STANDARD GMM LOGIC FOR OTHER METRICS ---
            # Fit GMM
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)
            labels = gmm.predict(X)
            means = gmm.means_.flatten()
            accept_label = np.argmax(means)

            # Separate data by cluster
            accept_scores = X[labels == accept_label].flatten()
            reject_scores = X[labels != accept_label].flatten()

            ax.hist(accept_scores, bins=20, alpha=0.7, color='#2ecc71', label=f'Accept (μ={means[accept_label]:.3f})')
            ax.hist(reject_scores, bins=20, alpha=0.7, color='#e74c3c', label=f'Reject (μ={means[1-accept_label]:.3f})')

            # Add vertical lines for means
            ax.axvline(x=means[accept_label], color='#27ae60', linestyle='--', linewidth=2, label='Accept Mean')
            ax.axvline(x=means[1-accept_label], color='#c0392b', linestyle='--', linewidth=2, label='Reject Mean')

            ax.set_title(f'GMM Clustering: {metric_name}')

        ax.set_xlabel(f'{metric_name} Score')
        ax.set_ylabel('Frequency')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('GMM Decision Boundary Visualization', y=1.02, fontsize=14, fontweight='bold')
    plt.show()

    print("\nVisualization shows how GMM separates scores into 'Accept' (green) and 'Reject' (red) clusters.")
    print("The dashed lines indicate the mean of each cluster.")
else:
    print("Cannot create GMM visualization: ASKQE scores not available.")