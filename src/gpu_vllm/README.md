# GPU Pipeline: History-Shock 2x2 Experiment

Self-contained folder for the history-shock simulation. Upload to the GPU
server and run — no other project files needed.

## Experiment Design

**Research question:** How much of an LLM recommendation's effect comes from
*what* it recommends (retrieval) versus *how* it recommends (expression)?

**Factorial design:** 2 retrieval conditions (generic / history-aware) x
2 expression conditions (generic / history-aware) = 4 cells per consumer.

- **Generic:** LLM sees only product specs and consumer profile.
- **History-aware:** LLM additionally receives real Amazon review summaries,
  rating counts, popularity tiers, and segment-level affinity scores.

**Scope:** 60 consumer personas x 30 headphones (real Amazon products) x
4 cells = 240 recommendation packages.

## Models

| Role | Model | Backend | Notes |
|------|-------|---------|-------|
| Supply (recommendation generation) | Qwen2.5-32B-Instruct-AWQ | vLLM on GPU | temp=0.7, seeded |
| Demand (absolute + pairwise eval) | GPT-5.3 | OpenAI API | ~600 calls |
| Analysis | — | scipy/numpy | BT decomposition, bootstrap |

## Tested Environment

| Package | Tested Version | Notes |
|---------|---------------|-------|
| vllm | 0.17.0 | V1 engine default; requires chat_template patch |
| transformers | 4.57.6 | |
| torch | 2.10.0 | |
| tokenizers | 0.22.2 | |

## Folder Structure

```
gpu_vllm/
├── README.md                     # this file
├── requirements.txt              # Python dependencies (pip)
├── pyproject.toml                # Python dependencies (uv, alternative)
├── run_all.sh                    # master runner (supply → eval → analyze)
├── supply_history_shock.py       # Step 1: vLLM batched supply generation
├── eval_gpt.py                   # Step 2: GPT absolute + pairwise evaluation
├── analyze.py                    # Step 3: BT decomposition + bootstrap CIs
├── prompts.py                    # prompt templates (shared with local pipeline)
└── data/                         # bundled input data (~536 KB)
    ├── catalogs/
    │   └── headphones_catalog.csv          # 30 real Amazon products
    ├── personas/
    │   └── headphones_personas.json        # 60 GPT-generated consumer personas
    ├── history_dgp/
    │   └── headphones_history_qualitative.json  # review-based history summaries
    └── gpt_exemplars/
        └── final_few_shot_prompts.json     # few-shot examples for local model
```

---

## Step-by-Step Instructions

### 1. Upload to GPU server

```bash
# From local machine:
scp -P 22863 -r \
  "/Users/qhan94/Library/CloudStorage/Dropbox/MarketingResearch/LLMRec-Causal/src/gpu_vllm" \
  root@connect.westb.seetacloud.com:~/gpu_vllm
```

### 2. SSH into the server

```bash
ssh -p 22863 root@connect.westb.seetacloud.com
```

### 3. Install Python dependencies

```bash
cd ~/gpu_vllm

# Step A: Install vllm FIRST — it pins the correct CUDA-compatible torch.
# Do NOT separately install torch; vllm brings its own.
pip install vllm

# Step B: Install remaining dependencies AFTER vllm
pip install -r requirements.txt
```

### 4. Download the model

```bash
# Enable AutoDL mirror and network speed-up
export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo

# No auth required (public model). ~18 GB, takes 10-15 min.
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \
  --local-dir /root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
```

### 5. Set environment variables

```bash
export SUPPLY_MODEL=/root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
export DATA_DIR=~/llmrec_results
export OPENAI_API_KEY=sk-...   # replace with your actual key

# Persist across SSH reconnects
cat >> ~/.bashrc << 'ENVEOF'
export SUPPLY_MODEL=/root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
export DATA_DIR=~/llmrec_results
export OPENAI_API_KEY=sk-...
ENVEOF
source ~/.bashrc
```

### 6. Verify environment

```bash
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$SUPPLY_MODEL')
print(f'Tokenizer loaded: {len(t)} vocab')
"
```

### 7. Smoke test

```bash
cd ~/gpu_vllm
python supply_history_shock.py --smoke
```

Expected output:
```
============================================================
  Environment Diagnostics
============================================================
  vllm:         0.17.x
  transformers: 4.5x.x
  torch:        2.x.x
  CUDA:         12.x
  GPU:          ...
  model:        /root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
  model exists: True
  enforce_eager: True (skip torch.compile)
============================================================

=== headphones: 3 personas ===
[MODEL] Loading /root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ ...
[MODEL] Loaded.

Phase 1: Building 6 retrieval prompts...
  ...
Phase 2: Building 12 expression prompts...
  ...
Phase 3: Parsing and saving...

=== Supply Complete ===
  Total packages: 12
```

- Model loads in ~60-90 seconds
- `enforce_eager=True` skips torch.compile (avoids Triton kernel collision)
- 3 personas, 6 retrieval + 12 expression = 18 calls
- 12 packages generated, finishes in < 5 minutes
- No parse failures

### 8. Full run

```bash
cd ~/gpu_vllm
nohup bash run_all.sh > ~/run_history_shock.log 2>&1 &
echo $! > ~/run_pid.txt

# Monitor progress
tail -f ~/run_history_shock.log
```

### 9. Download results to local machine

```bash
# From local machine (not on GPU server):
mkdir -p data/gpu_results
scp -P 22863 -r \
  root@connect.westb.seetacloud.com:~/llmrec_results/ \
  data/gpu_results/
```

---

## Pipeline Steps

| Step | Script | Model | Calls | Est. Time |
|------|--------|-------|-------|-----------|
| 1 | `supply_history_shock.py --full` | Qwen2.5-32B-AWQ (vLLM) | 360 | ~10-20 min |
| 2 | `eval_gpt.py --all --resume` | GPT-5.3 (API) | ~600 | ~20-30 min |
| 3 | `analyze.py` | none | — | < 1 min |
| **Total** | | | **~960** | **~30-50 min** |

**Step 1 detail:** Supply generation uses vLLM batched inference in two phases:
- Phase 1: all 120 retrieval prompts at once (60 generic + 60 history)
- Phase 2: all 240 expression prompts at once (4 cells x 60 personas)

This is ~10-20x faster than sequential Ollama calls on a local machine.

**Step 2 detail:** GPT evaluation runs two sub-tasks:
- Absolute scoring: 240 calls (60 clusters x 4 cells), 8 outcome scales each
- Pairwise comparison: 360 calls (60 clusters x 6 pairs), randomized A/B order

---

## Output Files

```
~/llmrec_results/
├── final_supply_rows.csv           # 240 rows, main supply output
├── supply_cache.jsonl              # raw supply cache (for debugging)
├── absolute_eval_rows.csv          # 240 GPT absolute scores
├── absolute_eval_cache.jsonl       # resume cache
├── pairwise_eval_rows.csv          # 360 GPT pairwise comparisons
├── pairwise_eval_cache.jsonl       # resume cache
└── analysis/
    ├── decomposition_bootstrap.csv # R/E/I effects with bootstrap CIs
    ├── decomposition_full.json     # full bootstrap distributions
    ├── win_rates.csv               # pairwise win rates for all 6 pairs
    └── absolute_eval_by_cell.csv   # mean scores by cell (00/10/01/11)
```

## Resume After Crash

All scripts use JSONL caching. If a step crashes mid-run, re-running it
skips already-completed work:

```bash
# Resume the full pipeline (supply caches per-cluster, eval caches per-call)
nohup bash run_all.sh > ~/run_history_shock.log 2>&1 &
```

Or run individual steps:

```bash
python eval_gpt.py --absolute --resume   # resume absolute eval only
python eval_gpt.py --pairwise --resume   # resume pairwise eval only
```

## Replicability

- **Supply:** temperature=0.7, per-persona seeds
  (MASTER_SEED 20260515 + persona_idx * 100 + cell offset)
- **Evaluation:** GPT API is non-deterministic, but all raw responses are
  cached in JSONL files for exact reproduction of downstream analysis
- **Analysis:** bootstrap seed=42, B=2000 cluster-level resamples

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPPLY_MODEL` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | vLLM supply model path |
| `DATA_DIR` | `~/llmrec_results` | Output directory for all results |
| `OPENAI_API_KEY` | (required) | OpenAI API key for GPT evaluation |
| `OPENAI_MODEL` | `gpt-5.3-chat-latest` | GPT model for evaluation |
| `TENSOR_PARALLEL` | `1` | Number of GPUs for tensor parallelism |

---

## Troubleshooting

### "duplicate template name" error (vLLM 0.17+)

**Root cause:** vLLM 0.17's V1 engine uses `torch.compile` by default. On
some GPUs (e.g., Blackwell sm_120), PyTorch 2.10's Triton Inductor hits a
collision in GPU kernel template registration (`torch/_inductor/select_algorithm.py`).
The word "template" in the error refers to **Triton GPU kernels**, not
Jinja2 chat templates.

**Fix:** `supply_history_shock.py` passes `enforce_eager=True` to `LLM()`,
which disables `torch.compile` entirely. No tokenizer patches needed.

If you see this error, verify `enforce_eager=True` is present in the
`LLM()` call.

### Restore tokenizer (if previously patched)

If the tokenizer_config.json was modified during earlier debugging, restore it:
```bash
# From backup (if available)
cp $SUPPLY_MODEL/tokenizer_config.json.bak_before_chat_template_fix \
   $SUPPLY_MODEL/tokenizer_config.json

# Or re-download
export HF_ENDPOINT=https://hf-mirror.com
source /etc/network_turbo
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \
  --local-dir /root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
```

### Version mismatch

If you see import errors or unexpected behavior, verify versions:
```bash
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
```

The script prints these automatically at startup via Environment Diagnostics.

### Clean stale env vars (if previously set)

```bash
unset VLLM_USE_V1
sed -i '/VLLM_USE_V1/d' ~/.bashrc
```
