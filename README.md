# DACV: A Dimension-Decoupled and Adversarial Framework for Auditable Complex Claim Verification

Reference implementation of DACV, a three-stage pipeline for complex claim verification:

1. **Decomposition** — the claim is split into event (E), attribution (A) and quantity (Q) dimensions, together with an auxiliary framing field.
2. **Debate and verification** — a supporting and a refuting debater argue each dimension against the fixed evidence pool; a per-dimension verifier returns true / false / unresolved with a confidence level.
3. **Composition** — a deterministic rule cascade maps the dimension verdicts to the final label. Stage 3 makes no LLM call.

## Repository layout

```
code/
  rules/evidence_pool.py   build the fixed initial evidence pool (BM25)
  agents/decomposer.py     Stage 1: dimension-aware decomposition
  stage2_runner.py         Stage 2: per-dimension debate and verification
  stage3_runner.py         Stage 3: deterministic label composition
  agents/                  debater and verifier agents
  rules/label_rules.py     three-class composition rules
  search/                  BM25 retrieval and the gated Serper client
  config.py                environment and model configuration
  PromptH.py               prompt loader
  run_logger.py            per-run logging
prompts/                   released agent prompts (decomposer, two debaters, verifier)
configs/                   example environment file and pipeline script
```

## Environment

Python >= 3.10 (reference run: 3.11.5).

```bash
pip install -r requirements.txt
```

| Variable | Purpose |
| --- | --- |
| `DEEPSEEK_API_KEY` | model API key (required) |
| `DEEPSEEK_BASE_URL` | API endpoint (default `https://api.deepseek.com`) |
| `DACV_MODEL_ID` | model identifier (default `deepseek-flash`) |
| `DACV_TEMPERATURE` | sampling temperature (default `0.1`) |
| `SERPER_API_KEY` | only needed with `--use_search` |
| `DACV_PROMPT_DIR` | optional override for the prompt directory |

## Data

RAWFC and LIAR-RAW are publicly available; we use the copies distributed at
<https://github.com/Nicozwy/CofCED>. Place them as `<data_dir>/<dataset>/<split>/*.json`.

## Running the pipeline

```bash
# Stage 0: fixed initial evidence pool
python code/rules/evidence_pool.py --data_dir <data_dir> --output_dir <pool_dir>

# Stage 1: decomposition
python code/agents/decomposer.py --dataset RAWFC --data_dir <data_dir> --task test

# Stage 2: debate and verification (add --use_search for the gated web search)
python code/stage2_runner.py --dataset RAWFC --task test --rounds 1

# Stage 3: deterministic label composition
python code/stage3_runner.py --dataset RAWFC --task test
```

Outputs are written under `code/results/` by default; every script accepts `--output_dir`.
Run `--help` on any entry point for its full parameter list.

## Rules and prompts

The released label-composition rules are in `code/rules/label_rules.py`; the released
prompt set (decomposer, supporting debater, refuting debater, verifier) is in `prompts/`.

## License

Released under the MIT License; see [LICENSE](LICENSE).

## Citation

Please cite the accompanying paper (under revision at *Information Processing & Management*). Full citation details are omitted here to preserve anonymity during review.

## Contact

Questions and bug reports are welcome through the issue tracker of this repository
(<https://github.com/gzh66666/dacv-fact-verification>).
