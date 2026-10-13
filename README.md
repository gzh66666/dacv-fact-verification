# DACV: A Dimension-Decoupled and Adversarial Framework for Auditable Complex Claim Verification

Reference implementation of **DACV**, a dimension-decoupled adversarial framework
for auditable complex claim verification. DACV separates *evidence-grounded
dimension-level assessment* from *deterministic label composition*: a claim is
decomposed into event (E), attribution (A) and quantity (Q) sub-claims, each
dimension is verified through evidence-grounded pro/con debate followed by
neutral adjudication, and the resulting verdicts are composed into the final
label by an explicit, fixed rule set.

**Three stages.**

| Stage | Entry point | Role |
|---|---|---|
| **① Dimension-aware decomposition** | `code/agents/decomposer.py` | Splits a claim into E/A/Q sub-claims with focus questions and emits the auxiliary framing field `F_auxiliary`. |
| **② Dimension-level adversarial debate and verification** | `code/stage2_runner.py` | BM25 retrieval from the fixed evidence pool, Pro/Con argumentation (`code/agents/debater.py`), neutral dimension-level adjudication (`code/agents/dim_verifier.py`), optional gated web search. |
| **③ Deterministic label composition** | `code/stage3_runner.py` | Applies the composition rules `g(·)` of `code/rules/label_rules.py` and prints accuracy, per-class precision/recall/F1, macro-F1 and the confusion matrix. **Stage 3 makes no LLM call.** |

Supporting component: the fixed initial evidence pool consumed by Stage 2 is
built by `code/rules/evidence_pool.py` from the dataset sentence-level
`is_evidence` annotations. It is not a learned stage and makes no LLM call.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Environment and dependencies](#environment-and-dependencies)
3. [Datasets](#datasets)
4. [Running the pipeline](#running-the-pipeline)
5. [Rules and prompts](#rules-and-prompts)
6. [License](#license)
7. [Citation](#citation)
8. [Contact](#contact)
9. [中文说明](#中文说明)

---

## Repository layout

```
.
├── code/
│   ├── config.py                      Model / endpoint / temperature resolution from environment
│   │                                  variables. No credential is stored in this file.
│   ├── PromptH.py                     Prompt loader: resolves <name>.txt from DACV_PROMPT_DIR,
│   │                                  code/prompts/, then prompts/ at the package root.
│   ├── run_logger.py                  Run-audit logging helper, used when --run_dir is given.
│   ├── stage2_runner.py               Stage 2 entry point.
│   ├── stage3_runner.py               Stage 3 entry point (rule-only; no LLM call).
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── _llm_utils.py              Shared JSON extraction/repair helpers and llm_config wrapping.
│   │   ├── decomposer.py              Stage 1: dimension-aware decomposition + F_auxiliary framing field.
│   │   ├── debater.py                 Pro/Con debater agents (Stage 2).
│   │   └── dim_verifier.py            Neutral dimension-level adjudicator (Stage 2).
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── evidence_pool.py           Fixed initial evidence pool builder (is_evidence annotations).
│   │   └── label_rules.py             g(·): composition rules, STRONG_FRAMING_TERMS,
│   │                                  LITERAL_MISMATCH_SIGNALS, detect_literal_mismatch().
│   ├── search/
│   │   ├── __init__.py
│   │   ├── bm25_retrieval.py          BM25 retrieval over the frozen evidence pool.
│   │   └── serper_client.py           Gated Serper web-search client; used only with --use_search.
│   └── ablations/
│       ├── wo_decomposition/run_wo_decomposition.py
│       ├── wo_stage2_cot/run_wo_stage2_cot.py
│       └── wo_label_rule/run_wo_label_rule.py
├── prompts/                           Released prompt set: 4 roles × {en, ch} = 8 files.
│   ├── Decomposer_{en,ch}.txt
│   ├── Pro_Debater_{en,ch}.txt
│   ├── Con_Debater_{en,ch}.txt
│   └── Dim_Verifier_{en,ch}.txt
├── configs/
│   ├── env.example.md                 Environment-variable reference.
│   └── example_rawfc_pipeline.ps1     Example Stage 0 → Stage 3 commands.
├── LICENSE                            MIT license text.
├── requirements.txt                   Runtime dependencies.
├── .gitignore                         Excludes caches, logs, results, local data and .env.
└── README.md                          This file.
```

Prompt files are resolved by `code/PromptH.py` in this order: `DACV_PROMPT_DIR`
(if set), `code/prompts/`, then `prompts/` at the package root, which is the
released location. Within a run, the released pipeline uses the English prompts
for `--dataset RAWFC` and `--dataset LIAR-RAW`; the Chinese variants are
included for reference only.

---

## Environment and dependencies

### Python and packages

* **Python 3.10 or newer.** The released code uses PEP 604 `X | Y` type unions
  (for example in `code/run_logger.py`) and built-in generics such as
  `tuple[str, str]`. The reference environment used for the reported runs is
  `Python >= 3.10 (reference run: 3.11.5)`.
* Dependencies (`requirements.txt`):

  ```
  pyautogen>=0.2,<0.3
  openai>=1.0
  rank_bm25>=0.2.2
  ```

  The packages imported by the released code at run time are `autogen`
  (provided by `pyautogen`) and `rank_bm25`. The Serper client uses the Python
  standard library only (`http.client`).

```powershell
# from the package root
python -m pip install -r requirements.txt
```

### Model endpoint and credentials

The code talks to an OpenAI-compatible chat endpoint and builds the client
configuration entirely from the environment (`code/config.py`):

* endpoint: `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`);
* model identifier: `DACV_MODEL_ID` (default `deepseek-flash`), sent as the
  `model` field of every request;
* sampling temperature: `DACV_TEMPERATURE` (default `0.1`), shared by all
  agents;
* provider-side extended thinking is requested as **disabled** for every call,
  and the AutoGen cache seed is fixed at `42`.

`--model_name` on the command line selects the configuration key and must be
`deepseek-flash`; the legacy alias `deepseek-v4-flash` is accepted and resolves
to the same key (`code/config.py`). Any other value raises `ValueError`, and an
unset `DEEPSEEK_API_KEY` aborts the run with an explicit error.

```powershell
# placeholders only — set these in your shell, never in the source tree
$env:DEEPSEEK_API_KEY  = "<YOUR_DEEPSEEK_API_KEY>"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DACV_MODEL_ID     = "deepseek-flash"
$env:DACV_TEMPERATURE  = "0.1"

# optional, only for --use_search
$env:SERPER_API_KEY    = "<YOUR_SERPER_API_KEY>"
```

### Environment variables read by the code

| Variable | Default | Read by | Purpose |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | *(none — required)* | `code/config.py` | LLM credential. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | `code/config.py` | OpenAI-compatible endpoint. |
| `DACV_MODEL_ID` | `deepseek-flash` | `code/config.py` | Model identifier sent to the endpoint. |
| `DACV_TEMPERATURE` | `0.1` | `code/config.py` | Sampling temperature shared by all agents. |
| `DACV_API_SEED` | unset | `code/agents/decomposer.py`, `code/stage2_runner.py` | Optional provider-side sampling seed, recorded in the run audit. |
| `DACV_MAX_TOKENS` | unset | `code/agents/decomposer.py`, `code/stage2_runner.py` | Optional completion cap, recorded in the run audit. |
| `SERPER_API_KEY` | *(none)* | `code/search/serper_client.py` | Required only when `--use_search` is passed. |
| `DACV_PROMPT_DIR` | `<package>/prompts` | `code/PromptH.py` | Overrides prompt lookup. |
| `DELPHIAGENT_ROOT` | unset | `code/PromptH.py` | Additional fallback prompt directory (`<root>/prompts`). |
| `DACV_RAWFC_DATA` | unset | `configs/example_rawfc_pipeline.ps1` | Convenience variable for the dataset root in the example script. |

`code/config.py` reads the process environment first and, on Windows, falls back
to the `HKCU\Environment` registry values, so credentials stored in the user
environment are picked up without reopening the shell. All credential handling
is environment-based: no key is stored in the source tree.

The same reference, in the same form, is in
[`configs/env.example.md`](configs/env.example.md).

---

## Datasets

RAWFC and LIAR-RAW are public third-party datasets and are downloaded from their
official distribution:

| Dataset | Official distribution |
|---|---|
| **RAWFC** | [`Nicozwy/CofCED` → `Datasets/RAWFC`](https://github.com/Nicozwy/CofCED/tree/main/Datasets/RAWFC) |
| **LIAR-RAW** | [`Nicozwy/CofCED` → `Datasets/LIAR-RAW`](https://github.com/Nicozwy/CofCED/tree/main/Datasets/LIAR-RAW) |

Both datasets are distributed through
[`Nicozwy/CofCED`](https://github.com/Nicozwy/CofCED).

Please observe the terms of use stated by the original dataset authors and cite
their publication when you use either dataset.

### On-disk layout expected by the code

Each split is a directory holding one JSON record per claim:

```
<DATA_ROOT>/RAWFC/{train,val,test}/<event_id>.json
<DATA_ROOT>/LIAR-RAW/{train,val,test}/<event_id>.json
```

The loaders read the following fields:

* `event_id`, `claim`, `label` — used by `code/agents/decomposer.py`,
  `code/rules/evidence_pool.py` and the Stage 2/3 runners;
* `reports[].tokenized[].sent`, `reports[].tokenized[].is_evidence`,
  `reports[].tokenized[].is_repeat` — read by `code/rules/evidence_pool.py`,
  which keeps the sentences with `is_evidence == 1` and skips those with
  `is_repeat == 1`, de-duplicating case-insensitively.

`--data_dir` conventions:

* `code/rules/evidence_pool.py --data_dir` is the **split directory** itself,
  e.g. `<DATA_ROOT>/RAWFC/test`;
* `code/agents/decomposer.py --data_dir` is a root that contains
  `<DATASET>/<split>/` or `<split>/`; both are probed.

### Derived artifacts written by the pipeline

```
results/evidence_pool/<DATASET>/<split>/<event_id>.json   # evidence pool
results/decomposition/<DATASET>/<split>/<event_id>.json   # Stage 1
results/stage2/<DATASET>/<split>/<event_id>.json          # Stage 2
results/stage3/<DATASET>/<split>/<event_id>.json          # Stage 3
```

The commands below always pass these directories explicitly. When a directory
flag is omitted, the scripts under `code/` resolve their defaults relative to
the `code/` directory itself (for example `code/results/decomposition`), so the
explicit form shown here is the recommended one.

---

## Running the pipeline

All commands are run from the package root. Copy-and-paste versions of the three
pipeline stages are in
[`configs/example_rawfc_pipeline.ps1`](configs/example_rawfc_pipeline.ps1).

```powershell
# dataset root that contains RAWFC/{train,val,test}/
$DataDir = "<PATH_TO_DATA_ROOT>"
```

### Evidence pool (preprocessing for Stage 2)

```powershell
python .\code\rules\evidence_pool.py `
  --data_dir "$DataDir\RAWFC\test" `
  --output_dir ".\results\evidence_pool\RAWFC\test" `
  --dataset RAWFC
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `--data_dir` | yes | — | The **split directory** holding `<event_id>.json` records. |
| `--output_dir` | yes | — | Destination directory for the per-claim pool files. |
| `--dataset` | no | `RAWFC` | `RAWFC` or `LIAR-RAW`. |

No LLM call and no API key are needed. Each output file contains `event_id`,
`claim`, `label`, `evidence_pool` and `pool_size`.

### Stage 1 — dimension-aware decomposition

```powershell
python .\code\agents\decomposer.py `
  --data_dir $DataDir `
  --dataset RAWFC --task test `
  --output_dir ".\results\decomposition" `
  --model_name deepseek-flash `
  --run_dir ".\logs\rawfc_run01" --run_id rawfc_run01
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `--data_dir` | yes | — | Root containing `<DATASET>/<task>/` or `<task>/`. |
| `--dataset` | yes | — | `RAWFC` or `LIAR-RAW`. |
| `--task` | no | `test` | `train`, `val` or `test`. |
| `--output_dir` | no | `code/results/decomposition` | Results root; files are written to `<output_dir>/<dataset>/<task>/`. |
| `--model_name` | no | `deepseek-flash` | Must resolve in `code/config.py` (`deepseek-v4-flash` is an accepted alias). |
| `--limit` | no | `-1` (all) | Cap on the number of claims. |
| `--skip_existing` | no | off | Resume an interrupted run. |
| `--run_dir`, `--run_id` | no | empty | Optional run-audit directory and stable run identifier. |

Each claim is written as `<event_id>.json` with `event_id`, `claim`, `label` and
a `decomposition` object holding `E_list`, `A`, `Q`, `F_auxiliary`, `parse_ok`
and `compound`.

### Stage 2 — dimension-level adversarial verification

```powershell
python .\code\stage2_runner.py `
  --dataset RAWFC --task test --rounds 1 `
  --decomp_dir ".\results\decomposition" `
  --pool_dir ".\results\evidence_pool" `
  --output_dir ".\results\stage2" `
  --model_name deepseek-flash `
  --run_dir ".\logs\rawfc_run01" --run_id rawfc_run01
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `--dataset` | yes | — | `RAWFC` or `LIAR-RAW`. |
| `--task` | no | `test` | `train`, `val` or `test`. |
| `--rounds` | no | `1` | `1` or `2` debate rounds. |
| `--decomp_dir` | no | `code/results/decomposition` | Root; `<dataset>/<task>` is appended. |
| `--pool_dir` | no | `code/results/evidence_pool` | Root; `<dataset>/<task>` is appended. |
| `--output_dir` | no | `code/results/stage2` | Root; `<dataset>/<task>` is appended. |
| `--model_name` | no | `deepseek-flash` | Resolved through `code/config.py`. |
| `--use_search` | no | off | Enables the gated Serper web search; requires `SERPER_API_KEY`. |
| `--topk_evidence` | no | `5` | BM25 sentences retrieved per dimension. |
| `--n_search_results` | no | `5` | Snippets added per external search call (capped at 10 by the client). |
| `--early_stop` | no | off | Stops the debate after `E = F` with high confidence when both A and Q are unverifiable. |
| `--limit`, `--event_ids`, `--skip_existing` | no | `-1`, empty, off | Debug and resume controls; `--event_ids` takes a comma-separated subset. |
| `--run_dir`, `--run_id` | no | empty | Optional run-audit directory and stable run identifier. |

For each claim, the runner retrieves evidence per dimension, runs the Pro and
Con debaters (a second round when `--rounds 2`), and asks the dimension verifier
for a `T`/`F`/`U` verdict with a confidence level. The output record carries
`event_id`, `claim`, `label`, `stage2_config`, `evidence_pool_size`,
`n_subclaims_debated`, `early_stopped`, `debates` and a `verdicts_summary` with
per-dimension verdicts.

### Stage 3 — deterministic label composition

```powershell
python .\code\stage3_runner.py `
  --dataset RAWFC --task test `
  --stage2_dir ".\results\stage2" `
  --decomp_dir ".\results\decomposition" `
  --output_dir ".\results\stage3" `
  --run_dir ".\logs\rawfc_run01" --run_id rawfc_run01
```

| Parameter | Required | Default | Notes |
|---|---|---|---|
| `--dataset` | yes | — | `RAWFC` or `LIAR-RAW`. |
| `--task` | no | `test` | `train`, `val` or `test`. |
| `--stage2_dir` | no | `code/results/stage2` | Root; `<dataset>/<task>` is appended. |
| `--decomp_dir` | no | `code/results/decomposition` | Root; supplies the Stage 1 `F_auxiliary` framing field. |
| `--output_dir` | no | `code/results/stage3` | Root; `<dataset>/<task>` is appended. |
| `--model_name` | no | `deepseek-flash` | Kept for command compatibility; Stage 3 makes no LLM call. |
| `--limit`, `--skip_existing` | no | `-1`, off | Debug and resume controls. |
| `--run_dir`, `--run_id` | no | empty | Optional run-audit directory and stable run identifier. |

Stage 3 reads the Stage 2 traces, applies `g(·)` and writes one record per claim
with `final_label`, `rule_label`, `rule_id`, `dim_verdicts`,
`framing_strength`, `framing_content` and `literal_mismatch`. It then prints the
per-class support/TP/FP/FN/precision/recall/F1 table, accuracy, macro-F1, the
confusion matrix and a sample of the errors.

### Ablation scripts

Three self-contained ablation variants ship with the package. Each mirrors one
row of the paper's ablation table, writes its outputs under
`<output_dir>/{stage2,stage3}/<dataset>/<task>/` (except `wo_label_rule`, which
writes `<output_dir>/<dataset>/<task>/`), and saves
`summary_<dataset>_<task>.json` in the output root. The default `--output_dir`
is `outputs/` inside each variant directory.

```powershell
# w/o dimension-aware decomposition
python .\code\ablations\wo_decomposition\run_wo_decomposition.py `
  --dataset RAWFC --task test `
  --pool_dir ".\results\evidence_pool" `
  --output_dir ".\results\ablations\wo_decomposition"

# w/o debate (verifier only, no pro/con transcripts)
python .\code\ablations\wo_stage2_cot\run_wo_stage2_cot.py `
  --dataset RAWFC --task test `
  --decomp_dir ".\results\decomposition" `
  --pool_dir ".\results\evidence_pool" `
  --output_dir ".\results\ablations\wo_stage2_cot"

# w/o compositional label rule (LLM aggregation replaces g(·))
python .\code\ablations\wo_label_rule\run_wo_label_rule.py `
  --dataset RAWFC --task test `
  --stage2_dir ".\results\stage2" `
  --decomp_dir ".\results\decomposition" `
  --output_dir ".\results\ablations\wo_label_rule"
```

| Script | Internal `ablation` tag | What it changes |
|---|---|---|
| `wo_decomposition/run_wo_decomposition.py` | `wo_dimensional_decomposition` | Treats the claim as one holistic event-type sub-claim; `A` and `Q` are set to `N/A` and no framing field is extracted. Retrieval, debate, adjudication and rule composition are unchanged. |
| `wo_stage2_cot/run_wo_stage2_cot.py` | `wo_stage2_cot` | The verifier judges each dimension directly, without receiving pro/con debate transcripts (`"debate": false`). |
| `wo_label_rule/run_wo_label_rule.py` | `wo_compositional_label_rule` | Replaces deterministic composition with an LLM aggregation prompt. |

| Parameter | `wo_decomposition` | `wo_stage2_cot` | `wo_label_rule` |
|---|---|---|---|
| `--model_name` | default `deepseek-v4-flash` | default `deepseek-v4-flash` | default `deepseek-v4-flash` |
| `--dataset` | `RAWFC` or `LIAR-RAW`, default `RAWFC` | same | same |
| `--task` | default `test` | default `test` | default `test` |
| `--pool_dir` | yes (default `code/results/evidence_pool`) | yes (default `code/results/evidence_pool`) | — |
| `--decomp_dir` | — | yes (default `code/results/decomposition`) | yes (default `code/results/decomposition`) |
| `--stage2_dir` | — | — | yes (default `code/results/stage2`) |
| `--output_dir` | default `<variant>/outputs` | same | same |
| `--rounds` | `1` or `2`, default `1` | — | — |
| `--use_search` | off | — (search is never used) | — |
| `--topk_evidence` | `5` | `5` | — |
| `--n_search_results` | `5` | — | — |
| `--limit`, `--skip_existing`, `--event_ids` | yes | yes | yes |

The default `--model_name` of the ablation scripts is the legacy alias
`deepseek-v4-flash`, which `code/config.py` resolves to `deepseek-flash`.

---

## Rules and prompts

Components are identified by the textual version tags below.

| Component | File | Version tag |
|---|---|---|
| Composition rules `g(·)` | `code/rules/label_rules.py` | **revision rules** |
| Framing dictionary | `code/rules/label_rules.py` → `STRONG_FRAMING_TERMS` | **released framing dictionary** — 17 terms |
| Literal-mismatch signatures | `code/rules/label_rules.py` → `LITERAL_MISMATCH_SIGNALS` | **released mismatch signatures** — 53 ordered regular expressions |
| Prompts | `prompts/` (8 files) | **released prompt set** — 4 roles × {en, ch} |
| Backbone configuration | `code/config.py` | `deepseek-flash` (legacy alias `deepseek-v4-flash`) |

### Composition rules in detail

`code/rules/label_rules.py` derives the final label from the dimension verdicts
(`T` true, `F` false, `U` unverifiable, `N/A` not applicable), the framing
strength of the Stage 1 `F_auxiliary` content (`strong` when a framing-dictionary
term matches, otherwise `weak`/`none`), the E-dimension confidence, and a
literal-mismatch flag computed by `detect_literal_mismatch()` over the combined
E and A verifier reasoning.

| Emitted `rule_id` | Condition | Label |
|---|---|---|
| `R1a` | `E = F` and a literal-mismatch signature fires | `half` |
| `R2` | `E = F` and `A = T` or `Q = T` | `half` |
| `R1b` | `E = F` and framing strength is `strong` | `half` |
| `R1c` | `E = F`, no adjustment applies | `false` |
| `R3.5` | `E = T`, `A` and `Q` ∈ {`T`, `N/A`}, and framing is `strong` or the E confidence is not `high` | `half` |
| `R3` | `E = T`, `A` and `Q` ∈ {`T`, `N/A`}, framing not `strong`, E confidence `high` | `true` |
| `R4` | `E = T` and `A = F` | `half` |
| `R5` | `E = T` and `Q = F` | `half` |
| `R6-true-exception` | `E = T`, at least one active `U`, at least one `T` among `A`/`Q`, and framing not `strong` | `true` |
| `R6-default` | all remaining verdict configurations, including all-`U` | `half` |

For compound claims, the per-event verdicts are aggregated before the rules run:
any `F` yields `F`; otherwise any `U` yields `U`; otherwise all-`T` yields `T`; an
empty event list yields `N/A`.

---

## License

Released under the **MIT License**; see [`LICENSE`](LICENSE) for the full text.
The copyright line names the five authors of the paper: Jie Hu, Zihan Guo,
Xin Yu, Dengke Gao and Tianrui Li.

The RAWFC and LIAR-RAW datasets are not covered by this repository's license;
they remain subject to the terms of their original distributors.

---

## Citation

The accompanying paper is under revision at *Information Processing & Management*.
Until the final version appears, please cite it as:

```bibtex
@article{dacv,
  author  = {Jie Hu and Zihan Guo and Xin Yu and Dengke Gao and Tianrui Li},
  title   = {DACV: A Dimension-Decoupled and Adversarial Framework for Auditable Complex Claim Verification},
  journal = {Information Processing \& Management},
  year    = {2026},
  note    = {Under revision},
  doi     = {to be assigned}
}
```

If you use RAWFC or LIAR-RAW, please also cite the original dataset publication
from the [CofCED repository](https://github.com/Nicozwy/CofCED).

---

## Contact

Questions and bug reports are welcome. For code-related questions, please open
an issue in this repository.

* repository / issue tracker: https://github.com/gzh66666/dacv-fact-verification

---

## 中文说明

本节仅为中文读者提供导航，内容与上文一致；如有出入，以上文（英文）为准。

**项目定位.** 本包是 DACV（维度解耦对抗式声明验证）的参考实现，实现三阶段流水线：
① 维度感知分解（`code/agents/decomposer.py`）→ ② 逐维度对抗辩论与核查
（`code/stage2_runner.py`）→ ③ 确定性规则组合（`code/stage3_runner.py`，**不调用 LLM**）。
其中 Stage 2 所需的固定证据池由 `code/rules/evidence_pool.py` 依据数据集的
`is_evidence` 标注预先构建。

**运行前提.** Python ≥ 3.10；`python -m pip install -r requirements.txt`；
需设置 `DEEPSEEK_API_KEY`（占位符，切勿写入源码），可选 `SERPER_API_KEY` 用于
`--use_search`。模型 endpoint、模型标识与温度分别由 `DEEPSEEK_BASE_URL`、
`DACV_MODEL_ID`（默认 `deepseek-flash`）、`DACV_TEMPERATURE`（默认 `0.1`）控制，
详细变量表见上文 [Environment and dependencies](#environment-and-dependencies)。

**数据.** RAWFC 与 LIAR-RAW 均为第三方公开数据集，请从官方地址
<https://github.com/Nicozwy/CofCED> 下载并保留官方划分，目录约定为
`<DATA_ROOT>/<DATASET>/<split>/<event_id>.json`。下载后即可按
[Datasets](#datasets) 与 [Running the pipeline](#running-the-pipeline) 的命令依次运行。

**规则与提示词.** 规则与提示词仅使用文字版本标识（revision rules、
released framing dictionary、released mismatch signatures、released prompt set），
被触发的规则以 `rule_id`（`R1a` … `R6-default`）写入 Stage 3 输出，见上文
[Composition rules in detail](#composition-rules-in-detail)。

**署名与引用.** 作者为 Jie Hu、Zihan Guo、Xin Yu、Dengke Gao（通讯作者）与 Tianrui Li；BibTeX 条目见 [Citation](#citation)。
公开仓库地址为 `https://github.com/gzh66666/dacv-fact-verification`，DOI 待正式发布
时补入。参考运行环境为 `Python >= 3.10 (reference run: 3.11.5)`。
