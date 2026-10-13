# Runtime configuration

All runtime settings are read from environment variables; no credential or
absolute path is stored in the source tree. `code/config.py` checks the process
environment first and, on Windows, also falls back to the `HKCU\Environment`
user-environment values.

## Required for any LLM run

| Variable | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | none | required; the run aborts with a clear error if unset |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `DACV_MODEL_ID` | `deepseek-flash` | model identifier sent to the endpoint |
| `DACV_TEMPERATURE` | `0.1` | sampling temperature shared by all agents |

`--model_name deepseek-v4-flash` is still accepted by `stage2_runner.py`,
`stage3_runner.py` and `config.py` and resolves to the same service; it exists
only so older command lines keep working.

## Optional

| Variable | Default | Notes |
|---|---|---|
| `DACV_API_SEED` | unset | provider-side sampling seed; recorded in the run audit |
| `DACV_MAX_TOKENS` | unset | completion cap; recorded in the run audit |
| `SERPER_API_KEY` | none | required only when `--use_search` is passed |
| `DACV_PROMPT_DIR` | `<package>/prompts` | overrides prompt lookup |
| `DELPHIAGENT_ROOT` | unset | only for reused DelphiAgent prompt files |

## Setup

```powershell
$env:DEEPSEEK_API_KEY = "<YOUR_API_KEY>"
$env:DACV_MODEL_ID     = "deepseek-flash"
$env:DACV_TEMPERATURE  = "0.1"
```
