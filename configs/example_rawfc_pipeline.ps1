# Example: full RAWFC three-stage run (PowerShell)
# Copy this file outside the package if you want to edit it.
# Every path below is relative to the package root except $DataDir.

$ErrorActionPreference = "Stop"

# RAWFC root that contains train/, val/ and test/
# Set the DACV_RAWFC_DATA environment variable, or edit the default below.
$DataDir = if ($env:DACV_RAWFC_DATA) { $env:DACV_RAWFC_DATA } else { ".\datasets\RAWFC" }

$RunId   = "rawfc_run01"
$LogDir  = ".\logs\$RunId"
$Results = ".\results"

# ---------------------------------------------------------------- Stage 0
python .\code\rules\evidence_pool.py `
  --data_dir "$DataDir\test" `
  --output_dir "$Results\evidence_pool\RAWFC\test" `
  --dataset RAWFC

# Training split, used only for the AFEV-style matched baseline demonstrations
python .\code\rules\evidence_pool.py `
  --data_dir "$DataDir\train" `
  --output_dir "$Results\evidence_pool\RAWFC\train" `
  --dataset RAWFC

# ---------------------------------------------------------------- Stage 1
python .\code\agents\decomposer.py `
  --data_dir $DataDir `
  --dataset RAWFC --task test `
  --output_dir "$Results\decomposition" `
  --model_name deepseek-flash `
  --run_dir $LogDir --run_id $RunId

# ---------------------------------------------------------------- Stage 2
# Omit --use_search to reproduce the "Serper off" condition.
python .\code\stage2_runner.py `
  --dataset RAWFC --task test --rounds 1 `
  --decomp_dir "$Results\decomposition" `
  --pool_dir "$Results\evidence_pool" `
  --output_dir "$Results\stage2" `
  --model_name deepseek-flash `
  --run_dir $LogDir --run_id $RunId

# ---------------------------------------------------------------- Stage 3
python .\code\stage3_runner.py `
  --dataset RAWFC --task test `
  --stage2_dir "$Results\stage2" `
  --output_dir "$Results\stage3" `
  --run_dir $LogDir --run_id $RunId
