# RAWFC per-sample predictions - highest-scoring run of the five R = 1 runs

This folder contains the per-sample prediction record of the **highest-scoring run
among the five** RAWFC R = 1 runs reported in the paper. It is the run whose
confusion matrix and error analysis are shown in the paper: accuracy 67.00 %,
macro-F1 67.22 %, half-class F1 63.16 % (n = 200).

## Files

| file | content |
| --- | --- |
| `<event_id>.json` | one record per RAWFC test claim (n = 200) |
| `rawfc_r1_best_of_five_predictions.jsonl` | the same 200 records, one JSON object per line |
| `metrics.json` | aggregate metrics recomputed from these records |

## Fields

| field | meaning |
| --- | --- |
| `event_id` | RAWFC test record identifier (identical to the identifier in the dataset release) |
| `gold_label` | gold three-class label (`true` / `half` / `false`) |
| `dim_verdicts` | dimension-level verdict and verifier confidence for `E` (event), `A` (attribution) and `Q` (quantity); verdict in {`T`, `F`, `U`, `N/A`}, confidence in {`high`, `medium`, `low`} |
| `framing_strength` | auxiliary framing signal in {`none`, `weak`, `strong`} |
| `framing_content` | the framing term that was matched, when one is present |
| `literal_mismatch` | whether a literal-mismatch signature fired for this claim |
| `rule_id` | deterministic rule that produced the final label (`R1`-`R10`, in the numbering used in the paper) |
| `final_label` | final DACV label used for evaluation |

## What is not included

The claim text and the verifier reasoning are **not** included in this release,
because they contain content that cannot be uploaded publicly for
sensitive-information reasons. The records therefore contain only model-derived
fields. Please obtain the RAWFC dataset from its original release and use these
records together with it.

## Reproducing the aggregate metrics

The records were produced by the deterministic Stage 3 derivation of the released
code from the Stage 2 outputs of the same run:

```
python stage3_runner.py --dataset RAWFC --task test
```

Aggregate results (see `metrics.json`): accuracy 0.6700, macro-P 0.6929,
macro-R 0.6702, macro-F1 0.6722, half-class F1 0.6316; confusion matrix (rows =
gold, columns = predicted): true 39/19/9, half 8/48/11, false 1/18/47; rule
activations R1-R10: 3, 24, 1, 67, 45, 15, 25, 7, 3, 10.

## Scope

These records correspond to the single highest-scoring run of the five RAWFC runs
at R = 1. They are not a five-run mean and they are not the matched comparison
runs; those results are reported separately in the paper. The records contain no
information beyond the fields listed above.

Released under the same terms as the accompanying code release.
