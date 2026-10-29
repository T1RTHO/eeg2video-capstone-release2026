# Preserved compact results

This directory contains small, selected evidence copied byte-for-byte from the verified final outputs, plus path-free provenance summaries. No generated videos, comparison images, ground-truth frames, checkpoints, predictor arrays, model caches, or raw data are included.

- `final_generation/` has the final run config, the RGB/temporal evaluation summary, all 200 per-clip metric rows, completion records with input/output hashes, and the evaluation environment record.
- `training_logs/` has the epoch-level CSV logs of the four final training runs (visual, semantic, dynamic, diffusion), copied unchanged; they are the source of `docs/media/model_training_summary.png`.
- `identification/` has aggregate results and per-clip pseudo-label agreement scores. Its provenance summary omits the original config's 200 machine-local input paths; the original config SHA-256 is recorded instead.
- `provenance/` retains diffusion run config/provenance and curated training summaries with checkpoint/config/summary hashes.
- `representative_examples.csv` lists the five predetermined sample IDs (test indices 0, 50, 100, 150 and 199), original local paths and hashes. The associated media is not copied.

The RGB summary contains a historical note saying classifier metrics were pending when that report was produced. The separate identification report is complete for all 200 clips; consult its own summary for those results.

The example comparison PNGs include reference imagery. Do not copy or publish any example media until dataset and output redistribution permissions are confirmed.
