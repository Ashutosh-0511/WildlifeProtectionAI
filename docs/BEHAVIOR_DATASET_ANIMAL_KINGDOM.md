# Behaviour Dataset: Animal Kingdom

## Decision
Animal Kingdom is the primary external dataset for the first behaviour-modeling stage. The official repository reports 30K video sequences for fine-grained multi-label action recognition, 50 hours of behaviour-grounded video and 33K pose frames across 850 species and six major animal classes. The action-recognition annotations use a Charades-style CSV with clip_id, clip_number, frame_number, clip_path and action_labels.

## Project adaptation
The project baseline uses a smaller single-label ontology:
- RESTING
- NORMAL_MOVEMENT
- RUNNING
- CHASING
- AGGRESSIVE_ABNORMAL
- UNKNOWN

The adapter preserves source action labels and applies an explicit deterministic keyword-priority mapping. This is an engineering transfer baseline, not a claim that the source taxonomy is equivalent to the project's conservation taxonomy. The mapping must be inspected and refined before reporting research results.

## Lowest-effort execution path
1. Obtain the Animal Kingdom action-recognition dataset and annotations under the dataset's current access/licensing terms.
2. Point the project at the action-recognition annotation CSV and extracted video directory.
3. Run scripts/prepare_animal_kingdom.py to extract fixed 8-frame sequences and create deterministic train/val/test manifests grouped by clip_id.
4. Inspect data/processed/animal_kingdom/manifest_summary.json and verify that the mapped classes have enough samples.
5. Train the CPU ResNet18+LSTM baseline with scripts/train_behavior_lstm.py.
6. Evaluate the frozen test manifest with scripts/evaluate_behavior.py.
7. Only after the model passes evaluation, run scripts/run_pipeline.py with --behavior-checkpoint to connect behaviour to the existing detection → tracking → SpeciesNet → risk path.
8. Keep Animal Kingdom metrics separate from any Indian/target-domain evaluation.

## Reproducibility
Record dataset version/download date, annotation checksum, generated manifest checksum, mapping version, code commit, random seed, model checkpoint and evaluation metrics.

## Citation
Ng, Xun Long, et al. “Animal Kingdom: A Large and Diverse Dataset for Animal Behavior Understanding.” CVPR 2022, pp. 19023–19034.
