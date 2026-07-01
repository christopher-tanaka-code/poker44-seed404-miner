# Poker44 Seed404 Miner Model Card

## Model

- Name: poker44-seed404-windows-local-stack
- Version: see artifact metadata in `models/poker44_seed404_local_model.joblib`
- Framework: scikit-learn local stacking model
- Artifact: `models/poker44_seed404_local_model.joblib`
- Metrics: `reports/train_metrics.json`

## Training data

This model was trained only on the public Poker44 benchmark API and released benchmark chunks.

No validator-only labels, hidden live labels, non-public Poker44 production labels, private evaluation labels, Discord-shared labels, or production-only data were used.

## Input/output contract

The miner receives `DetectionSynapse(chunks=...)`.

Each received chunk is treated as one scoring unit. The model returns exactly one risk score per received chunk.

Low score means human-like behavior. High score means bot-like behavior.

## Feature boundary

The model uses only miner-visible hand/action/chunk payload fields.

The model does not use `hand_id`, `chunkId`, `chunkHash`, `sourceDate`, release date, pagination order, or hidden labels as predictive features.

## Validation

Training, validation, and test splits are separated by benchmark release date to reduce release-specific overfitting.

See `reports/train_metrics.json` for the latest local validation and test metrics.

## Safety

Poker44 scoring can penalize false positives heavily, so this model is calibrated conservatively with a validation-selected threshold targeting controlled human false-positive rate.
