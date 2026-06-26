# Poker44 Seed404 Miner Model Card

## Model

- Name: poker44-seed404-windows-local-stack
- Version: seed404-2026-06-25
- Framework: scikit-learn local stacking model
- Artifact: models/poker44_seed404_local_model.joblib

## Training data

This model was trained only on the public Poker44 benchmark API / released benchmark chunks.

No validator-only labels, hidden live labels, non-public Poker44 production labels, or private evaluation labels were used.

## Input/output contract

The miner receives `DetectionSynapse(chunks=...)`.

Each `chunk` is treated as one scoring unit. The model returns exactly one risk score per chunk, where low scores mean human-like behavior and high scores mean bot-like behavior.

## Seed selection

Seed 404 was selected by multi-seed sweep across multiple latest-date holdout windows, prioritizing safe FPR, stable reward, and recall above the previous baseline.

## Safety

Poker44 scoring heavily penalizes false positives. This model is calibrated to keep human FPR conservative rather than blindly maximizing bot recall.
