from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/poker44_dataset.joblib")
    p.add_argument("--out", default="models/poker44_seed404_local_model.joblib")
    p.add_argument("--metrics-out", default="reports/train_metrics.json")
    p.add_argument("--seed", type=int, default=404)
    p.add_argument("--target-fpr", type=float, default=0.10)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_auc(y, p):
    try:
        if len(set(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def evaluate(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )

    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    fpr = fp / max(fp + tn, 1)

    return {
        "rows": int(len(y)),
        "bots": int(y.sum()),
        "humans": int((y == 0).sum()),
        "roc_auc": safe_auc(y, p),
        "average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))) if len(set(y)) > 1 else None,
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": float(threshold),
    }


def choose_threshold_for_fpr(y, p, target_fpr=0.10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    best = 0.5
    best_recall = -1.0

    for t in candidates:
        pred = (p >= t).astype(int)
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        tp = int(((y == 1) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())

        fpr = fp / max(fp + tn, 1)
        recall = tp / max(tp + fn, 1)

        if fpr <= target_fpr and recall > best_recall:
            best = float(t)
            best_recall = float(recall)

    return best


def date_split(meta: pd.DataFrame):
    dates = sorted([d for d in meta["source_date"].astype(str).unique() if d])
    if len(dates) < 3:
        idx = np.arange(len(meta))
        return idx[: int(0.7 * len(idx))], idx[int(0.7 * len(idx)) : int(0.85 * len(idx))], idx[int(0.85 * len(idx)) :]

    n = len(dates)
    train_dates = set(dates[: max(1, int(n * 0.70))])
    val_dates = set(dates[max(1, int(n * 0.70)) : max(2, int(n * 0.85))])
    test_dates = set(dates[max(2, int(n * 0.85)) :])

    if not val_dates:
        val_dates = {dates[-2]}
        train_dates.discard(dates[-2])
    if not test_dates:
        test_dates = {dates[-1]}
        train_dates.discard(dates[-1])

    source_dates = meta["source_date"].astype(str)

    train_idx = np.where(source_dates.isin(train_dates))[0]
    val_idx = np.where(source_dates.isin(val_dates))[0]
    test_idx = np.where(source_dates.isin(test_dates))[0]

    return train_idx, val_idx, test_idx


def make_base_models(seed: int):
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed + 1,
            n_jobs=-1,
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=seed + 2,
        ),
    }


def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    metrics_path = Path(args.metrics_out)

    data = joblib.load(dataset_path)

    X: pd.DataFrame = data["X"].copy()
    y = np.asarray(data["y"]).astype(int)
    meta: pd.DataFrame = data["meta"].copy()
    feature_names = list(data["feature_names"])

    X = X.reindex(columns=feature_names).replace([np.inf, -np.inf], np.nan)

    train_idx, val_idx, test_idx = date_split(meta)

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    base_defs = make_base_models(args.seed)

    groups = meta.iloc[train_idx]["source_date"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))

    if len(unique_groups) >= 3:
        n_splits = min(5, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X_train_imp, y_train, groups)
    else:
        n_splits = min(5, int(np.bincount(y_train).min())) if len(set(y_train)) > 1 else 2
        n_splits = max(2, n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        splits = splitter.split(X_train_imp, y_train)

    splits = list(splits)

    oof_base = np.zeros((len(y_train), len(base_defs)), dtype=float)
    fitted_base_models = {}

    for model_col, (name, model_def) in enumerate(base_defs.items()):
        for fold, (tr, va) in enumerate(splits):
            model = clone(model_def)
            model.fit(X_train_imp[tr], y_train[tr])

            if hasattr(model, "predict_proba"):
                oof_base[va, model_col] = model.predict_proba(X_train_imp[va])[:, 1]
            else:
                oof_base[va, model_col] = model.predict(X_train_imp[va])

        final_model = clone(model_def)
        final_model.fit(X_train_imp, y_train)
        fitted_base_models[name] = final_model

    meta_model = LogisticRegression(
        C=0.75,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=args.seed,
    )
    meta_model.fit(oof_base, y_train)

    def base_matrix(X_imp):
        cols = []
        for model in fitted_base_models.values():
            if hasattr(model, "predict_proba"):
                cols.append(model.predict_proba(X_imp)[:, 1])
            else:
                cols.append(model.predict(X_imp))
        return np.vstack(cols).T

    val_base = base_matrix(X_val_imp)
    test_base = base_matrix(X_test_imp)

    raw_val = meta_model.predict_proba(val_base)[:, 1]
    raw_test = meta_model.predict_proba(test_base)[:, 1]

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=args.seed,
    )
    calibrator.fit(raw_val.reshape(-1, 1), y_val)

    val_scores = calibrator.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    test_scores = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    threshold = choose_threshold_for_fpr(y_val, val_scores, target_fpr=args.target_fpr)

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset": str(dataset_path),
        "model_out": str(out_path),
        "rows_total": int(len(y)),
        "features": int(len(feature_names)),
        "source_dates": sorted(meta["source_date"].astype(str).unique().tolist()),
        "splits": {
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
            "train_dates": sorted(meta.iloc[train_idx]["source_date"].astype(str).unique().tolist()),
            "val_dates": sorted(meta.iloc[val_idx]["source_date"].astype(str).unique().tolist()),
            "test_dates": sorted(meta.iloc[test_idx]["source_date"].astype(str).unique().tolist()),
        },
        "val": evaluate(y_val, val_scores, threshold=threshold),
        "test": evaluate(y_test, test_scores, threshold=threshold),
    }

    artifact = {
        "imputer": imputer,
        "base_models": fitted_base_models,
        "meta_model": meta_model,
        "calibrator": calibrator,
        "feature_names": feature_names,
        "remap": {
            "threshold": float(threshold),
            "temperature": 0.25,
        },
        "metadata": {
            "model_name": "poker44-seed404-windows-local-stack",
            "model_version": "seed404-refresh-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
            "framework": "scikit-learn local stacking model",
            "training_data_statement": "Public Poker44 benchmark API only; miner-visible chunk features only.",
            "private_data_attestation": "No validator-only labels, hidden live labels, non-public Poker44 production labels, or private evaluation labels were used.",
            "data_attestation": "Uses miner-visible hand/action/chunk payload fields. Does not use hand_id, chunkId, chunkHash, sourceDate, release date, pagination order, or hidden labels as predictive features.",
            "target_fpr": float(args.target_fpr),
            "validation_threshold": float(threshold),
            "metrics": metrics,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path, compress=3)

    metrics["artifact_sha256"] = sha256_file(out_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved model:", out_path)
    print("Saved metrics:", metrics_path)
    print("Artifact SHA256:", metrics["artifact_sha256"])
    print("Validation:", json.dumps(metrics["val"], indent=2))
    print("Test:", json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
