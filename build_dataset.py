#!/usr/bin/env python3
"""
Windows-local Poker44 benchmark dataset builder.

Usage:
  python build_dataset.py --download --out data/poker44_dataset.joblib

Or, if you already copied training_benchmark.txt from VPS:
  python build_dataset.py --benchmark training_benchmark.txt --out data/poker44_dataset.joblib

Outputs:
  data/poker44_dataset.joblib

Dependencies:
  pip install numpy pandas scikit-learn joblib requests
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests


API_BASE = "https://api.poker44.net/api/v1/benchmark"
API_CHUNK_LIMIT = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="")
    parser.add_argument("--out", type=str, default="data/poker44_dataset.joblib")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--release-limit", type=int, default=40)
    parser.add_argument("--chunk-limit", type=int, default=48)
    parser.add_argument("--cache-json", type=str, default="data/training_benchmark.json")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def safe_len(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple, dict)) else 0


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def q(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p)
    return float(ent)


def unwrap_api(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def http_get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=45)
    if response.status_code >= 400:
        print("HTTP error:", response.status_code, response.text[:500])
    response.raise_for_status()
    return response.json()


def download_benchmark(cache_path: Path, release_limit: int, chunk_limit: int) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_limit = min(int(chunk_limit), API_CHUNK_LIMIT)

    releases_payload = unwrap_api(
        http_get_json(f"{API_BASE}/releases", {"limit": int(release_limit)})
    )

    if isinstance(releases_payload, dict):
        releases = releases_payload.get("releases") or releases_payload.get("items") or []
    elif isinstance(releases_payload, list):
        releases = releases_payload
    else:
        releases = []

    releases = [r for r in releases if isinstance(r, dict) and r.get("sourceDate")]
    releases = sorted(releases, key=lambda r: r["sourceDate"])

    print(f"Found releases: {len(releases)}")

    all_records: list[dict[str, Any]] = []

    for release in releases:
        source_date = release["sourceDate"]
        print(f"Downloading sourceDate={source_date}")

        cursor = None
        seen_cursors: set[str] = set()
        date_records: list[dict[str, Any]] = []

        while True:
            params: dict[str, Any] = {
                "sourceDate": source_date,
                "limit": chunk_limit,
            }
            if cursor:
                params["cursor"] = cursor

            raw = http_get_json(f"{API_BASE}/chunks", params=params)
            data = unwrap_api(raw)

            if isinstance(data, dict):
                chunks = data.get("chunks") or data.get("items") or []

                next_cursor = (
                    data.get("nextCursor")
                    or data.get("next_cursor")
                    or data.get("cursor")
                    or data.get("next")
                )

                for key in ("pagination", "metadata", "meta"):
                    obj = data.get(key)
                    if isinstance(obj, dict):
                        next_cursor = (
                            next_cursor
                            or obj.get("nextCursor")
                            or obj.get("next_cursor")
                            or obj.get("cursor")
                            or obj.get("next")
                        )

            elif isinstance(data, list):
                chunks = data
                next_cursor = None
            else:
                chunks = []
                next_cursor = None

            date_records.extend([c for c in chunks if isinstance(c, dict)])

            if not next_cursor:
                break

            next_cursor = str(next_cursor)
            if next_cursor in seen_cursors:
                break

            seen_cursors.add(next_cursor)
            cursor = next_cursor
            time.sleep(0.15)

        print(f"  records={len(date_records)}")
        all_records.extend(date_records)

    if not all_records:
        raise RuntimeError("No benchmark records downloaded.")

    payload = {"data": {"sourceDate": "multi-release", "chunks": all_records}}
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    print(f"Saved raw benchmark: {cache_path}")
    print(f"Total benchmark records: {len(all_records)}")

    return cache_path


def load_json(path: Path) -> Any:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_actions(hand: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    raw_actions = hand.get("actions")
    if isinstance(raw_actions, list):
        actions.extend([a for a in raw_actions if isinstance(a, dict)])

    streets = hand.get("streets")
    if isinstance(streets, dict):
        for street_value in streets.values():
            if isinstance(street_value, dict):
                street_actions = street_value.get("actions")
                if isinstance(street_actions, list):
                    actions.extend([a for a in street_actions if isinstance(a, dict)])
            elif isinstance(street_value, list):
                actions.extend([a for a in street_value if isinstance(a, dict)])

    elif isinstance(streets, list):
        for street in streets:
            if isinstance(street, dict):
                street_actions = street.get("actions")
                if isinstance(street_actions, list):
                    actions.extend([a for a in street_actions if isinstance(a, dict)])

    return actions


def action_type(action: dict[str, Any]) -> str:
    for key in ("action_type", "type", "action", "name"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return value.lower().strip()
    return "unknown"


def action_street(action: dict[str, Any]) -> str:
    value = action.get("street")
    if isinstance(value, str) and value:
        return value.lower().strip()
    return "unknown"


def actor(action: dict[str, Any]) -> str:
    for key in ("actor_seat", "seat", "player_id", "actor", "player"):
        value = action.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def player_count(hand: dict[str, Any]) -> int:
    players = hand.get("players")
    if isinstance(players, list):
        return len(players)
    if isinstance(players, dict):
        return len(players)
    return 0


def outcome_amounts(hand: dict[str, Any]) -> list[float]:
    values: list[float] = []
    outcome = hand.get("outcome")

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_l = str(key).lower()
                if any(tok in key_l for tok in ("win", "profit", "net", "amount", "bb")):
                    if isinstance(value, (int, float)):
                        values.append(safe_float(value))
                collect(value)
        elif isinstance(obj, list):
            for item in obj:
                collect(item)

    collect(outcome)
    return values


def hand_features(hand: dict[str, Any]) -> dict[str, float]:
    actions = get_actions(hand)
    action_types = [action_type(a) for a in actions]
    streets = [action_street(a) for a in actions]
    actors = [actor(a) for a in actions]

    type_counts = Counter(action_types)
    street_counts = Counter(streets)
    actor_counts = Counter(actors)

    amount_values: list[float] = []
    pot_before_values: list[float] = []
    pot_after_values: list[float] = []
    normalized_values: list[float] = []

    for a in actions:
        for key in ("amount", "raise_to", "call_to", "bet", "size"):
            if key in a:
                amount_values.append(abs(safe_float(a.get(key))))

        if "normalized_amount_bb" in a:
            normalized_values.append(abs(safe_float(a.get("normalized_amount_bb"))))

        if "pot_before" in a:
            pot_before_values.append(abs(safe_float(a.get("pot_before"))))

        if "pot_after" in a:
            pot_after_values.append(abs(safe_float(a.get("pot_after"))))

    n_actions = len(actions)
    n_players = player_count(hand)
    n_streets = len([s for s in set(streets) if s != "unknown"])

    aggressive = (
        type_counts["bet"]
        + type_counts["raise"]
        + type_counts["raises"]
        + type_counts["allin"]
        + type_counts["all-in"]
    )
    passive = type_counts["call"] + type_counts["calls"] + type_counts["check"] + type_counts["checks"]
    folds = type_counts["fold"] + type_counts["folds"]

    showdown = 0.0
    outcome = hand.get("outcome")
    if isinstance(outcome, dict) and outcome:
        showdown = 1.0

    out_amounts = outcome_amounts(hand)

    f: dict[str, float] = {
        "hand_actions": float(n_actions),
        "hand_players": float(n_players),
        "hand_streets": float(n_streets),
        "hand_unique_actors": float(len([a for a in set(actors) if a != "unknown"])),
        "hand_showdown": showdown,
        "hand_aggressive_actions": float(aggressive),
        "hand_passive_actions": float(passive),
        "hand_folds": float(folds),
        "hand_action_entropy": entropy_from_counts(type_counts),
        "hand_actor_entropy": entropy_from_counts(actor_counts),
        "hand_street_entropy": entropy_from_counts(street_counts),
        "hand_amount_mean": mean(amount_values),
        "hand_amount_max": max(amount_values) if amount_values else 0.0,
        "hand_amount_std": stdev(amount_values),
        "hand_norm_amount_mean": mean(normalized_values),
        "hand_norm_amount_max": max(normalized_values) if normalized_values else 0.0,
        "hand_pot_before_mean": mean(pot_before_values),
        "hand_pot_after_mean": mean(pot_after_values),
        "hand_outcome_abs_sum": float(sum(abs(x) for x in out_amounts)),
        "hand_outcome_max_abs": max([abs(x) for x in out_amounts], default=0.0),
    }

    action_den = max(float(n_actions), 1.0)
    for name in [
        "fold",
        "call",
        "check",
        "bet",
        "raise",
        "allin",
        "all-in",
        "post",
        "blind",
        "unknown",
    ]:
        f[f"hand_action_ratio_{name.replace('-', '_')}"] = float(type_counts[name] / action_den)

    f["hand_aggression_ratio"] = float(aggressive / max(passive + folds, 1))
    f["hand_actions_per_player"] = float(n_actions / max(n_players, 1))

    return f


def chunk_features(chunk: list[dict[str, Any]]) -> dict[str, float]:
    hand_fs = [hand_features(h) for h in chunk if isinstance(h, dict)]
    out: dict[str, float] = {}

    if not hand_fs:
        return {"hand_count": 0.0}

    keys = sorted(set().union(*[set(f.keys()) for f in hand_fs]))

    out["hand_count"] = float(len(hand_fs))

    for key in keys:
        vals = [float(f.get(key, 0.0)) for f in hand_fs]
        out[f"{key}_mean"] = mean(vals)
        out[f"{key}_std"] = stdev(vals)
        out[f"{key}_min"] = min(vals) if vals else 0.0
        out[f"{key}_max"] = max(vals) if vals else 0.0
        out[f"{key}_q25"] = q(vals, 0.25)
        out[f"{key}_q50"] = q(vals, 0.50)
        out[f"{key}_q75"] = q(vals, 0.75)
        out[f"{key}_sum"] = float(sum(vals))

    # Chunk-level pattern features.
    action_type_total = Counter()
    street_total = Counter()
    actor_total = Counter()
    action_counts_per_hand: list[float] = []
    player_counts_per_hand: list[float] = []

    for hand in chunk:
        actions = get_actions(hand)
        action_counts_per_hand.append(float(len(actions)))
        player_counts_per_hand.append(float(player_count(hand)))

        for action in actions:
            action_type_total[action_type(action)] += 1
            street_total[action_street(action)] += 1
            actor_total[actor(action)] += 1

    total_actions = sum(action_type_total.values())
    den = max(float(total_actions), 1.0)

    out["chunk_total_actions"] = float(total_actions)
    out["chunk_action_entropy"] = entropy_from_counts(action_type_total)
    out["chunk_street_entropy"] = entropy_from_counts(street_total)
    out["chunk_actor_entropy"] = entropy_from_counts(actor_total)
    out["chunk_actions_per_hand_mean"] = mean(action_counts_per_hand)
    out["chunk_actions_per_hand_std"] = stdev(action_counts_per_hand)
    out["chunk_players_per_hand_mean"] = mean(player_counts_per_hand)
    out["chunk_players_per_hand_std"] = stdev(player_counts_per_hand)

    for name in [
        "fold",
        "call",
        "check",
        "bet",
        "raise",
        "allin",
        "all-in",
        "post",
        "blind",
        "unknown",
    ]:
        out[f"chunk_action_ratio_{name.replace('-', '_')}"] = float(action_type_total[name] / den)

    aggressive = (
        action_type_total["bet"]
        + action_type_total["raise"]
        + action_type_total["raises"]
        + action_type_total["allin"]
        + action_type_total["all-in"]
    )
    passive = (
        action_type_total["call"]
        + action_type_total["calls"]
        + action_type_total["check"]
        + action_type_total["checks"]
    )
    folds = action_type_total["fold"] + action_type_total["folds"]

    out["chunk_aggression_ratio"] = float(aggressive / max(passive + folds, 1))
    out["chunk_fold_ratio"] = float(folds / den)
    out["chunk_call_check_ratio"] = float(passive / den)

    clean: dict[str, float] = {}
    for key, value in out.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            x = float(value)
            if math.isfinite(x):
                clean[key] = x

    return clean


def parse_label(value: Any) -> int:
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"bot", "synthetic_bot", "1", "true"}:
            return 1
        if s in {"human", "0", "false"}:
            return 0
    return int(value)


def extract_groups(payload: Any) -> list[dict[str, Any]]:
    root = unwrap_api(payload)

    if isinstance(root, dict) and isinstance(root.get("chunks"), list):
        return [g for g in root["chunks"] if isinstance(g, dict)]

    if isinstance(root, list):
        return [g for g in root if isinstance(g, dict)]

    raise RuntimeError("Could not find benchmark chunks list in JSON.")


def build_dataset(benchmark_path: Path, output_path: Path) -> None:
    payload = load_json(benchmark_path)
    groups = extract_groups(payload)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    meta_rows: list[dict[str, Any]] = []

    for group_idx, group in enumerate(groups):
        chunks = group.get("chunks") or []
        labels_raw = group.get("groundTruth")
        if labels_raw is None:
            labels_raw = group.get("groundTruthLabels") or []

        if not isinstance(chunks, list) or not isinstance(labels_raw, list):
            continue

        if len(chunks) != len(labels_raw):
            print(
                f"Skipping group {group_idx}: chunks={len(chunks)} labels={len(labels_raw)} mismatch"
            )
            continue

        source_date = str(group.get("sourceDate") or "")
        release_version = str(group.get("releaseVersion") or "")
        schema_version = str(group.get("schemaVersion") or "")
        split = str(group.get("split") or "")
        chunk_id = str(group.get("chunkId") or f"group_{group_idx}")
        chunk_hash = str(group.get("chunkHash") or "")

        for item_idx, (chunk, label_raw) in enumerate(zip(chunks, labels_raw)):
            if not isinstance(chunk, list):
                continue

            feature_row = chunk_features(chunk)
            if not feature_row:
                continue

            rows.append(feature_row)
            labels.append(parse_label(label_raw))
            meta_rows.append(
                {
                    "source_date": source_date,
                    "release_version": release_version,
                    "schema_version": schema_version,
                    "split": split,
                    "chunk_id": chunk_id,
                    "chunk_hash": chunk_hash,
                    "item_index": item_idx,
                }
            )

    if not rows:
        raise RuntimeError("No rows extracted from benchmark.")

    X = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    y = np.asarray(labels, dtype=np.int8)
    meta = pd.DataFrame(meta_rows)

    feature_names = sorted(X.columns.tolist())
    X = X.reindex(columns=feature_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "X": X,
            "y": y,
            "meta": meta,
            "feature_names": feature_names,
            "source_file": str(benchmark_path),
        },
        output_path,
        compress=3,
    )

    print("Saved dataset:", output_path)
    print("Rows:", len(y))
    print("Features:", len(feature_names))
    print("Bots:", int(y.sum()))
    print("Humans:", int((y == 0).sum()))
    print("Source dates:", sorted(meta["source_date"].unique().tolist()))


def main() -> None:
    args = parse_args()

    if args.download:
        benchmark_path = download_benchmark(
            Path(args.cache_json),
            release_limit=args.release_limit,
            chunk_limit=args.chunk_limit,
        )
    else:
        if not args.benchmark:
            raise SystemExit("Pass --download or --benchmark path/to/training_benchmark.txt")
        benchmark_path = Path(args.benchmark)

    build_dataset(benchmark_path, Path(args.out))


if __name__ == "__main__":
    main()
