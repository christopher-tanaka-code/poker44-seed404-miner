from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.synapse import DetectionSynapse
from poker44_local_runtime import LocalWindowsStackPredictor


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
        ).strip()
    except Exception:
        return ""


def _ensure_ns(obj, name):
    value = getattr(obj, name, None)
    if value is None:
        value = SimpleNamespace()
        setattr(obj, name, value)
    return value


def _default(ns, name, value):
    if getattr(ns, name, None) in (None, ""):
        setattr(ns, name, value)


def build_runtime_config():
    """
    Compatibility layer for newer Bittensor config objects where nested
    namespaces like config.neuron/config.blacklist may be None.
    """
    cfg = Miner.config()

    wallet = _ensure_ns(cfg, "wallet")
    subtensor = _ensure_ns(cfg, "subtensor")
    axon = _ensure_ns(cfg, "axon")
    neuron = _ensure_ns(cfg, "neuron")
    blacklist = _ensure_ns(cfg, "blacklist")
    logging_cfg = _ensure_ns(cfg, "logging")

    # Force runtime values from env. Miner.config() may already contain
    # wallet.name="default" / wallet.hotkey="default", so default-only assignment
    # is not enough here.
    wallet.name = os.getenv("POKER44_WALLET_NAME", "chris-11")
    wallet.hotkey = os.getenv("POKER44_HOTKEY_NAME", "default")
    wallet.path = os.getenv("POKER44_WALLET_PATH", os.path.expanduser("~/.bittensor/wallets"))

    subtensor.network = os.getenv("POKER44_SUBTENSOR_NETWORK", "finney")
    axon.port = int(os.getenv("POKER44_AXON_PORT", "8091"))

    cfg.netuid = int(os.getenv("POKER44_NETUID", "126"))

    _default(neuron, "name", os.getenv("POKER44_NEURON_NAME", "poker44_topminer"))
    _default(neuron, "device", os.getenv("POKER44_DEVICE", "cpu"))
    _default(neuron, "epoch_length", int(os.getenv("POKER44_EPOCH_LENGTH", "50")))
    _default(neuron, "disable_set_weights", False)
    _default(neuron, "wait_for_inclusion", True)
    _default(neuron, "wait_for_finalization", True)
    _default(neuron, "moving_average_alpha", 0.05)
    _default(neuron, "num_concurrent_forwards", 1)
    _default(neuron, "timeout", float(os.getenv("POKER44_NEURON_TIMEOUT", "60")))
    _default(neuron, "axon_off", False)

    _default(blacklist, "force_validator_permit", True)
    _default(blacklist, "allow_non_registered", False)
    _default(blacklist, "allowed_validator_hotkeys", [])

    _default(logging_cfg, "logging_dir", os.path.expanduser("~/.bittensor/miners"))

    return cfg


class Miner(BaseMinerNeuron):
    """
    Poker44 seed404 miner.

    Contract:
    - receives DetectionSynapse(chunks=...)
    - returns exactly one risk_score per chunk
    - each score is clamped to [0, 1]
    """

    def __init__(self, config=None):
        super().__init__(config=config)

        model_path = Path(
            os.getenv(
                "POKER44_LOCAL_STACK_PATH",
                str(REPO_ROOT / "models" / "poker44_seed404_local_model.joblib"),
            )
        ).resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        self.backend = os.getenv("POKER44_LOCAL_BACKEND_NAME", "windows-local-stack-seed404")
        self.predictor = LocalWindowsStackPredictor(model_path)

        repo_url = os.getenv(
            "POKER44_MODEL_REPO_URL",
            "https://github.com/christopher-tanaka-code/poker44-seed404-miner",
        )
        repo_commit = git_commit(REPO_ROOT)

        implementation_files = [
            REPO_ROOT / "neurons" / "seed404_miner.py",
            REPO_ROOT / "poker44_local_runtime.py",
            REPO_ROOT / "build_dataset.py",
        ]
        train_model_path = REPO_ROOT / "train_model.py"
        if train_model_path.exists():
            implementation_files.append(train_model_path)

        artifact_sha256 = sha256_file(model_path)
        implementation_sha256 = sha256_files(implementation_files)

        self.model_manifest = {
            "schema_version": "1.0",
            "open_source": True,
            "repo_url": repo_url,
            "repo_commit": repo_commit,
            "model_name": "poker44-seed404-windows-local-stack",
            "model_version": os.getenv("POKER44_MODEL_VERSION", "seed404-2026-06-25"),
            "framework": "sklearn-stacking-local",
            "license": "MIT",
            "training_data_statement": (
                "Trained only on the public Poker44 benchmark API / released benchmark "
                "chunks. No validator-only labels or private production labels used."
            ),
            "training_data_sources": [
                "https://api.poker44.net/api/v1/benchmark/releases",
                "https://api.poker44.net/api/v1/benchmark/chunks",
            ],
            "private_data_attestation": (
                "No private validator labels, non-public Poker44 production labels, "
                "or hidden evaluation labels were used."
            ),
            "data_attestation": "Public benchmark data only.",
            "artifact_url": (
                f"{repo_url}/blob/main/models/poker44_seed404_local_model.joblib"
            ),
            "artifact_sha256": artifact_sha256,
            "model_card_url": f"{repo_url}/blob/main/MODEL_CARD.md",
            "inference_mode": "local-joblib",
            "implementation_files": [
                str(path.relative_to(REPO_ROOT)) for path in implementation_files
            ],
            "implementation_sha256": implementation_sha256,
            "notes": (
                "Seed 404 selected by multi-seed sweep across multiple latest-date "
                "holdout windows. Runtime serves one score per DetectionSynapse chunk."
            ),
        }

        bt.logging.info(f"Seed404 Poker44 Miner started with backend={self.backend}")
        bt.logging.info(f"Model artifact={model_path}")
        bt.logging.info(f"Model artifact sha256={artifact_sha256}")
        bt.logging.info(f"Implementation sha256={implementation_sha256}")
        bt.logging.info(f"Manifest repo={repo_url}")
        bt.logging.info(f"Manifest commit={repo_commit}")
        bt.logging.info(f"Manifest model={self.model_manifest['model_name']}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        caller = None
        try:
            caller = synapse.dendrite.hotkey if synapse.dendrite else None
        except Exception:
            caller = None

        bt.logging.info(
            f"Validator query received | caller={caller} "
            f"chunks={len(chunks)} backend={self.backend}"
        )

        try:
            scores = self.predictor.predict_chunk_scores(chunks)
        except Exception as exc:
            bt.logging.error(f"Predictor failure: {exc}")
            scores = [0.5 for _ in chunks]

        if len(scores) != len(chunks):
            bt.logging.error(
                f"Score length mismatch: scores={len(scores)} chunks={len(chunks)}. "
                "Using neutral fallback."
            )
            scores = [0.5 for _ in chunks]

        scores = [max(0.0, min(1.0, float(s))) for s in scores]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        if scores:
            bt.logging.info(
                f"Scored {len(chunks)} chunks | "
                f"min={min(scores):.6f} max={max(scores):.6f} "
                f"mean={sum(scores) / len(scores):.6f} "
                f"backend={self.backend}"
            )
        else:
            bt.logging.info("Scored 0 chunks.")

        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    cfg = build_runtime_config()
    with Miner(config=cfg) as miner:
        bt.logging.info("Seed404 miner running.")
        while True:
            bt.logging.info(
                f"heartbeat backend={getattr(miner, 'backend', None)} "
                f"uid={getattr(miner, 'uid', None)} "
                f"is_running={getattr(miner, 'is_running', None)} "
                f"thread_alive={getattr(getattr(miner, 'thread', None), 'is_alive', lambda: None)()}"
            )
            time.sleep(60)
