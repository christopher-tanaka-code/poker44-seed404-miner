import argparse
import os
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.validator.synapse import DetectionSynapse
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44_local_runtime import LocalWindowsStackPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--netuid", type=int, default=int(os.getenv("NETUID", "126")))

    parser.add_argument("--wallet.name", "--wallet-name", dest="wallet_name", default=os.getenv("WALLET_NAME", "chris-11"))
    parser.add_argument("--wallet.hotkey", "--wallet-hotkey", "--hotkey", dest="wallet_hotkey", default=os.getenv("HOTKEY", "default"))
    parser.add_argument("--wallet.path", "--wallet-path", dest="wallet_path", default=os.getenv("WALLET_PATH", "/root/.bittensor/wallets"))

    parser.add_argument("--subtensor.chain_endpoint", "--chain-endpoint", dest="chain_endpoint", default=os.getenv("CHAIN_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443"))
    parser.add_argument("--subtensor.network", "--network", dest="network", default=os.getenv("NETWORK", "finney"))

    parser.add_argument("--axon.ip", "--axon-ip", dest="axon_ip", default=os.getenv("AXON_IP", "0.0.0.0"))
    parser.add_argument("--axon.port", "--axon-port", dest="axon_port", type=int, default=int(os.getenv("AXON_PORT", "8091")))
    parser.add_argument("--axon.external_ip", "--axon-external-ip", dest="axon_external_ip", default=os.getenv("AXON_EXTERNAL_IP", "138.201.140.119"))
    parser.add_argument("--axon.external_port", "--axon-external-port", dest="axon_external_port", type=int, default=int(os.getenv("AXON_EXTERNAL_PORT", os.getenv("AXON_PORT", "8091"))))

    parser.add_argument("--logging.debug", action="store_true", dest="logging_debug")
    parser.add_argument(
        "--blacklist.allowed_validator_hotkeys",
        dest="blacklist_allowed_validator_hotkeys",
        nargs="*",
        default=[],
    )

    return parser


class Miner:
    def __init__(self):
        self.args = build_parser().parse_args()

        self.repo_root = Path(__file__).resolve().parents[1]
        self.model_path = self.repo_root / "models" / "poker44_seed404_local_model.joblib"

        expected_hotkey_path = Path(self.args.wallet_path) / self.args.wallet_name / "hotkeys" / self.args.wallet_hotkey
        print(f"Wallet name: {self.args.wallet_name}")
        print(f"Wallet hotkey: {self.args.wallet_hotkey}")
        print(f"Expected hotkey path: {expected_hotkey_path}")

        if not expected_hotkey_path.exists():
            raise FileNotFoundError(
                f"Hotkey file not found: {expected_hotkey_path}. "
                f"Run: btcli wallet list"
            )

        print(f"Loading Poker44 model from {self.model_path}")
        self.model = LocalWindowsStackPredictor(self.model_path)

        model_meta = dict(getattr(self.model, "metadata", {}) or {})

        self.model_manifest = build_local_model_manifest(
            repo_root=self.repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                self.repo_root / "poker44_local_runtime.py",
                self.repo_root / "build_dataset.py",
                self.model_path,
            ],
            defaults={
                "schema_version": "1",
                "open_source": True,
                "repo_url": "https://github.com/christopher-tanaka-code/poker44-seed404-miner",
                "model_name": model_meta.get("model_name", "poker44-seed404-supervised"),
                "model_version": model_meta.get("model_version", "seed404-refresh"),
                "framework": model_meta.get("framework", "scikit-learn local stack"),
                "license": "MIT",
                "inference_mode": "remote",
                "training_data_statement": model_meta.get(
                    "training_data_statement",
                    "Trained only on public Poker44 benchmark releases and miner-visible chunk fields.",
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": model_meta.get(
                    "private_data_attestation",
                    "No validator-only labels, hidden live labels, non-public Poker44 production labels, or private evaluation labels were used.",
                ),
                "data_attestation": (
                    "Uses only miner-visible hand/action/chunk payload fields. "
                    "Does not use chunkId, chunkHash, sourceDate, pagination order, or hidden labels as predictive features."
                ),
                "artifact_url": "models/poker44_seed404_local_model.joblib",
                "model_card_url": "MODEL_CARD.md",
                "notes": "Minimal axon runtime with explicit wallet loading. Bypasses metagraph sync.",
            },
        )

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

        print(f"Manifest status: {self.manifest_compliance.get('status')}")
        print(f"Manifest digest: {self.manifest_digest}")

        print("Creating explicit Bittensor wallet...")
        self.wallet = bt.Wallet(
            name=self.args.wallet_name,
            hotkey=self.args.wallet_hotkey,
            path=self.args.wallet_path,
        )

        print("Creating subtensor...")
        # Bittensor SDK v10 Subtensor accepts network/config/log_verbose.
        # It does not accept chain_endpoint directly.
        self.subtensor = bt.Subtensor(network=self.args.network)

        print("Creating axon...")
        axon_kwargs = {
            "wallet": self.wallet,
            "ip": self.args.axon_ip,
            "port": self.args.axon_port,
        }

        if self.args.axon_external_ip:
            axon_kwargs["external_ip"] = self.args.axon_external_ip

        if self.args.axon_external_port:
            axon_kwargs["external_port"] = self.args.axon_external_port

        self.axon = bt.Axon(**axon_kwargs)

        self.allowed_validator_hotkeys = {
            str(h).strip()
            for h in self.args.blacklist_allowed_validator_hotkeys
            if str(h).strip()
        }

        self.axon.attach(
            forward_fn=self.forward,
            blacklist_fn=self.blacklist,
            priority_fn=self.priority,
        )

        print(
            f"Axon ready on {self.args.axon_ip}:{self.args.axon_port}, "
            f"external {self.args.axon_external_ip}:{self.args.axon_external_port}"
        )

    @staticmethod
    def _clamp01(value: float) -> float:
        try:
            value = float(value)
        except Exception:
            return 0.5
        if value != value:
            return 0.5
        return max(0.0, min(1.0, value))

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        try:
            scores = self.model.predict_chunk_scores(chunks)
        except Exception as exc:
            print(f"Model inference failed: {exc}")
            scores = [0.5 for _ in chunks]

        if len(scores) != len(chunks):
            print(f"Bad score length: scores={len(scores)} chunks={len(chunks)}")
            scores = [0.5 for _ in chunks]

        scores = [round(self._clamp01(score), 6) for score in scores]

        synapse.risk_scores = scores
        synapse.predictions = [score >= 0.5 for score in scores]
        synapse.model_manifest = dict(self.model_manifest)

        print(
            f"Scored {len(chunks)} chunks. "
            f"min={min(scores) if scores else None} "
            f"max={max(scores) if scores else None} "
            f"positives={sum(s >= 0.5 for s in scores)}"
        )

        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        hotkey = None
        if synapse.dendrite is not None:
            hotkey = synapse.dendrite.hotkey

        if self.allowed_validator_hotkeys:
            if hotkey in self.allowed_validator_hotkeys:
                return False, "Allowed validator hotkey"
            return True, "Not in validator allowlist"

        return False, "Allowed because metagraph RPC is unavailable"

    async def priority(self, synapse: DetectionSynapse) -> float:
        return 1.0

    def run(self):
        print(f"Serving axon on netuid={self.args.netuid}")
        self.axon.serve(netuid=self.args.netuid, subtensor=self.subtensor)
        self.axon.start()

        print("Poker44 explicit-wallet supervised miner running.")

        try:
            while True:
                try:
                    block = self.subtensor.get_current_block()
                except Exception as exc:
                    block = f"unknown: {exc}"
                print(f"Miner alive. block={block}")
                time.sleep(300)
        except KeyboardInterrupt:
            self.axon.stop()
            print("Miner stopped.")


if __name__ == "__main__":
    Miner().run()
