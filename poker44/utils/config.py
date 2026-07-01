"""Bittensor configuration helpers vendored for the Poker44 subnet."""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import bittensor as bt


def add_args(cls, parser: argparse.ArgumentParser) -> None:
    if parser is None:
        parser = argparse.ArgumentParser()

    bt.logging.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.Axon.add_args(parser)

    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=126)

    # Required by check_config().
    parser.add_argument(
        "--neuron.name",
        type=str,
        default="miner",
        help="Neuron name used for logging/checkpoint paths.",
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        default="cpu",
        help="Torch device to execute forwards on.",
    )
    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        default=50,
        help="Blocks between mandatory syncs.",
    )
    parser.add_argument(
        "--neuron.disable_set_weights",
        action="store_true",
        help="Skip setting weights on-chain.",
    )
    parser.add_argument(
        "--neuron.wait_for_inclusion",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be included.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_inclusion",
        action="store_false",
        dest="neuron.wait_for_inclusion",
        help="Do not wait for inclusion.",
    )
    parser.add_argument(
        "--neuron.wait_for_finalization",
        action="store_true",
        default=True,
        help="Wait for weight-setting extrinsics to be finalized.",
    )
    parser.add_argument(
        "--no-neuron.wait_for_finalization",
        action="store_false",
        dest="neuron.wait_for_finalization",
        help="Do not wait for finalization.",
    )
    parser.add_argument(
        "--neuron.moving_average_alpha",
        type=float,
        default=0.05,
        help="Exponential moving average smoothing factor.",
    )
    parser.add_argument(
        "--neuron.num_concurrent_forwards",
        type=int,
        default=1,
        help="Concurrent forward coroutines.",
    )
    parser.add_argument(
        "--neuron.timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds for validator-to-miner query.",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=5 * 60,
        help="Default delay between validator ingestion cycles.",
    )
    parser.add_argument(
        "--neuron.axon_off",
        action="store_true",
        help="Disable serving the axon endpoint.",
    )

    parser.add_argument(
        "--blacklist.force_validator_permit",
        action="store_true",
        default=True,
        help="Only allow requests from validators with permits.",
    )
    parser.add_argument(
        "--no-blacklist.force_validator_permit",
        action="store_false",
        dest="blacklist.force_validator_permit",
        help="Allow registered callers without validator permits.",
    )
    parser.add_argument(
        "--blacklist.allow_non_registered",
        action="store_true",
        default=False,
        help="Allow requests from non-registered entities.",
    )
    parser.add_argument(
        "--blacklist.allowed_validator_hotkeys",
        nargs="*",
        default=[],
        help="Optional allowlist of validator hotkeys permitted to query miners.",
    )

    parser.add_argument(
        "--wandb.off",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb.offline",
        action="store_true",
        default=False,
        help="Run Weights & Biases offline.",
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        default="poker44-validators",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        default="",
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--wandb.notes",
        type=str,
        default="",
        help="Optional notes for W&B.",
    )


def add_validator_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--validator.manual_players",
        nargs="*",
        default=[],
        help="Player descriptors to track manually.",
    )


def add_miner_args(cls, parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--miner.mock",
        action="store_true",
        help="Placeholder flag retained for compatibility.",
    )


def _ensure_neuron_defaults(config: "bt.Config") -> None:
    if getattr(config, "neuron", None) is None:
        config.neuron = SimpleNamespace()

    defaults = {
        "name": "miner",
        "device": "cpu",
        "epoch_length": 50,
        "disable_set_weights": False,
        "wait_for_inclusion": True,
        "wait_for_finalization": True,
        "moving_average_alpha": 0.05,
        "num_concurrent_forwards": 1,
        "timeout": 180.0,
        "axon_off": False,
    }

    for key, value in defaults.items():
        if not hasattr(config.neuron, key) or getattr(config.neuron, key) is None:
            setattr(config.neuron, key, value)


def check_config(cls, config: "bt.Config"):
    """Checks/validates the config namespace object."""
    _ensure_neuron_defaults(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )

    config.neuron.full_path = os.path.expanduser(full_path)

    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)


def config(cls) -> "bt.Config":
    parser = argparse.ArgumentParser()
    cls.add_args(parser)

    # SDK compatibility across bittensor versions.
    factory = getattr(bt, "config", None)
    if callable(factory):
        try:
            cfg = factory(parser)
        except TypeError:
            cfg = factory(parser=parser)
    else:
        cfg = bt.Config(parser=parser)

    _ensure_neuron_defaults(cfg)
    return cfg
