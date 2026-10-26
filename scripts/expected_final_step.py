"""Print the step a finished run's last checkpoint should have, from its all_config.yaml.

Usage: python scripts/expected_final_step.py checkpoints/<project>/<run>/all_config.yaml

The train loader packs each eval interval's epochs into full batches and drops the last partial one, so a
run ends at (epochs // eval_interval) * (eval_interval * train_groups // (global_batch_size // rollouts)). This assumes
one example per group, as in the N-Queens datasets.
"""
import json
import os
import sys

import yaml


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_groups = 0
    for data_path in config["data_paths"]:
        with open(os.path.join(data_path, "train", "dataset.json")) as f:
            train_groups += json.load(f)["total_groups"]
    eval_interval = config["eval_interval"] or config["epochs"]
    # With several rollouts per example, each step uses global_batch_size // rollouts examples
    examples_per_step = config["global_batch_size"] // config.get("rollouts", 1)
    print((config["epochs"] // eval_interval) * (eval_interval * train_groups // examples_per_step))


if __name__ == "__main__":
    main(sys.argv[1])
