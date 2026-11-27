"""Print the single-checkpoint results the report needs (state noise, posterior sampling, unseen solutions), one line each.

Usage: python scripts/report_numbers.py [task ...]    (default: every task with results)
       STEPS=1 python scripts/report_numbers.py ...      (also single-step evaluations with a sampling option, e.g. <run>_noise1_step<N>)

Reads results/<task>-val/*.json, results/<task>-test/*.json and results/<task>-train/*.json written by scripts/multi_evaluate.sh, skipping the
per-checkpoint files of scripts/score_checkpoints.sh (those go through scripts/checkpoint_trends.py). Prints validity,
unique-solution validity and distinct@100; for the train split also the unseen share against a uniform sampler's and
distinct unseen per input.
"""
import glob
import json
import os
import re
import sys

tasks = sys.argv[1:] or sorted({os.path.basename(d).rsplit("-", 1)[0] for d in glob.glob("results/*-val") + glob.glob("results/*-test") + glob.glob("results/*-train")})
for task in tasks:
    for split in ("val", "test", "train"):
        for path in sorted(glob.glob(f"results/{task}-{split}/*.json")):
            name = os.path.basename(path)[:-5]
            if name == "summary" or (re.search(r"_step\d+$", name) and not (os.environ.get("STEPS") and re.search(r"_(noise[\d.]+|posterior)_step\d+$", name))):
                continue
            with open(path) as f:
                r = json.load(f)
            g = r.get("groups", {})
            if "all" not in g:
                continue
            div = g["all"].get("diversity", {}).get(str(r["num_samples"]), {})
            unique = 100 * g["1"]["validity"] if "1" in g else float("nan")
            line = (f"{task:9s} {split:5s} {name:62s} valid {100 * g['all']['validity']:5.1f}  unique {unique:5.1f}  "
                    f"distinct {div.get('distinct_valid', float('nan')):5.1f}")
            h = r.get("held_out")
            if h:
                line += (f"  | unseen {100 * h['unseen_share_of_valid']:.1f} (uniform {100 * h['uniform_unseen_share']:.1f}),"
                         f" distinct unseen {h['distinct_unseen_per_query']:.1f}, inputs {h['queries']}")
            print(line)
