# Validity or diversity: GRAM on puzzles with many solutions

COL864 project, built on the [GRAM](https://github.com/ahn-ml/gram-release) code.

**Start here:** [`writeup/icml2027/paper.pdf`](writeup/icml2027/paper.pdf) is the current report (status 24 November
2026), and [`writeup/glossary.md`](writeup/glossary.md) defines every term it uses. Runs outside the main story are in
[`writeup/icml2027/extra_results.pdf`](writeup/icml2027/extra_results.pdf). Everything else in `writeup/`
(`PROJECT_OVERVIEW.md`, `proposal*.tex`, `report*.tex`) is from earlier phases of the project and out of date.

## The story

GRAM makes a tiny recursive reasoning network generative: random stochastic guidance at every transition, drawn from
a posterior that sees the target in training and from a learned prior at test time, with a KL weight β. We compare it
at matched compute with baselines whose stochastic guidance never sees the target. On all five tasks GRAM gives the
most distinct valid answers, a baseline is the most valid, and no model is both. Three questions support this:

1. **Q1:** how should β be chosen for a new task? (It works only in a narrow band.)
2. **Q2:** is a posterior that sees the target needed? (Not for validity; but no posterior-free model reaches GRAM's
   diversity.)
3. **Q3:** do the models produce solutions that were never in their training data? (Yes, at a uniform sampler's rate.)

## Models (one backbone, `models/recursive_reasoning/gram.py`)

| Model | `scripts/multi_compare.sh` method | Notes |
|---|---|---|
| GRAM | `gram`, β via `KL_COEF` | reference β by default; also trained at half and twice it |
| Best-of-K | `gram_std_bestofk` | N(0, I) prior, K = 8 rollouts, loss on the rollout closest to a stored solution |
| MinLoss | `gram_std_bestofk` with `ROLLOUTS=1` | best-of-K with one rollout |
| State noise | `NOISE=<sigma>` in `scripts/multi_evaluate.sh` | test time only, added to the high-level state |
| Posterior sampling | `LATENT=posterior` in `scripts/multi_evaluate.sh` | diagnostic for GRAM, needs the answer |

Extra-results only: STRM (`strm_bestofk`), sample-level latent (`LATENT_SCOPE=sample`), training-time noise (`TRAIN_NOISE`).

## Tasks

| Task | `<task>` | Solutions per input | Source |
|---|---|---|---|
| Sudoku 9x9 | `sudoku` | 1-1024 | generated |
| Maze 15x15 | `maze` | 1-924 | generated |
| Graph 3-colouring, 10 nodes | `graph` | 1-9842 | generated |
| Heyawake 10x10 | `heyawake` | 1-1023 | Pencil Puzzle Bench, human-made |
| Nurikabe 10x10 | `nurikabe` | 1-1023 | Pencil Puzzle Bench, human-made |

N-Queens (`nqueens`) is extra-results only; light up (`lightup`) is built but unused. At most 16 solutions are stored
per training input, so inputs with more have solutions held out of training (Q3).

## Usage

```bash
pip install -r requirements.txt
bash scripts/build_multi_datasets.sh sudoku                                         # CPU only
PPBENCH_SOURCE=path/to/full_dataset.jsonl bash scripts/build_multi_datasets.sh heyawake nurikabe   # needs pip install clingo
EXTRA_ARGS="wandb_offline=True" bash scripts/multi_compare.sh sudoku gram_std_bestofk  # train 150k steps
bash scripts/score_checkpoints.sh sudoku gram_std_bestofk_steps150000_seed42           # score checkpoints on validation
python scripts/checkpoint_trends.py gram_std_bestofk_steps150000_seed42 sudoku          # run summary (last 4 checkpoints)
NOISE=0.5 QUERIES=500 bash scripts/multi_evaluate.sh sudoku gram_std_bestofk_steps150000_seed42 val   # state noise
bash scripts/multi_evaluate.sh sudoku gram_kl0.05_steps150000_seed42 train            # unseen solutions (Q3)
python scripts/report_numbers.py sudoku                                                # print those results
```

Pencil Puzzle Bench's `full_dataset.jsonl` is at huggingface.co/datasets/bluecoconut/pencil-puzzle-bench. The
validation split is for every choice made from results; the test split is only for reporting. See each script's
header for options. The code before the project was narrowed to these models is at the git tag `pre-prune`.
