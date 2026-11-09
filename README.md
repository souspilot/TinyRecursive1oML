# Matching GRAM with a simpler training objective

COL864 project, built on the [GRAM](https://github.com/ahn-ml/gram-release) code. GRAM trains its latent with a
posterior that sees the answer, and needs a KL weight β chosen per task. We ask:

1. How should β be chosen for a new task, so training neither collapses nor leaks the answer?
2. Does training without a posterior (no answer in the latent, no KL, no β) match GRAM?
3. Do these models produce valid solutions that were never in their training data?

## Runs

| Run | Method | Script argument |
|---|---|---|
| 1 | GRAM with a sweep of β | `gram` with `KL_COEF` |
| 2 | Best-of-K with N(0, I) latents, no posterior | `gram_std_bestofk` |

Both use the same recursive backbone (`models/recursive_reasoning/gram.py`) on four tasks with several solutions per
input: 15x15 mazes, 10x10 N-Queens, Sudoku and graph 3-colouring. At most 16 solutions are stored per training input,
so inputs with more have solutions held out of training.

## Usage

```bash
pip install -r requirements.txt
bash scripts/build_multi_datasets.sh                                   # all four datasets, CPU only
STEPS=150000 bash scripts/multi_compare.sh sudoku gram_std_bestofk     # train (or: KL_COEF=0.1 ... sudoku gram)
bash scripts/multi_evaluate.sh sudoku gram_std_bestofk_steps150000_seed42   # sample and score, test and train
bash scripts/score_checkpoints.sh sudoku gram_std_bestofk_steps150000_seed42   # score checkpoints on the validation split
python scripts/checkpoint_trends.py gram_std_bestofk_steps150000_seed42         # validity and diversity per checkpoint
```

The validation split is for choosing beta and reading training curves; the test split is only for reporting. N-Queens has no unused inputs, so its validation split is the first 1000 test inputs, and its final numbers come from `test_no_val` (see `dataset/build_validation_split.py`).

`scripts/analyze_multi_samples.py --check_data` checks a dataset on its own. See each script's header for options.

The code before this project was narrowed down is at the git tag `pre-prune`. `writeup/` holds the proposals and reports.
