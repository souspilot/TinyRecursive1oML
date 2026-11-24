# Commit, Solve, Sample: project overview

Written for someone new to the project. Status as of 4 November 2026. This is a COL864 project.

## The project in brief

Many reasoning problems have more than one correct answer. A good reasoner has to do three things:

1. **Commit:** make every part of its answer follow the same solution.
2. **Solve:** get the problem right, especially when only one answer exists.
3. **Sample:** produce other correct answers when asked again. Voting, best-of-n and test-time compute all depend on this.

These are hard to separate in LLMs, because nobody knows the full set of correct answers. So we study them on small puzzles where we can count every solution exactly, using tiny recursive reasoning models.

Our main question is **why these models fail on inputs with many answers, and what fixes each failure.** We care most about results of the form "model A with inference X beats model B with inference Y, even though B beats A with plain inference". A list of measurements alone isn't the goal.

## Background

**TRM (Tiny Recursive Model).** It applies one small network repeatedly to a latent state, then predicts every output position at once. Training supervises up to 16 recursion steps, and a "halting head" learns when to stop. It is deterministic, so one input gives one answer.

**One-of-many learning (1oML).** If you train on all solutions of an input, the standard story is that the model averages them and outputs nonsense. 1oML objectives train towards one solution per input instead:

| Objective | Target for each input |
|---|---|
| Naive | All stored solutions |
| Random | One fixed random solution |
| Unique | Only trains on inputs with a unique solution |
| MinLoss | Whichever solution is currently closest |
| I-ExplR | A solution sampled by model likelihood |

**GRAM.** It adds a latent variable to the recursion. During training the latent comes from a posterior that sees the answer, and at test time from a prior that doesn't. One global KL weight, β (`KL_COEF` in the scripts), controls how much information the latent carries:

- **Too low, it leaks:** training looks perfect because the answer passes through the latent, but prior samples are junk.
- **Too high, it collapses:** every sample comes out the same.

**Decoding.** Besides the normal one-shot readout, we use *commit-and-re-run*: fix one decision, feed it back as part of the input, run the model again, and repeat. The idea comes from self-reducibility in complexity theory (Jerrum, Valiant and Vazirani, 1986): with exact conditional probabilities, this samples solutions uniformly.

## Tasks

| Task | Size | Solutions per input | Notes |
|---|---|---|---|
| N-Queens completion | 10×10, 1–3 queens given | 1 to 20+ | Test inputs also appear in training, so it can be memorised |
| Mazes | 15×15 | 1–924 shortest paths, counted exactly | Test mazes are new |
| Random 3-SAT | 16 variables | 1 to thousands, enumerated exactly | Newest task, tests whether our findings carry over |

## How the project evolved

1. **We started by trying to improve GRAM.** The idea was that one KL weight can't suit both unique and highly ambiguous inputs, and that a per-input KL budget would fix it. This was mostly refuted: training twice as long removed most of the trade-off, and the budget failed outright (3.4% and 0%).
2. **Decoding turned out to be the bigger story.** Naive TRM doesn't appear to fail by averaging. Re-running it one decision at a time helps enormously.
3. **We then moved to three hypotheses:**
   - **D1 (decoding):** naive training learns the solutions and fails at decoding them.
   - **D2 (two kinds of error):** what remains are solver errors on unique inputs, and the training objective moves those in a direction that depends on the domain.
   - **D3 (diversity):** a deterministic model picks its answer in the first recursion steps, so perturbing its starting state or the input's symmetries at test time gives diversity without training a latent.

`writeup/report.tex` has an older set of hypotheses (H1–H5) and how each one fared.

## Results so far

These are mostly seed 42. N-Queens has shown up to 8 points of variance between seeds. Numbers are the percentage of valid answers on one sample unless stated.

| Task | Model and decoding | Result | On unique inputs | Coverage at 100 samples |
|---|---|---|---|---|
| Mazes | Naive TRM, one shot | 37.2 | 97.6 | 2.6 |
| Mazes | Naive TRM, re-run every step (about 15 passes) | **99.4** | 98.5 | n/a |
| Mazes | MinLoss, one shot | 94.0 | 85.3 | 8.4 |
| Mazes | MinLoss, noise + symmetries | 94.0 | n/a | 54.1 |
| Mazes | I-ExplR, one shot | 96.6 | 92.8 | 8.6 |
| Mazes | GRAM, β=0.015, 100k steps, with symmetries | 91.3 | 83.7 | **68.7** |
| N-Queens | Naive TRM: one shot / row readout / re-run | 1.0 / 66.7 / 77.9 | 62.7 (re-run) | n/a |
| N-Queens | MinLoss, noise + symmetries | **96.6** | 93.5 | 99.5 |
| N-Queens | GRAM | 91.0 | 82.8 | **99.9** |

*Coverage* is the share of all solutions found on multi-solution inputs. On unique N-Queens boards, naive TRM's 62.7 compares with about 93 for MinLoss.

What we take from this so far:

- **A real crossover on mazes:** naive TRM is the worst one-shot model, but the best single-answer model once it re-runs at every step. The next test is whether MinLoss and I-ExplR also rise above it with the same decoding.
- **Objectives mainly affect solver errors:** naive is the best solver on unique mazes and the worst on unique N-Queens boards.
- **Test-time diversity works:** a deterministic model nearly matches GRAM on N-Queens, but falls short on mazes (54% vs 69% coverage, and less even).
- **GRAM's real strength is even, scattered diversity:** its errors cancel under voting. Its first sample is not its strength.
- **The halting head as a verifier is split:** it helps TRM (96.4 → 99.3 on mazes) and hurts GRAM (91.3 → 76.7).

## Known problem: part of the write-up needs re-measuring

**We found a bug on 4 November.** Models train with a stablemax loss, but our decoders read their probabilities with softmax, which exaggerates confidence: a true 0.90 read as 0.98. This is now fixed on `feature/sat-dataset`, and the old softmax numbers are treated as wrong.

| Status | What it covers |
|---|---|
| **Unaffected** | One-shot, test-time diversity and GRAM sampling accuracy, since these don't use the probabilities |
| **Mostly unaffected** | Greedy every-step decoding, which picks the same value per decision, though the commit order may shift slightly |
| **Must be re-measured** | The per-cell marginals, `tie_breaking.py` confidence numbers, anything gated on 0.99 ("re-run when unsure"), and sampled commit and walk results |

So **"confidently inconsistent" and "abstains at forks" are not established until re-measured**, even though `report.tex` and `proposal.tex` state them. The accuracy figures for D1 should hold, but the explanation around them may change.

## 3-SAT: the current work

Each formula encodes as 232 tokens: 16 variable slots and 3 tokens per clause. Only the 16 variable slots are labelled. Each model trained for about 50k steps, on seed 42. GRAM uses β=0.2, scaled up from the maze value because the KL is summed while the reconstruction loss is averaged per labelled token.

**No model solves it,** and memorisation is not the reason:

| Model or rule | Train | Test | Test, 100 tries |
|---|---|---|---|
| Random guess | n/a | 0.4 | 20.4 |
| Naive TRM | 6.5 | 6.6 | 20.2 (views) |
| MinLoss | 7.1 | 6.3 | 19.4 (views) |
| I-ExplR | 7.6 | 6.7 | 17.3 (views) |
| GRAM (prior) | 0.7 | 0.7 | 29.6 |
| Majority sign, one pass | n/a | 7.8 | 13.2 |
| Majority sign, re-counted after each commit | n/a | **58.5** | **79.5** |

The random-guess row is exact on the test set. The majority-sign rows are from one run on 2,000 fresh formulas from the same generator. *Views* means re-encoding the same formula with variables renamed and negated and clauses reordered.

- **TRM scores about what counting signs scores.** Its pattern across solution counts looks similar too, but we haven't compared their answers on the same formulas yet.
- **The task is within reach of a simple sequential rule.** The re-counted rule has no backtracking, and it gets 58.5%.
- **GRAM's lead at 100 samples isn't meaningful yet.** It's only about 1.5× random guessing.
- **GRAM's latent seems to carry the answer.** Its training metric reads about 81% with the posterior, while the prior gets 0.7% on training formulas.
- **The objectives can't be compared yet.** Everything sits near the floor, so differences between them are within noise.

**Leading hypothesis (untested).** The dataset gives at most 25% of the variables, but commit-and-re-run moves on to partial assignments with most variables set, which the model has never seen. Self-reducibility needs a predictor that's good on all partial assignments. If so, models trained across the full range of givens should stay near sign-count level in one shot and rise sharply with commit decoding. One caution: hint training added nothing on mazes.

## Open work

1. Rerun SAT commit decoding with the fixed probabilities. The latest summary still has the old samples.
2. Re-measure the N-Queens and maze marginals and `tie_breaking.py`, then correct `report.tex` and `proposal.tex`.
3. Add both sign-count baselines to `scripts/analyze_sat_samples.py`, and measure how often TRM's answers agree with the one-pass rule.
4. Train TRM on SAT with `max_given_share` near 1, tested on the same kind of formulas (about 6 hours per run).
5. From the report's plan: re-run-every-step for MinLoss, I-ExplR and GRAM; second seeds; mode mixing in failed N-Queens boards.

## Finding your way around

| Location | Contents |
|---|---|
| `pretrain.py`, `models/` | Training, models, losses (including the 1oML objectives) |
| `dataset/build_*_dataset.py` | Dataset builders |
| `evaluators/` | Exact verifiers and solution counting |
| `sample_predictions.py` | One-shot sampling with noise, symmetries and SAT views |
| `commit_decode.py`, `walk_decode.py`, `sat_commit_decode.py` | Commit-and-re-run for each task |
| `scripts/*_compare.sh` | Train a method: `bash scripts/sat_compare.sh <method>` |
| `scripts/sat_sampling.sh`, `scripts/maze_h2_worker.sh` | Resumable sampling queues, one worker per GPU |
| `scripts/analyze_*_samples.py` | Turn samples into `results/<task>/summary.md` |
| `scripts/sat_train_test.py` | SAT validity on train vs test formulas |
| `writeup/` | `proposal.tex` and `report.tex` |

**Compute.** Jobs run under PBS on the IIT Delhi cluster (scai nodes), with 2× A100 80GB per job and a 24-hour walltime. A TRM run takes about 6 hours, and GRAM about 9.

**Where things live.** Checkpoints, samples and `results/` are on the cluster under `~/TinyRecursive1oML`, not in git. Training curves are on W&B in the `Sat-16-compare` project and its siblings, and some uploads stalled partway.

**Git.** PRs go to the fork `souspilot/TinyRecursive1oML`, never upstream `ahn-ml/gram-release`. The latest work is on `feature/sat-dataset`. PRs [#14](https://github.com/souspilot/TinyRecursive1oML/pull/14) (walk decoding) and [#15](https://github.com/souspilot/TinyRecursive1oML/pull/15) (maze round 2) are still open.

## Things to watch out for

- **GRAM's training metrics mislead.** They use the posterior, which sees the answer, so watch the eval metrics or measure samples directly.
- **Training metrics are partial.** They only cover examples that halted at that step, and the losses are per step. The final W&B summary can rest on a couple of dozen examples.
- **Always compare with a simple baseline.** On SAT, the numbers looked like learning until we computed the sign-count rule.
- **Continued runs aren't valid evidence.** Continuing a finished run at constant learning rate degraded every model, so compare only runs trained from scratch.
- **Compare compute fairly.** Re-running costs about 15 passes per maze, so compare it with sampling at similar cost.
- **β is task-specific and sensitive.** The good range on mazes is narrow (0.01 leaks, 0.02 collapses).
