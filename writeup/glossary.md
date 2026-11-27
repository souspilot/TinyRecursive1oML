# Glossary

Terms for all writing about this project, the paper included. Use the **term** exactly as written here, and not the
words listed under *Avoid*. Where TRM (Jolicoeur-Martineau 2025), GRAM (Baek et al. 2026) or PTRM (Sghaier et al.
2026) already name something, we use their name; where they disagree, the entry says which we follow and why.

Rules for the paper: it uses only the terms defined here, and a reader with general maths and science knowledge
should be able to follow it with this glossary at hand. Terms marked **[coined]** are ours, not from the literature:
each was agreed with the authors and must be motivated in the paper where it first appears. A new term needs the
same agreement before it is used. The main report (paper.tex) tells only the presentation story; every other run
(N-Queens, the $K$ sweep, second seeds, STRM, other sources of randomness) is in the extra-results document
(extra_results.tex), the future appendix. The main text reports one seed or the mean over seeds.

Symbols: $x$ input, $y$ answer, $h$ / $l$ high-level / low-level state, $\epsilon$ stochastic guidance, $T$ transitions
per supervision step, $n$ low-level updates per transition, $N_{\text{sup}}$ supervision steps, $K$ rollouts,
$\beta$ KL weight, $\sigma$ state-noise scale, $Y(x)$ all solutions, $S(x)$ stored solutions.

---

## 1. Problems and data

**Input** ($x$). The puzzle given to the model: a token sequence of fixed length, one token per cell, with given
cells filled and the rest blank. *GRAM: "input problem". Avoid: question, query, prompt, instance.*

**Answer** ($y$, prediction $\hat y$). The token sequence the model outputs, one token per cell. Any sequence of the
right length is an answer, valid or not. *TRM: "proposed solution"; GRAM: "output". Avoid: prediction (except for the
symbol $\hat y$), output, response.*

**Valid answer / solution.** An answer that satisfies every constraint of the task, checked by the task's checker
(not by comparison with a label). For mazes, a path from start to goal of minimum length. *A solution is always
valid; "invalid solution" is not used.*

**Solution set** ($Y(x)$). Every valid answer of input $x$; $|Y(x)|$ is the **number of solutions**. The datasets
store the exact $|Y(x)|$ for every input.

**Unique-solution input / multi-solution input.** An input with $|Y(x)| = 1$ / $|Y(x)| > 1$. *1oML: "os" / "ms".
Avoid: single-answer, ambiguous input.*

**Stored solutions** ($S(x)$) **[coined]**. The solutions kept in the training data for $x$: all of $Y(x)$ if $|Y(x)| \le 16$,
otherwise 16 drawn at random. $S(x) \subseteq Y(x)$. Training only ever sees stored solutions. *Avoid: labels, targets
(for the set), ground truths.*

**Target.** The stored solution that a loss is computed against in one training step. *GRAM: "target $y$".*

**Unseen solution** **[coined]**. A valid answer to a training input that is not among its stored solutions,
$y \in Y(x) \setminus S(x)$. Only defined for training inputs with $|Y(x)| > 16$. *Avoid: held-out solution, novel
solution (in most papers "held-out" means held-out inputs, not solutions of a training input).*

**Train / validation / test split.** Disjoint sets of inputs. The **validation split** has 1000 inputs (seed 4242 for the
generated tasks; drawn from the validation base puzzles for heyawake and nurikabe) and is used for every choice made
from results (β, run summaries). The test split is used only to report. N-Queens has no
inputs outside train and test, so its validation split is the first 1000 test inputs and its reported test split is
the rest (`test_no_val`).

**Task.** One of the five tasks of the main report: Sudoku, maze, graph colouring (the **generated tasks**), heyawake
and nurikabe (the **human-made tasks**, from Pencil Puzzle Bench, PPBench). N-Queens appears only in the extra-results
document. Written: *Sudoku*, *maze*, *graph colouring* (GRAM: "Graph Coloring"; we use British spelling
consistently), *heyawake*, *nurikabe*, *N-Queens* (10-Queens when the size matters). Our Sudoku task is neither GRAM's
Sudoku-Extreme nor its unconditional Sudoku generation.

**Base puzzle** **[coined]**. A human-made puzzle as published in PPBench, with a unique solution, before it is turned,
mirrored and loosened into inputs. Base puzzles are split 80/10/10 into training, validation and test before any input
is made, so no base puzzle appears in two splits. *Avoid: source puzzle, seed puzzle.*

**Loosen** (a number). Heyawake: remove a room's number. Nurikabe: replace a number by "?" (an area of any size).
Either only removes a rule, so the solutions of the base puzzle stay solutions. *Plain English, not a term of art;
avoid: delete a clue (for nurikabe, deleting the number outright would change the rules).*

**Saturated / unsaturated task** **[coined]**. A task is saturated when the best model compared on it is at least 95%
valid, so validity hardly separates the models and only diversity does. Sudoku and graph colouring are saturated;
maze and heyawake are not (nurikabe pending). *Avoid: solved, easy task.*

---

## 2. Network and recursion (shared by every model here)

**Recursive reasoning model.** A network that produces its answer by applying the same small network many times to
a state, as in HRM and TRM. *GRAM: "Recursive Reasoning Models (RRMs)".*

**High-level state** ($h$) / **low-level state** ($l$). The two per-position hidden states that recursion updates.
The answer is decoded from $h$ only. These are HRM's $z_H$ / $z_L$ and TRM's $y$ / $z$; we follow GRAM's $h$ / $l$
because $z$ means "latent" in TRM but "latent state $(h, l)$" in GRAM. Each block ends in an RMS normalisation without a
learned scale, so every position of $h$ and $l$ has root-mean-square 1. *Avoid: carry, hidden, $z_H$/$z_L$ in prose.*

**Deterministic update** ($u_t$). The new high-level state computed from $h$ and $l$ in one transition, before the
stochastic guidance is added. *GRAM's term and symbol.*

**Transition.** One update of the latent state: $n$ updates of $l$, then one update of $h$ (giving $u_t$), then the
stochastic guidance is added. *GRAM: "latent transition"; TRM: "latent recursion" (its "full recursion process" is the
same computation). Here $n = 4$. Avoid: cycle, H-cycle, iteration (our config names `H_cycles`, `L_cycles` appear only
in code).*

**Supervision step.** $T$ consecutive transitions, followed by decoding an answer from $h$ and applying the loss. Only
the last transition carries gradient; the state is passed to the next supervision step without gradient. Here $T = 3$.
*GRAM and TRM: "supervision step". Avoid: step (alone, which means optimiser step), segment.*

**Deep supervision.** Applying the loss after every supervision step, up to $N_{\text{sup}} = 16$ per input. *TRM and
GRAM term.*

**Halting head.** A linear head on $h$ trained with binary cross-entropy to predict whether the current answer is
valid. In training, an input stops after the supervision step at which the head predicts valid (with exploration). At
test time we never halt early: every sample runs all 16 supervision steps, and the halting head is not used to choose
among samples. *GRAM: "halting head"/"halt head"; TRM: "Q head"; the mechanism is ACT (adaptive computation time).
Avoid: Q head in prose, verifier.*

**Decoding.** Each position's token is the argmax of the output head; no temperature sampling.

**Optimiser step.** One parameter update on 768 sequences (the **global batch**). Training length is counted in
optimiser steps (150k unless stated). *Avoid: iteration.*

**Epoch.** One pass over the training inputs. Runs set the epoch count so that the optimiser steps come within 2.5% of
the target.

**EMA weights.** The exponential moving average of the parameters (decay 0.999), used for every evaluation.

---

## 3. Latent variables

**Stochastic guidance** ($\epsilon_t$). The random per-position vector (256 dimensions) drawn at every transition and
added, after a linear projection, to the deterministic update: $h_t = u_t + W\epsilon_t$. It is the only source of
randomness in a sample unless state noise is on. *GRAM's term ("learnable stochastic guidance"). We say* latent *only
as the adjective (latent variable model) or when quoting. Avoid: latent $z$, noise (reserved for state noise), code.*

**Prior** ($p_\theta(\epsilon_t \mid u_t)$). The distribution of the stochastic guidance used at test time: a diagonal
Gaussian whose mean and variance a head predicts from $u_t$. In the best-of-K baseline the prior is fixed to
$\mathcal N(0, I)$ (the **standard-normal prior**). *GRAM: "prior head".*

**Posterior** ($q_\phi(\epsilon_t \mid u_t, y)$). A diagonal Gaussian predicted from $u_t$ and a representation of the
target. Same architecture as the prior head, separate weights. GRAM draws the stochastic guidance from the posterior
in training. *Avoid: encoder, recognition network.*

**KL term.** $\mathrm{KL}(q_\phi \,\|\, p_\theta)$, summed over positions and transitions. **Training KL** is this
quantity as logged during training, before multiplying by $\beta$.

**KL weight** ($\beta$). The multiplier of the KL term in GRAM's loss. The cross-entropy is averaged over cells while
the KL term is summed, so the useful $\beta$ scales roughly as 1 / (number of labelled cells). *GRAM: "KL weight
$\beta$", "KL coefficient". Avoid: KL coef, beta (spelled out) in formulas.*

**Reference β** ($\beta_{\text{ref}}$) / **tuned β** **[coined]**. The β a task uses by default: the GRAM authors' value where they trained that
task (N-Queens 0.045, graph colouring 0.45, Sudoku 0.05), and elsewhere 4.5 / labelled cells, fixed before any result
on the task: maze 0.02, heyawake and nurikabe 0.045. Every task is also trained at half and twice its reference β. Any other β is a
**tuned β** and is labelled as such wherever it appears.

**KL balancing** (coefficient $\alpha = 0.8$). The KL term is computed as
$\alpha\,\mathrm{KL}(\mathrm{sg}(q)\,\|\,p) + (1-\alpha)\,\mathrm{KL}(q\,\|\,\mathrm{sg}(p))$, where sg stops the
gradient, so the prior moves towards the posterior faster than the reverse (from DreamerV2). *GRAM: "KL balance
coefficient", no symbol; $\alpha$ is ours.*

---

## 4. Training objectives (methods)

**GRAM.** The network above trained with posterior stochastic guidance, one random stored solution per input per
epoch as the target, deep supervision, and loss = cross-entropy + β × KL term (balanced) + halting loss. GRAM calls this
a truncated surrogate of the ELBO; we say **GRAM objective**. *Avoid: VAE, CVAE (except when relating to Sohn et al.).*

**Rollout.** One of the $K$ copies of a training input that the best-of-K objective runs side by side, each with its
own stochastic guidance, through every supervision step. *PTRM: "stochastic rollouts". Used only for training; at test
time the word is* sample.

**Best-of-K objective** (the **best-of-K baseline**, or **BoK** in tables). The same network with the standard-normal
prior in training and at test time, so no posterior, no KL term and no β. Each input is run as $K$ rollouts; the
cross-entropy is applied only to the rollout closest to any stored solution, against that solution, while the halting
loss applies to every rollout:
$\mathcal L(x) = \min_{j \le K} \min_{s \in S(x)} \ell(\hat y_j, s) + \tfrac1K \sum_j \mathcal L_{\text{halt}}(\hat y_j)$.
Default $K = 8$. Best-of-many training (Bhattacharyya et al. 2018) combined with MinLoss. *It is a training
objective only: never "best-of-K sampling", and not GRAM's "best-of-N" (a selection rule, below).*

**MinLoss.** The loss against whichever stored solution is closest to the answer,
$\min_{s \in S(x)} \ell(\hat y, s)$ (Nandwani et al. 2021). The best-of-K objective with $K = 1$ is MinLoss alone.
Motivation (to state in the paper, since 1oML is not widely known): training against one fixed stored solution
penalises a valid answer whenever it differs from that particular solution, so a deterministic model learns an average
of the stored solutions, which is usually invalid; MinLoss only penalises the distance to the nearest stored solution,
so every stored solution is a correct answer.

**Matched compute.** All methods train for the same number of optimiser steps on 768 sequences per step. The
best-of-K objective therefore sees $768 / K$ inputs per step, and each input $K$ times less often than GRAM. At test
time every method draws the same number of samples, each running 16 supervision steps.

---

## 5. Test-time sampling

**Sample.** One answer produced by running an input through all 16 supervision steps with fresh stochastic guidance
at every transition. Every evaluation draws 100 independent samples per input and scores all of them; nothing is
selected or reranked. *GRAM: "parallel sampling", "trajectory" for the sequence of states. Avoid: rollout (training
only), generation, draw (as a noun).*

**Prior sampling.** Stochastic guidance from the prior. The default; "sample" without qualification means this.

**Posterior sampling.** Stochastic guidance from GRAM's posterior at test time, with a stored solution of the input
(drawn at random per sample) as the target. It needs the answer, so it is a **diagnostic**, never a method: it measures
what the network can decode when the stochastic guidance is in the region the posterior trained it on.

**State noise** ($\sigma$) **[coined]**. Gaussian noise $\mathcal N(0, \sigma^2 I)$ added to the high-level state after the
stochastic guidance, in the first $T - 1$ transitions of every supervision step (32 additions per sample), at test
time only. Since $h$ has root-mean-square 1 per position, $\sigma$ is relative to the state's scale on every task.
*This differs from PTRM, which adds its noise to TRM's $z$ (our low-level state) once per supervision step and selects
among rollouts with the Q head; cite PTRM as related, not as the same method. Avoid: test-time noise, perturbation,
temperature.*

**Selection rule.** A way to return one answer from several samples: GRAM's majority voting or best-of-N with its
value head (LPRM), PTRM's best-Q@K. **We use no selection rule**; all metrics are over all samples. *Avoid: best-of-K
for selection.*

---

## 6. Metrics

All metrics are computed per input from 100 samples and averaged over inputs. $v$ = number of valid samples, $d$ =
number of distinct valid answers among them.

**Validity.** The share of samples that are valid, $v / 100$, averaged over all inputs. Equals single-sample accuracy
(GRAM's "Acc." at $N = 1$ for N-Queens, "validity" for Sudoku generation). *Avoid: accuracy, success rate.*

**Unique-solution validity** **[coined]**. Validity on unique-solution inputs only. Since the answer is fully
determined, a drop here means the model fails at the task itself, not that it spreads over solutions. *Counterpart:
1oML's "accuracy on os" is the same quantity for a deterministic model; we name it after validity because our models
draw many samples per input, and it is reported next to validity on all inputs.*

**Validity by number of solutions.** Validity within groups of inputs by $|Y(x)|$ (1, 2–3, 4–15, 16–63, 64–255, 256+).

**Distinct@n** (in full: *distinct valid answers among $n$ samples*). $d$ counted over the first $n$ samples, on
multi-solution inputs. Headline $n = 100$. *Avoid: diversity (as a number), unique answers, distinct solutions.* The
word **diversity** names the property, never a value. *Counterparts: GRAM reports coverage@n (below) and, for
Sudoku generation, the fraction of valid samples that are unique. Distinct@n is preferred as the headline because (i)
coverage divides by $|Y(x)|$, so on inputs with hundreds of solutions it is near 0 for every model at $n = 100$ and the
average over inputs is dominated by inputs with few solutions, while distinct@n is bounded by $\min(n, |Y(x)|)$ and
compares models on every input; and (ii) a fraction among valid samples, like the ratio to uniform, rises when there
are fewer valid samples, while distinct@n counts only valid answers per $n$ samples, so it falls when validity falls.
Report coverage@n alongside where $n$ is comparable to $|Y(x)|$ (N-Queens).*

**Coverage@n.** $d / |Y(x)|$ over the first $n$ samples, on multi-solution inputs: the fraction of all solutions found.
Reported at $n = 10$ where 100 samples already reach every solution (N-Queens). *GRAM: "Cov@20", "coverage, the
fraction of valid solutions recovered".*

**Ratio to uniform** **[coined]**. $d$ divided by $|Y(x)|\,(1 - (1 - 1/|Y(x)|)^{v})$, the expected number of distinct answers in
$v$ uniform draws from the solution set. It conditions on $v$, so it approaches 1 for any model with few valid samples;
reported only in the extra-results document and never used to compare methods.

**Unseen share** / **distinct unseen** **[coined]**. On training inputs with $|Y(x)| > 16$: the share of valid samples that are unseen solutions, against
$1 - 16/|Y(x)|$ for a sampler uniform over $Y(x)$. **Distinct unseen**: the number of distinct unseen solutions among
100 samples.

**Blended answer** **[coined]**. An invalid answer in which every cell agrees with at least one stored solution but which matches
none: the kind of answer a model trained on several targets without usable stochastic guidance outputs.

**Run summary** **[coined]**. The mean of a metric over the last four scored checkpoints of a run, each scored on 500 validation
inputs with 100 samples. Single checkpoints move by 10–20% in distinct@100, so comparisons use run summaries.

**Seed spread** **[coined]**. The difference between two runs that differ only in the training seed; the smallest difference
between methods we read as real. Extra-results document only.

---

## 7. Failure modes

**Posterior collapse.** The posterior matches the prior, so the stochastic guidance carries no information about the
target: training KL ≈ 0 (below $10^{-2}$ here). Samples then become deterministic, and on multi-solution inputs the
model outputs blended answers. Happens above the band. *GRAM's term. Avoid: collapse (alone), mode collapse (GRAM uses
it for a deterministic model losing solutions, a different thing).*

**Prior–posterior mismatch** **[coined]**. The posterior encodes the target in stochastic guidance that the prior does not
produce, so prior sampling gives guidance the network was not trained on. Signature: posterior sampling is valid while
prior sampling is not, including on unique-solution inputs. Happens below the band. *Counterparts: the "holes
problem" (Rezende & Viola 2018) names regions where the prior has mass and the aggregate posterior does not, measured
without a decoder; the amortisation gap is an encoder's suboptimality. Ours is defined by what the network does with
the two kinds of stochastic guidance, which posterior sampling measures directly, so it gets its own name. Avoid:
leakage, prior hole, amortisation gap.*

**Band** (of β) **[coined]**. The range of β in which neither failure dominates. Measured in factors of two around the best β.
