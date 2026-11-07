"""Sample predictions from a checkpoint on a test split and save them in resumable shards.

Each shard holds every sample for a contiguous range of test queries and is written atomically. Shards are
divided between processes (torchrun ranks), and a restarted job skips shards that already exist, so an
interrupted run only loses the shards in progress. Sampling is seeded by the queries in each batch, so the
result doesn't depend on how many processes or jobs produced it.

Usage:
    torchrun --nproc-per-node 2 sample_predictions.py --checkpoint checkpoints/<project>/<run>/final/checkpoint.pt \\
        --data data/nqueens-10-1oml --num_samples 100 --out samples/nqueens_10/<run>
    (several splits with one model load: --split test,train --out samples/nqueens-{split}/<run>)

Output: <out>/meta.json and <out>/shard_<i>.npz with `query_start`, `preds` (queries x samples x seq_len, uint8)
and `q` (the halting head's final logit per sample, the model's own estimate that its answer is correct).

Test-time options: --halt_steps runs more (or fewer) recursion steps than in training (depth scaling), and
--latent_noise adds Gaussian noise of that scale to the latent state z_L before each recursion step after the
first (as in the Probabilistic Tiny Recursive Model, which then picks the sample with the highest halting logit).
--init_noise adds the noise to the initial latent state (z_H and z_L) instead: the recursion commits to one of
several solutions within its first supervision step (scripts/tie_breaking.py), so later noise comes too late.
--symmetry decodes sample j on the board turned by the j-th of the square's 8 symmetries (rotations, reflections)
and maps the answer back; it suits tasks whose rules are symmetric (N-Queens, mazes). --sat_views is its analogue
for 3-SAT (dataset/build_sat_dataset.py): every sample after the first decodes the formula with its variables
renamed, their signs flipped, and its clauses and their literals reordered at random, and maps the assignment back.
"""
import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import torch

# Checkpoints pickle these classes from pretrain.py run as __main__; importing them here lets torch.load find them
from pretrain import PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig  # noqa: F401
from dataset.common import PuzzleDatasetMetadata
from models.ema import EMAHelper
from models.losses import IGNORE_LABEL_ID
from utils.functions import load_model_class


def load_split(data_path: str, split: str = "test"):
    split_dir = os.path.join(data_path, split)
    with open(os.path.join(split_dir, "dataset.json")) as f:
        metadata = PuzzleDatasetMetadata(**json.load(f))
    data = {}
    for field in ("inputs", "labels", "puzzle_identifiers", "puzzle_indices", "label_set", "num_solutions"):
        path = os.path.join(split_dir, f"all__{field}.npy")
        if os.path.exists(path):
            data[field] = np.load(path, mmap_mode="r")
    return data, metadata


def make_batch(data, metadata: PuzzleDatasetMetadata, start: int, end: int):
    """The batch PuzzleDataset would collate for test examples [start, end)."""
    puzzle_ids = np.searchsorted(data["puzzle_indices"], np.arange(start, end), side="right") - 1
    batch = {
        "inputs": data["inputs"][start:end],
        "labels": data["labels"][start:end],
        "puzzle_identifiers": data["puzzle_identifiers"][puzzle_ids],
    }
    for field in ("label_set", "num_solutions"):
        if field in data:
            batch[field] = data[field][start:end]
    batch = {k: torch.from_numpy(np.array(v, dtype=np.int32)) for k, v in batch.items()}
    if metadata.ignore_label_id is not None:
        for field in ("labels", "label_set"):
            if field in batch:
                batch[field][batch[field] == metadata.ignore_label_id] = IGNORE_LABEL_ID
    return batch


def load_model(checkpoint_path: str, metadata: PuzzleDatasetMetadata, temperature: float, device: str):
    """Build the model with its loss head and load its EMA weights, as eval_pass_at_k.py does."""
    # Checkpoints pickle these classes as defined in __main__ (pretrain.py); make them findable there whichever
    # script is running
    main_module = sys.modules["__main__"]
    for cls in (PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig):
        if not hasattr(main_module, cls.__name__):
            setattr(main_module, cls.__name__, cls)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
        temperature=temperature,
    )
    with torch.device(device):
        model = load_model_class(config.arch.name)(model_cfg)
        # rollouts only matters in training, but best-of-K loss heads check it on construction
        model = load_model_class(config.arch.loss.name)(model, **config.arch.loss.__pydantic_extra__,
                                                         rollouts=getattr(config, "rollouts", 1))

    strip = lambda state: {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(strip(checkpoint["model"]), assign=True)
    if "ema" in checkpoint:
        ema = EMAHelper()
        ema.register(model)
        ema.load_state_dict(strip(checkpoint["ema"]))
        ema.ema(model)

    if temperature > 0 and not hasattr(model.model.config, "temperature"):
        print(f"Warning: {config.arch.name} has no temperature setting, so --temperature is ignored")
    return model.to(device).eval(), checkpoint.get("step")


SYMMETRIES = [(k, flip) for flip in (False, True) for k in range(4)]  # quarter turns, then a left-right flip


def apply_views(x: torch.Tensor, views: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Transform each flattened square board in x (N, n*n) by its symmetry SYMMETRIES[views[i]] (or undo it)."""
    n = int(round(x.shape[1] ** 0.5))
    out = x.clone()
    for v, (k, flip) in enumerate(SYMMETRIES):
        rows = views == v
        if not rows.any():
            continue
        g = x[rows].view(-1, n, n)
        if not inverse:
            g = torch.rot90(g, k, (1, 2))
            g = g.flip(2) if flip else g
        else:
            g = g.flip(2) if flip else g
            g = torch.rot90(g, -k, (1, 2))
        out[rows] = g.reshape(len(g), -1)
    return out


def sat_views(x: torch.Tensor, n: int, identity: torch.Tensor):
    """A random equivalent of each 3-SAT formula in x (N, n + 3 * max_clauses): variable i becomes perm[:, i],
    negated where flip[:, i], with clauses (padding kept last) and the literals in each clause shuffled. Rows where
    identity is set are left as they are. Returns the new inputs, perm and flip (for undo_sat_views)."""
    num, device = len(x), x.device
    perm = torch.where(identity[:, None], torch.arange(n, device=device), torch.rand(num, n, device=device).argsort(1))
    flip = ~identity[:, None] & (torch.rand(num, n, device=device) < 0.5)
    # Token map per row: POS(i) -> POS/NEG(perm[i]), NEG(i) -> NEG/POS(perm[i]), FREE(i) -> FREE(perm[i])
    table = torch.arange(3 + 3 * n, device=device).repeat(num, 1)
    table[:, 3:3 + n] = torch.where(flip, 3 + n + perm, 3 + perm)
    table[:, 3 + n:3 + 2 * n] = torch.where(flip, 3 + perm, 3 + n + perm)
    table[:, 3 + 2 * n:] = 3 + 2 * n + perm
    mapped = table.gather(1, x.long())
    slots = torch.empty_like(mapped[:, :n]).scatter_(1, perm, mapped[:, :n])
    clauses = mapped[:, n:].view(num, -1, 3)
    shuffle = lambda shape: torch.rand(shape, device=device)
    literal_order = torch.where(identity[:, None, None], torch.arange(3, device=device), shuffle(clauses.shape).argsort(2))
    clauses = clauses.gather(2, literal_order)
    is_pad = clauses[:, :, 0] == 0
    clause_order = torch.where(identity[:, None], torch.arange(clauses.shape[1], device=device),
                               (shuffle(is_pad.shape) + 2 * is_pad).argsort(1))
    clauses = clauses.gather(1, clause_order[:, :, None].expand_as(clauses))
    return torch.cat([slots, clauses.flatten(1)], 1).to(x.dtype), perm, flip


def undo_sat_views(preds: torch.Tensor, perm: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Map assignments decoded on sat_views formulas back to the original variables (FALSE = 1, TRUE = 2)."""
    n = perm.shape[1]
    values = preds[:, :n].gather(1, perm)
    values = torch.where(flip & ((values == 1) | (values == 2)), 3 - values, values)
    return torch.cat([values, preds[:, n:]], 1)


@contextlib.contextmanager
def noisy_init(model, scale: float, generator: torch.Generator = None):
    """Add Gaussian noise of this scale to the initial latent state (z_H and z_L, per example and position)."""
    if scale <= 0:
        yield
        return
    inner = model.model.inner
    original = inner.reset_carry

    def reset(flag, carry):
        carry = original(flag, carry)
        for field in ("z_H", "z_L"):
            z = getattr(carry, field)
            noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=torch.float32).to(z.dtype)
            setattr(carry, field, torch.where(flag.view(-1, 1, 1), z + scale * noise, z))
        return carry

    inner.reset_carry = reset
    try:
        yield
    finally:
        inner.reset_carry = original


@torch.inference_mode()
def sample(model, batch, num_samples: int, device: str, latent_noise: float = 0.0, init_noise: float = 0.0,
           symmetry: bool = False, sat_vars: int = 0):
    """Run the model to completion on num_samples copies of each query.

    sat_vars: with n > 0, decode every sample after each query's first on a random sat_views equivalent (n variables).
    Returns preds (queries, samples, seq_len) and the final halting logits (queries, samples)."""
    expanded = {k: v.to(device).repeat_interleave(num_samples, dim=0) for k, v in batch.items()}
    views = torch.arange(len(expanded["inputs"]), device=device) % num_samples % len(SYMMETRIES) if symmetry else None
    if symmetry:
        expanded["inputs"] = apply_views(expanded["inputs"], views)
    if sat_vars:
        identity = torch.arange(len(expanded["inputs"]), device=device) % num_samples == 0
        expanded["inputs"], perm, flip = sat_views(expanded["inputs"], sat_vars, identity)
    with torch.device(device):
        carry = model.initial_carry(expanded)
    first = True
    with noisy_init(model, init_noise):
        while True:
            if latent_noise > 0 and not first:
                z_l = carry.inner_carry.z_L
                carry.inner_carry.z_L = z_l + latent_noise * torch.randn_like(z_l)
            carry, _, _, outputs, all_finish = model(carry=carry, batch=expanded, return_keys={"preds", "q_halt_logits"})
            first = False
            if all_finish:
                break
    preds = apply_views(outputs["preds"], views, inverse=True) if symmetry else outputs["preds"]
    if sat_vars:
        preds = undo_sat_views(preds, perm, flip)
    queries = batch["inputs"].shape[0]
    return preds.view(queries, num_samples, -1), outputs["q_halt_logits"].float().view(queries, num_samples)


# Tokens per forward pass (queries x samples x positions) that fit an 80GB A100 with room to spare: the default
# --effective_batch_size is this divided by the sequence length (with the puzzle embedding positions)
TOKENS_PER_PASS = 1_500_000


def sample_split(args, split, out, max_queries, rank, world_size, device, get_model):
    """Sample one split into resumable shards at out; get_model() loads the model on first use."""
    data, metadata = load_split(args.data, split)
    num_queries = len(data["inputs"]) if max_queries is None else min(max_queries, len(data["inputs"]))
    num_shards = math.ceil(num_queries / args.shard_size)

    # The settings that determine the samples; a rerun with different ones must not mix with existing shards
    meta = dict(checkpoint=args.checkpoint, data=args.data, num_samples=args.num_samples, temperature=args.temperature,
                shard_size=args.shard_size, seed=args.seed, num_queries=num_queries, num_shards=num_shards)
    # Recorded only when set, so directories sampled before these options existed still resume
    meta.update({k: v for k, v in (("halt_steps", args.halt_steps), ("latent_noise", args.latent_noise),
                                   ("init_noise", args.init_noise), ("symmetry", args.symmetry),
                                   ("sat_views", args.sat_views),
                                   ("split", split if split != "test" else None),
                                   ("probabilities", "training_loss" if args.temperature > 0 else None)) if v})
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
        mismatched = {k: (existing.get(k), v) for k, v in meta.items() if existing.get(k) != v}
        if mismatched:
            raise ValueError(f"{out} holds samples made with other settings (existing, requested): {mismatched}. "
                             f"Use another --out or delete it.")
    else:
        with open(f"{meta_path}.tmp{rank}", "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(f"{meta_path}.tmp{rank}", meta_path)

    shard_path = lambda i: os.path.join(out, f"shard_{i:05d}.npz")
    todo = [i for i in range(rank, num_shards, world_size) if not os.path.exists(shard_path(i))]
    print(f"[rank {rank}/{world_size}] {out}: {len(todo)} of {len(range(rank, num_shards, world_size))} shards to sample",
          flush=True)
    if not todo:
        return

    model = get_model(metadata)
    batch_size = args.effective_batch_size or max(args.num_samples, TOKENS_PER_PASS // (metadata.seq_len + 16))
    queries_per_pass = max(1, batch_size // args.num_samples)
    sat_vars = (metadata.vocab_size - 3) // 3 if args.sat_views else 0

    for done, shard in enumerate(todo, 1):
        start, end = shard * args.shard_size, min((shard + 1) * args.shard_size, num_queries)
        preds = np.empty((end - start, args.num_samples, metadata.seq_len), dtype=np.uint8)
        q = np.empty((end - start, args.num_samples), dtype=np.float32)
        for batch_start in range(start, end, queries_per_pass):
            batch_end = min(batch_start + queries_per_pass, end)
            torch.manual_seed(args.seed * 1_000_003 + batch_start)
            batch = make_batch(data, metadata, batch_start, batch_end)
            batch_preds, batch_q = sample(model, batch, args.num_samples, device, args.latent_noise, args.init_noise,
                                          args.symmetry, sat_vars)
            preds[batch_start - start:batch_end - start] = batch_preds.cpu().numpy()
            q[batch_start - start:batch_end - start] = batch_q.cpu().numpy()

        tmp_path = f"{shard_path(shard)}.tmp{rank}"
        with open(tmp_path, "wb") as f:
            np.savez(f, query_start=start, preds=preds, q=q)
        os.replace(tmp_path, shard_path(shard))
        print(f"[rank {rank}] shard {shard} (queries {start}-{end - 1}) saved, {done}/{len(todo)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Dataset directory; its test split is sampled (see --split)")
    parser.add_argument("--split", default="test",
                        help="Split(s) to sample, comma-separated: test, train (e.g. to look for solutions held out of "
                             "training), or both (test,train); the model is loaded once for all of them")
    parser.add_argument("--out", required=True,
                        help="Output directory; with several splits it must contain {split}")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0, help="Per-cell sampling temperature (TRM); 0 = argmax")
    parser.add_argument("--shard_size", type=int, default=512, help="Test queries per shard")
    parser.add_argument("--effective_batch_size", type=int, default=None,
                        help=f"Queries x samples per forward pass (default: {TOKENS_PER_PASS:,} tokens / sequence length)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--halt_steps", type=int, default=None, help="Recursion steps per sample (default: as trained)")
    parser.add_argument("--latent_noise", type=float, default=0.0, help="Scale of Gaussian noise added to z_L per step")
    parser.add_argument("--init_noise", type=float, default=0.0, help="Scale of Gaussian noise added to the initial latent")
    parser.add_argument("--symmetry", action="store_true", help="Sample j decodes the board under the j-th of 8 symmetries")
    parser.add_argument("--sat_views", action="store_true",
                        help="3-SAT: samples after the first decode the formula with variables renamed and negated, clauses reordered")
    parser.add_argument("--max_queries", type=int, default=None, help="Sample only the first queries (for quick checks)")
    parser.add_argument("--max_train_queries", type=int, default=None,
                        help="As --max_queries, for the train split only (e.g. 2000 of 100k training inputs)")
    args = parser.parse_args()
    splits = [s for s in args.split.split(",") if s]
    if len(splits) > 1 and "{split}" not in args.out:
        parser.error("with several splits, --out must contain {split}")

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    loaded = {}

    def get_model(metadata):
        # Splits of one dataset share vocabulary and sequence length, so one model serves them all
        if "model" not in loaded:
            model, step = load_model(args.checkpoint, metadata, args.temperature, device)
            if args.halt_steps is not None:
                model.model.config.halt_max_steps = args.halt_steps
            print(f"[rank {rank}] Loaded {args.checkpoint} (step {step}) on {device}", flush=True)
            loaded["model"] = model
        return loaded["model"]

    for split in splits:
        max_queries = args.max_train_queries if split == "train" and args.max_train_queries else args.max_queries
        sample_split(args, split, args.out.replace("{split}", split), max_queries, rank, world_size, device, get_model)


if __name__ == "__main__":
    main()
