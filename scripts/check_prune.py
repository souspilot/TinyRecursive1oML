"""Temporary: print results that must not change when the repo is pruned. Run on the backup and on the pruned repo
and compare the output. Delete this file once they match.

Usage: python scripts/check_prune.py <repo> <run_checkpoint.pt> <data_dir_1oml>   (on a GPU)"""
import hashlib
import inspect
import os
import sys

repo, checkpoint_path, data_path = [os.path.abspath(a) for a in sys.argv[1:4]]
os.environ["DISABLE_COMPILE"] = "1"
os.chdir(repo)
sys.path[0] = repo

import torch  # noqa: E402
import sample_predictions as sp  # noqa: E402
from utils.functions import load_model_class  # noqa: E402

device = "cuda"
data, metadata = sp.load_split(data_path, "test")
train, _ = sp.load_split(data_path, "train")
extra = (0.0,) if "temperature" in inspect.signature(sp.load_model).parameters else ()
model, step = sp.load_model(checkpoint_path, metadata, *extra, device)
print("step", step)

# 1. Sampling a finished run: same weights, same draws
torch.manual_seed(0)
preds, q = sp.sample(model, sp.make_batch(data, metadata, 0, 16), 4, device)
print("samples", hashlib.sha256(preds.cpu().numpy().tobytes()).hexdigest()[:16], f"{q.sum().item():.4f}")


def losses(net, batch, name):
    """Four training steps' losses from fixed randomness."""
    batch = {k: v.to(device) for k, v in batch.items()}
    torch.manual_seed(1)
    with torch.device(device):
        carry = net.initial_carry(batch)
    for i in range(4):
        carry, loss, metrics, _, _ = net(carry=carry, batch=batch, return_keys=set())
        print(name, i, f"{loss.item():.5f}", f"{metrics.get('kl_loss', torch.zeros(1)).item():.5f}", int(carry.halted.sum()))


# 2. The best-of-K objective, on the trained weights
config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
size, k = config.global_batch_size, config.rollouts
batch = sp.make_batch(train, metadata, 0, size // k)
model.train()
losses(model, {n: v.repeat_interleave(k, dim=0) for n, v in batch.items()}, "bestofk")

# 3. GRAM's objective (posterior, KL balancing), from fixed initial weights
model_config = dict(config.arch.__pydantic_extra__, batch_size=size, vocab_size=metadata.vocab_size, seq_len=metadata.seq_len,
                    num_puzzle_identifiers=metadata.num_puzzle_identifiers, causal=False,
                    train_latent="posterior", prior_noise="learned")
loss_config = {n: v for n, v in config.arch.loss.__pydantic_extra__.items() if n != "kl_budget"}
loss_config.update(one_of_many="naive", kl_coef=0.1, kl_balance_alpha=0.8)
torch.manual_seed(2)
with torch.device(device):
    gram = load_model_class(config.arch.loss.name)(load_model_class(config.arch.name)(model_config), **loss_config)
gram.train()
losses(gram, sp.make_batch(train, metadata, 0, size), "gram")
