from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_LABEL_ID = -100

def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def token_log_probs(logits, loss_type: str):
    if loss_type == "stablemax_cross_entropy":
        return log_stablemax(logits.to(torch.float64), dim=-1)
    return F.log_softmax(logits.to(torch.float32), dim=-1)


ONE_OF_MANY_METHODS = ("naive", "minloss", "iexplr", "selectr")


class TargetSelector(nn.Module):
    """G_phi of 1oML SelectR (https://arxiv.org/abs/2008.11990): scores each candidate target
    given the input and the current prediction. A small transformer over cells stands in for the
    paper's NLM; it sees the soft prediction rather than the argmax of a lagged model copy."""

    def __init__(self, vocab_size: int, seq_len: int, hidden_size: int, num_layers: int, num_heads: int):
        super().__init__()
        self.embed_inputs = nn.Embedding(vocab_size, hidden_size)
        self.embed_targets = nn.Embedding(vocab_size, hidden_size)
        self.proj_preds = nn.Linear(vocab_size, hidden_size)
        self.embed_pos = nn.Parameter(torch.randn(seq_len, hidden_size) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden_size, num_heads, 2 * hidden_size, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.score_head = nn.Linear(hidden_size, 1)

        # Training steps taken, to time the switch from MinLoss targets to selected targets
        self.steps = nn.Buffer(torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, pred_probs: torch.Tensor) -> torch.Tensor:
        h = self.embed_inputs(inputs) + self.embed_targets(targets) + self.proj_preds(pred_probs) + self.embed_pos
        return self.score_head(self.encoder(h).mean(1)).squeeze(-1)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, kl_coef: float = 0.0, kl_balance_alpha: Optional[float] = None, free_bits: Optional[float] = None, value_coef: float = 1.0, data_path: Optional[str] = None,
                 one_of_many: str = "naive", selectr_pretrain_steps: int = 250, selectr_hidden_size: int = 128, selectr_layers: int = 2, selectr_heads: int = 4):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.loss_fn = globals()[loss_type]
        self.kl_coef = kl_coef
        self.kl_balance_alpha = kl_balance_alpha  # If not None, use KL balancing (e.g., 0.8)
        self.free_bits = free_bits  # If not None, apply free bits to KL (e.g., 0.1)
        self.value_coef = value_coef  # Coefficient for value loss

        # Detect task type from data_path for task-specific verification
        self.is_nqueen = False
        self.is_graph_coloring = False
        self.is_sudoku = False
        if data_path is not None:
            data_path_lower = data_path.lower()
            self.is_nqueen = 'nqueen' in data_path_lower
            self.is_graph_coloring = 'graph-coloring' in data_path_lower or 'graph_coloring' in data_path_lower
            self.is_sudoku = 'sudoku' in data_path_lower

        # One-of-many learning (1oML): how to pick the training target among an input's solutions.
        # naive uses `labels` as given; the others need `label_set` from a --one-of-many dataset.
        assert one_of_many in ONE_OF_MANY_METHODS, f"one_of_many must be one of {ONE_OF_MANY_METHODS}, got {one_of_many}"
        self.one_of_many = one_of_many
        self.selectr_pretrain_steps = selectr_pretrain_steps
        if one_of_many == "selectr":
            self.selector = TargetSelector(model.config.vocab_size, model.config.seq_len,
                                           selectr_hidden_size, selectr_layers, selectr_heads)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def target_set_losses(self, logits: torch.Tensor, label_set: torch.Tensor):
        """Mean token loss of the prediction against every candidate target.

        Returns (B, K) losses, (B, K) token counts, and a (B, K) mask of real targets."""
        logprobs = token_log_probs(logits, self.loss_type)
        mask = label_set != IGNORE_LABEL_ID
        index = torch.where(mask, label_set, 0).to(torch.long).unsqueeze(-1)
        target_logprobs = torch.gather(logprobs.unsqueeze(1).expand(-1, label_set.shape[1], -1, -1), -1, index).squeeze(-1)
        counts = mask.sum(-1)
        losses = -torch.where(mask, target_logprobs, 0).sum(-1) / counts.clamp_min(1)
        return losses, counts, counts > 0

    @torch.compiler.disable
    def selectr_weights(self, inputs, logits, preds, label_set, target_valid, minloss_weights):
        """SelectR: G_phi picks the target distribution, trained to maximise the expected reward
        (fraction of cells where the current prediction agrees with the target).

        Returns target weights for the prediction loss, the selector loss, and its metrics."""
        has_target = target_valid.any(-1)
        b_idx, k_idx = target_valid.nonzero(as_tuple=True)

        scores = self.selector(inputs[b_idx].to(torch.long),
                               label_set[b_idx, k_idx].clamp_min(0).to(torch.long),
                               torch.softmax(logits[b_idx].detach().to(torch.float32), dim=-1))
        all_scores = torch.full(target_valid.shape, -torch.inf, dtype=scores.dtype, device=scores.device)
        all_scores = all_scores.index_put((b_idx, k_idx), scores)
        probs = torch.softmax(torch.where(has_target.unsqueeze(-1), all_scores, 0), dim=-1) * target_valid

        target_mask = label_set != IGNORE_LABEL_ID
        reward = ((preds.unsqueeze(1) == label_set) & target_mask).sum(-1) / target_mask.sum(-1).clamp_min(1)
        # Expected (1 - reward): same gradient as maximising reward, since probs sum to 1 per input
        selector_loss = (probs * (1 - reward)).sum()

        # Train the prediction network on MinLoss targets until G_phi has warmed up, then on the
        # expected loss under G_phi's distribution (as in the paper's experiments)
        if self.training:
            self.selector.steps += 1
        warm = self.selector.steps > self.selectr_pretrain_steps
        weights = torch.where(warm, probs.detach(), minloss_weights)

        metrics = {
            "selectr_reward": (probs * reward).sum(-1).detach(),
            "selectr_explore": (probs.argmax(-1) != minloss_weights.argmax(-1)) & has_target,
        }
        return weights, selector_loss, metrics

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # One-of-many: pick the target(s) to train on among the input's solutions
        target_weights = None
        selector_loss = 0
        selector_metrics = {}
        if self.one_of_many != "naive":
            assert "post_mu" not in outputs, "one_of_many requires a model that does not condition on labels (e.g. TRM, not GRAM)"
            assert "label_set" in new_carry.current_data, "one_of_many needs a dataset built with --one-of-many (label_set)"
            label_set = new_carry.current_data["label_set"]
            target_losses, target_counts, target_valid = self.target_set_losses(outputs["logits"], label_set)

            with torch.no_grad():
                minloss_weights = F.one_hot(torch.where(target_valid, target_losses, torch.inf).argmin(-1), label_set.shape[1]).to(target_losses.dtype)
                if not self.training or self.one_of_many == "minloss":
                    target_weights = minloss_weights
                elif self.one_of_many == "iexplr":
                    # Sample a target with the probability the model assigns to it (Gumbel-max)
                    log_likelihood = torch.where(target_valid, -target_losses * target_counts, -torch.inf)
                    log_likelihood = torch.where(target_valid.any(-1, keepdim=True), log_likelihood, 0)
                    gumbel = -torch.log(-torch.log(torch.rand_like(log_likelihood).clamp(1e-20, 1)))
                    target_weights = F.one_hot((log_likelihood + gumbel).argmax(-1), label_set.shape[1]).to(target_losses.dtype)

            if target_weights is None:
                target_weights, selector_loss, selector_metrics = self.selectr_weights(
                    new_carry.current_data["inputs"], outputs["logits"], torch.argmax(outputs["logits"], dim=-1).detach(),
                    label_set, target_valid, minloss_weights)

            # Token-level metrics use the most-weighted target
            labels = torch.gather(label_set, 1, target_weights.argmax(-1).view(-1, 1, 1).expand(-1, 1, label_set.shape[-1])).squeeze(1)

        with torch.no_grad():
            # Preds - sample if temperature > 0 during inference, else greedy
            temperature = getattr(self.model.config, 'temperature', 0.0)
            if not self.model.training and temperature > 0.0:
                probs = torch.softmax(outputs["logits"], dim=-1)
                outputs["preds"] = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[:-1])
            else:
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            # Get predictions
            preds = outputs["preds"]
            
            # Compute sequence-level correctness with task-specific verification
            # For N-Queens and Graph Coloring: verify constraints instead of label matching
            if self.is_nqueen:
                from evaluators.nqueens_utils import batch_verify_nqueens
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_nqueens(inputs, preds)
            elif self.is_graph_coloring:
                from evaluators.graph_coloring_utils import batch_verify_coloring
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_coloring(inputs, preds)
            elif self.is_sudoku:
                from evaluators.sudoku_utils import batch_verify_sudoku
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_sudoku(inputs, preds)
            else:
                # Fallback to label matching for other tasks
                is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Token-level accuracy (always computed for metrics)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

            # Accuracy split by queries with one solution (os) vs many (ms), as reported in 1oML
            if "num_solutions" in new_carry.current_data:
                num_solutions = new_carry.current_data["num_solutions"]
                for suffix, subset in (("_os", num_solutions == 1), ("_ms", num_solutions > 1)):
                    metrics["count" + suffix] = (valid_metrics & subset).sum()
                    metrics["exact_accuracy" + suffix] = (valid_metrics & subset & seq_is_correct).sum()

            for k, v in selector_metrics.items():
                metrics[k] = torch.where(valid_metrics, v, 0).sum()

        # Losses
        if target_weights is None:
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        else:
            lm_loss = (target_weights * target_losses).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        if torch.is_tensor(selector_loss):
            metrics["selectr_loss"] = selector_loss.detach()
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # KL divergence (if variational)
        kl_loss = 0
        if "post_mu" in outputs and "prior_mu" in outputs:
            # Skip KL computation if fix_std is enabled (logvar ≈ -1e10)
            # This avoids division by exp(-1e10) ≈ 0 which causes NaN
            if outputs["prior_logvar"].min() < -1e5:
                kl_loss = torch.tensor(0.0, device=outputs["prior_mu"].device)
                metrics["kl_loss"] = kl_loss.detach()
            else:
                def compute_kl(post_mu, post_logvar, prior_mu, prior_logvar):
                    return torch.sum(0.5 * (
                        prior_logvar - post_logvar - 1
                        + torch.exp(post_logvar - prior_logvar)
                        + ((post_mu - prior_mu) ** 2) / torch.exp(prior_logvar)
                    ))

                if self.kl_balance_alpha is not None:
                    # KL balancing: alpha * KL(sg(post) || prior) + (1-alpha) * KL(post || sg(prior))
                    alpha = self.kl_balance_alpha
                    kl_loss = (
                        alpha * compute_kl(outputs["post_mu"].detach(), outputs["post_logvar"].detach(),
                                          outputs["prior_mu"], outputs["prior_logvar"])
                        + (1 - alpha) * compute_kl(outputs["post_mu"], outputs["post_logvar"],
                                                   outputs["prior_mu"].detach(), outputs["prior_logvar"].detach())
                    )
                else:
                    # Standard KL
                    kl_loss = compute_kl(outputs["post_mu"], outputs["post_logvar"],
                                        outputs["prior_mu"], outputs["prior_logvar"])
                # Apply free bits if specified (per sample)
                if self.free_bits is not None:
                    kl_loss = torch.clamp(kl_loss, min=self.free_bits)
                metrics["kl_loss"] = kl_loss.detach()

        # Value loss (if value head is enabled)
        value_loss = 0
        if "value_pred" in outputs:
            # Compute current accuracy (what the value should predict)
            current_accuracy = seq_is_correct.float() if self.is_sudoku else (is_correct.sum(-1).float() / loss_counts.clamp_min(1).float())
            # MSE loss between predicted value and current accuracy
            value_loss = F.mse_loss(outputs["value_pred"], current_accuracy, reduction="sum")
            metrics["value_loss"] = value_loss.detach()
            metrics["value_mae"] = (outputs["value_pred"] - current_accuracy).abs().mean().detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + self.kl_coef * kl_loss + self.value_coef * value_loss + selector_loss, metrics, detached_outputs, new_carry.halted.all()

