import torch
import torch.nn.functional as F

def perplexity(logits: torch.Tensor, targets: torch.Tensor):
    """
    Compute perplexity given model logits and true token IDs.
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    batch, seq_len, vocab = logits.shape
    flat_logits = logits.view(-1, vocab)
    flat_targets = targets.view(-1)

    loss = F.cross_entropy(flat_logits, flat_targets, reduction="mean")
    return torch.exp(loss)
