import torch

def _ngram_counts(tokens, n):
    counts = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i+n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts

def bleu_score(preds: list[str], refs: list[str], max_n=4):
    """
    Compute a simple BLEU score for a batch of sentences.
    Args:
        preds: list of predicted strings
        refs: list of reference strings
        max_n: maximum n-gram order
    Returns:
        BLEU float (0.0–1.0)
    """
    assert len(preds) == len(refs)
    total_score = 0.0

    for p, r in zip(preds, refs):
        p_tokens = p.split()
        r_tokens = r.split()

        precision_scores = []
        for n in range(1, max_n + 1):
            p_counts = _ngram_counts(p_tokens, n)
            r_counts = _ngram_counts(r_tokens, n)

            overlap = 0
            for gram, pc in p_counts.items():
                rc = r_counts.get(gram, 0)
                overlap += min(pc, rc)

            precision = overlap / (sum(p_counts.values()) + 1e-9)
            precision_scores.append(precision)

        score = torch.exp(torch.mean(torch.log(torch.tensor(precision_scores))))
        total_score += score.item()

    return total_score / len(preds)
