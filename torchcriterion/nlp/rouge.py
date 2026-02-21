import torch

def rouge_l_score(pred: str, ref: str):
    """
    Compute ROUGE-L between two sentences
    (Longest Common Subsequence based).
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()

    m, n = len(pred_tokens), len(ref_tokens)

    # DP LCS
    dp = torch.zeros((m+1, n+1), dtype=torch.int32)
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == ref_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    lcs = dp[m][n].item()

    precision = lcs / (m + 1e-9)
    recall = lcs / (n + 1e-9)
    f_score = (2 * precision * recall) / (precision + recall + 1e-9)

    return f_score

def rouge_l_batch(preds: list[str], refs: list[str]):
    return sum(rouge_l_score(p, r) for p, r in zip(preds, refs)) / len(preds)
