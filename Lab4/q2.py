# q2_smoothing.py
import math
from collections import Counter

# -------------------------------
# Smoothing Probability Function
# -------------------------------
def smoothed_prob(ngram, ngram_counts, lower_order_counts, vocab_size, smoothing="add1", k=1.0):
    """
    ngram: tuple of tokens (length n)
    ngram_counts: Counter of n-grams
    lower_order_counts: Counter of (n-1)-grams
    vocab_size: number of unique tokens (|V|)
    smoothing: "add1", "addk", or "addtype"
    k: constant for add-k smoothing
    """
    count_ngram = ngram_counts[ngram]

    if len(ngram) == 1:
        # For unigrams, denominator = total tokens
        denominator = sum(ngram_counts.values())
    else:
        history = ngram[:-1]
        denominator = lower_order_counts[history]

    if smoothing == "add1":  # Laplace smoothing
        numerator = count_ngram + 1
        denominator = denominator + vocab_size
    elif smoothing == "addk":
        numerator = count_ngram + k
        denominator = denominator + k * vocab_size
    elif smoothing == "addtype":
        numerator = count_ngram + vocab_size
        # denominator unchanged → not a true distribution
    else:
        raise ValueError("Unknown smoothing method")

    if denominator == 0:
        return 1e-12  # avoid div by zero
    return numerator / denominator
