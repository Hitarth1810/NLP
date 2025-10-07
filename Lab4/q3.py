# q3_sentence_prob.py
import random
import math
from collections import Counter
from q2 import smoothed_prob   # import from Q2

# -------------------------------
# Sentence Probability
# -------------------------------
def sentence_probability(sentence_tokens, n, ngram_counts, lower_counts, vocab_size, smoothing="add1", k=1.0):
    if len(sentence_tokens) < n:
        return 0.0

    ngrams = [tuple(sentence_tokens[i:i+n]) for i in range(len(sentence_tokens)-n+1)]
    log_prob = 0.0
    for ng in ngrams:
        p = smoothed_prob(ng, ngram_counts, lower_counts, vocab_size, smoothing, k)
        log_prob += math.log(p)
    return math.exp(log_prob)  # convert back from log-sum

# -------------------------------
# Example: Evaluate on 1000 sentences
# -------------------------------
def evaluate_models(sentences, tokenize,
                    unigram_counts, bigram_counts, trigram_counts, quadgram_counts):
    random.seed(42)
    test_sentences = random.sample(sentences, 1000)

    vocab_size = len(unigram_counts)

    for s in test_sentences[:5]:  # preview first 5
        tokens = tokenize(s)

        prob_uni = sentence_probability(tokens, 1, unigram_counts, None, vocab_size, smoothing="add1")
        prob_bi = sentence_probability(tokens, 2, bigram_counts, unigram_counts, vocab_size, smoothing="add1")
        prob_tri = sentence_probability(tokens, 3, trigram_counts, bigram_counts, vocab_size, smoothing="add1")
        prob_quad = sentence_probability(tokens, 4, quadgram_counts, trigram_counts, vocab_size, smoothing="add1")

        print("\nSentence:", s)
        print("Unigram prob (add-1):", prob_uni)
        print("Bigram prob (add-1):", prob_bi)
        print("Trigram prob (add-1):", prob_tri)
        print("Quadgram prob (add-1):", prob_quad)
