from datasets import load_dataset
import re
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    name="indiccorp_v2",
    split="guj_Gujr",
    streaming=True
)

sentences = []
for i, row in enumerate(dataset):
    if row["text"].strip():
        sentences.append(row["text"].strip())
    if i >= 20000:
        break

print("Loaded:", len(sentences))

# -------------------------
# TOKENIZE
# -------------------------
def tokenize(s):
    return re.findall(r'\w+', s.lower())

tokenized = [tokenize(s) for s in sentences]

# -------------------------
# TRAIN / VAL / TEST SPLIT
# -------------------------
N = len(tokenized)
train_end = int(0.8*N)
val_end   = int(0.9*N)

train_tokens = tokenized[:train_end]
val_tokens   = tokenized[train_end:val_end]
test_tokens  = tokenized[val_end:]

print("Train:", len(train_tokens), 
      "Val:", len(val_tokens), 
      "Test:", len(test_tokens))


# ---------- KATZ BACKOFF 4-GRAM (SHORT VERSION) -----------
from collections import Counter
import math, heapq

# ===== 1. Build n-gram counts =====
def ngram_counts(sentences, n):
    c = Counter()
    for s in sentences:
        for i in range(len(s)-n+1):
            c[tuple(s[i:i+n])] += 1
    return c

uni = ngram_counts(train_tokens,1)
bi  = ngram_counts(train_tokens,2)
tri = ngram_counts(train_tokens,3)
quad = ngram_counts(train_tokens,4)
V = set([w[0] for w in uni])
D = 0.5  # discount

# ===== 2. Katz Backoff Probability =====
def katz_prob(hist3, w):
    c_hist = sum(quad.get(hist3+(x,),0) for x in V)
    c_hw = quad.get(hist3+(w,),0)

    if c_hist == 0:
        return (tri.get(hist3[1:]+(w,),0) + 1) / sum(tri.values())  # simple fallback

    if c_hw > 0:
        return (c_hw - D) / c_hist
    else:
        # leftover mass
        seen = [x for x in V if quad.get(hist3+(x,),0) > 0]
        leftover = 1 - sum((quad.get(hist3+(x,),0)-D)/c_hist for x in seen)
        # lower prob
        lower = (tri.get(hist3[1:]+(w,),0)+1) / sum(tri.values())
        return leftover * lower

# ===== 3. Greedy generation =====
def greedy_katz():
    hist = ("<s>","<s>","<s>")
    out = []
    for _ in range(30):
        best = max(V, key=lambda w: katz_prob(hist,w))
        if best == "</s>": break
        out.append(best)
        hist = (hist[1], hist[2], best)
    return " ".join(out)

# ===== 4. Beam search (beam=20) =====
def beam_katz():
    beams = [(0.0, ("<s>","<s>","<s>"), [])]
    for _ in range(30):
        new = []
        for logp, hist, sent in beams:
            for w in V:
                p = katz_prob(hist, w)
                if p > 0:
                    new.append((logp + math.log(p), 
                                (hist[1],hist[2],w),
                                sent + [w]))
        beams = heapq.nlargest(20, new)
        if not beams:
            break
    return " ".join(beams[0][2])

# ===== 5. Generate 100 sentences =====
katz_greedy_100 = [greedy_katz() for _ in range(100)]
katz_beam_100   = [beam_katz()   for _ in range(100)]

print("Katz Backoff Done.")
