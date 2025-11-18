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
# ---------- KNESER-NEY 4-GRAM (SHORT VERSION) ----------
from collections import Counter
import math, heapq

# ===== 1. N-gram counts =====
uni = Counter(tuple(s[i:i+1]) for s in train_tokens for i in range(len(s)-0))
bi  = Counter(tuple(s[i:i+2]) for s in train_tokens for i in range(len(s)-1))
tri = Counter(tuple(s[i:i+3]) for s in train_tokens for i in range(len(s)-2))
quad = Counter(tuple(s[i:i+4]) for s in train_tokens for i in range(len(s)-3))
V = set(w[0] for w in uni)
D = 0.75

# Precompute continuation count (KN unigram)
cont = Counter()
for (w1,w2),_ in bi.items():
    cont[w2] += 1
total_cont = sum(cont.values())

def kn_unigram(w):
    return cont[w]/total_cont if w in cont else 1/total_cont

# ===== 2. Recursive KN probability =====
def p_kn(hist, w):
    n = len(hist)+1
    if n == 1:
        return kn_unigram(w)

    if n == 4:
        c_hw = quad.get(hist+(w,),0)
        c_h  = sum(quad.get(hist+(x,),0) for x in V)
    elif n == 3:
        c_hw = tri.get(hist+(w,),0)
        c_h  = sum(tri.get(hist+(x,),0) for x in V)
    else:
        c_hw = bi.get(hist+(w,),0)
        c_h  = sum(bi.get(hist+(x,),0) for x in V)

    # First term
    first = max(c_hw - D, 0) / c_h if c_h > 0 else 0

    # Lambda (discount mass)
    N_h = len({x for x in V if (hist+(x,)) in quad}) if n == 4 else 1
    lamb = D * N_h / c_h if c_h > 0 else 0

    # Lower order
    lower = p_kn(hist[1:], w)
    return first + lamb * lower

# ===== 3. Greedy generator =====
def greedy_kn():
    hist = ("<s>","<s>","<s>")
    out=[]
    for _ in range(30):
        best = max(V, key=lambda w: p_kn(hist,w))
        if best=="</s>": break
        out.append(best)
        hist = (hist[1], hist[2], best)
    return " ".join(out)

# ===== 4. Beam Search =====
def beam_kn():
    beams=[(0.0,("<s>","<s>","<s>"),[])]
    for _ in range(30):
        new=[]
        for logp,hist,sent in beams:
            for w in V:
                p=p_kn(hist,w)
                if p>0:
                    new.append((logp+math.log(p),(hist[1],hist[2],w),sent+[w]))
        beams=heapq.nlargest(20,new)
    return " ".join(beams[0][2])

# ===== 5. Generate 100 sentences =====
kn_greedy_100=[greedy_kn() for _ in range(100)]
kn_beam_100=[beam_kn() for _ in range(100)]

print("Kneser-Ney Done.")
