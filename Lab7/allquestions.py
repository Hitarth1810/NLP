from datasets import load_dataset
from collections import Counter
import math, re

# -------------------------
# LOAD DATA
# -------------------------
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

# -------------------------
# FUNCTIONS
# -------------------------
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def count_ngrams(sentences):
    uni = Counter()
    bi  = Counter()
    for sent in sentences:
        uni.update(get_ngrams(sent, 1))
        bi.update(get_ngrams(sent, 2))
    return uni, bi

# -------------------------
# 1. TRAIN UNIGRAM + BIGRAM
# -------------------------
train_uni, train_bi = count_ngrams(train_tokens)
total_unigrams = sum(train_uni.values())

print("Unigrams:", len(train_uni))
print("Bigrams:", len(train_bi))

# -------------------------
# PMI for val/test
# -------------------------
def compute_pmi_sentence(tokens):
    out = []
    for (w1,w2) in get_ngrams(tokens, 2):
        c_xy = train_bi.get((w1,w2), 0) + 1
        c_x  = train_uni.get((w1,), 0) + 1
        c_y  = train_uni.get((w2,), 0) + 1

        p_xy = c_xy / total_unigrams
        p_x  = c_x  / total_unigrams
        p_y  = c_y  / total_unigrams

        pmi = math.log2(p_xy / (p_x*p_y))
        out.append(((w1,w2), pmi))
    return out

val_pmi  = [compute_pmi_sentence(s) for s in val_tokens]
test_pmi = [compute_pmi_sentence(s) for s in test_tokens]

print("PMI done.")

# -------------------------
# TF-IDF
# -------------------------
def compute_idf(sentences):
    N = len(sentences)
    df = Counter()
    for s in sentences:
        df.update(set(s))
    idf = {w: math.log(N/df[w]) for w in df}
    return idf

idf = compute_idf(train_tokens)

def tfidf(sentences, idf):
    vecs = []
    for s in sentences:
        tf = Counter(s)
        vec = {w: tf[w]*idf.get(w, 0) for w in tf}
        vecs.append(vec)
    return vecs

train_tfidf = tfidf(train_tokens, idf)
val_tfidf   = tfidf(val_tokens, idf)
test_tfidf  = tfidf(test_tokens, idf)

print("TF-IDF done.")
