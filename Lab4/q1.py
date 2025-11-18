# build_language_models.py
from datasets import load_dataset
from collections import Counter
import re

# -------------------------------
# 1. Load Dataset
# -------------------------------
dataset = load_dataset(
    "ai4bharat/IndicCorpV2",
    name="indiccorp_v2",
    split="guj_Gujr",
    streaming=True
)

sentences = []
for i, row in enumerate(dataset):
    text = row["text"].strip()
    if text:
        sentences.append(text)
    if i >= 10000:
        break

print(f"Loaded {len(sentences)} sentences")

# -------------------------------
# 2. Tokenization
# -------------------------------
def tokenize(text):
    return re.findall(r'\w+', text.lower())

tokenized_sentences = [tokenize(s) for s in sentences]

print("Sample tokenized:", tokenized_sentences[:3])

# -------------------------------
# 3. Train / Val / Test split (Correct)
# -------------------------------
N = len(tokenized_sentences)
train_end = int(0.8 * N)
val_end = int(0.9 * N)

train_tokens = tokenized_sentences[:train_end]
val_tokens   = tokenized_sentences[train_end:val_end]
test_tokens  = tokenized_sentences[val_end:]

print("Train:", len(train_tokens))
print("Val:", len(val_tokens))
print("Test:", len(test_tokens))

# -------------------------------
# 4. Save to file
# -------------------------------
with open("tokenized_sentences.txt", "w", encoding="utf-8") as f:
    for sent in tokenized_sentences:
        f.write(" ".join(sent) + "\n")

print("Saved tokenized sentences.")

# -------------------------------
# 5. Build n-grams (correct sentence-level)
# -------------------------------
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# Build sentence-safe n-grams
unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()
quadgram_counts = Counter()

for sent in tokenized_sentences:
    unigram_counts.update(get_ngrams(sent, 1))
    bigram_counts.update(get_ngrams(sent, 2))
    trigram_counts.update(get_ngrams(sent, 3))
    quadgram_counts.update(get_ngrams(sent, 4))

# -------------------------------
# Summary
# -------------------------------
print(f"Unigrams:  {len(unigram_counts)}")
print(f"Bigrams:   {len(bigram_counts)}")
print(f"Trigrams:  {len(trigram_counts)}")
print(f"4-grams:   {len(quadgram_counts)}")
