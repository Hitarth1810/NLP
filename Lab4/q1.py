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
    if i >= 10000:  # Adjust this for larger training data
        break
print(f"Loaded {len(sentences)} sentences")

# -------------------------------
# 2. Tokenization
# -------------------------------
def tokenize(text):
    return re.findall(r'\w+', text.lower())

tokenized_sentences = [tokenize(s) for s in sentences]

# ✅ Save all tokenized sentences to a text file
output_file = "tokenized_sentences.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for tokens in tokenized_sentences:
        f.write(" ".join(tokens) + "\n")  # join tokens with spaces

print(f"\n✅ Saved {len(tokenized_sentences)} tokenized sentences to '{output_file}'")

# Optional: show a few examples
print("\nSample tokenized sentences:")
for i, tokens in enumerate(tokenized_sentences[:5]):
    print(f"{i+1}: {tokens}")

# -------------------------------
# 3. Build N-grams
# -------------------------------
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

unigram_counts = Counter(get_ngrams(all_tokens, 1))
bigram_counts = Counter(get_ngrams(all_tokens, 2))
trigram_counts = Counter(get_ngrams(all_tokens, 3))
quadgram_counts = Counter(get_ngrams(all_tokens, 4))

print(f"Unigrams: {len(unigram_counts)}")
print(f"Bigrams: {len(bigram_counts)}")
print(f"Trigrams: {len(trigram_counts)}")
print(f"Quadgrams: {len(quadgram_counts)}")
